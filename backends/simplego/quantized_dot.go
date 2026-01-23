// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

func init() {
	setNodeExecutor(backends.OpTypeQuantizedDot, priorityGeneric, execQuantizedDot)
	setNodeExecutor(backends.OpTypeQuantizedDotInt8, priorityGeneric, execQuantizedDotInt8)
}

// QuantType identifies the quantization method for QuantizedDot.
type QuantType int

const (
	// QuantTypeNF4 is 4-bit Normal Float quantization.
	QuantTypeNF4 QuantType = iota
	// QuantTypeInt4 is 4-bit symmetric integer quantization.
	QuantTypeInt4
)

// String returns the string representation of the QuantType.
func (q QuantType) String() string {
	switch q {
	case QuantTypeNF4:
		return "NF4"
	case QuantTypeInt4:
		return "Int4"
	default:
		return "Unknown"
	}
}

// quantizedDotNodeData holds metadata for QuantizedDot execution.
type quantizedDotNodeData struct {
	// Dimensions
	M, K, N   int // M=batch, K=inFeatures, N=outFeatures
	GroupSize int // Elements per scale group

	// QuantType for NF4/Int4 distinction
	QuantType QuantType
}

// EqualNodeData implements nodeDataComparable for quantizedDotNodeData.
func (q *quantizedDotNodeData) EqualNodeData(other nodeDataComparable) bool {
	o := other.(*quantizedDotNodeData)
	return q.M == o.M && q.K == o.K && q.N == o.N &&
		q.GroupSize == o.GroupSize && q.QuantType == o.QuantType
}

// quantizedDotInt8NodeData holds metadata for QuantizedDotInt8 execution.
type quantizedDotInt8NodeData struct {
	M, K, N   int
	GroupSize int
}

// EqualNodeData implements nodeDataComparable for quantizedDotInt8NodeData.
func (q *quantizedDotInt8NodeData) EqualNodeData(other nodeDataComparable) bool {
	o := other.(*quantizedDotInt8NodeData)
	return q.M == o.M && q.K == o.K && q.N == o.N && q.GroupSize == o.GroupSize
}

// Compile-time checks that node data types implement nodeDataComparable.
var (
	_ nodeDataComparable = (*quantizedDotNodeData)(nil)
	_ nodeDataComparable = (*quantizedDotInt8NodeData)(nil)
)

// QuantizedDot performs fused dequantization and matrix multiplication.
// This dequantizes weights on-the-fly during matmul, reducing memory usage from O(K*N) to O(K).
//
// Parameters:
//   - inputOp: [M, K] activation matrix (float32)
//   - packedWeightsOp: packed uint8 weights for NF4/Int4
//   - scalesOp: [K, numGroups] per-group scales (float32)
//   - M, K, N: dimensions (M=batch, K=inFeatures, N=outFeatures)
//   - groupSize: elements per scale group (along N dimension)
//   - quantType: QuantTypeNF4 or QuantTypeInt4
//
// Returns: [M, N] output matrix
func (f *Function) QuantizedDot(
	inputOp, packedWeightsOp, scalesOp backends.Value,
	M, K, N, groupSize int,
	quantType QuantType,
) (backends.Value, error) {
	inputs, err := f.verifyAndCastValues("QuantizedDot", inputOp, packedWeightsOp, scalesOp)
	if err != nil {
		return nil, err
	}
	input, packedWeights, scales := inputs[0], inputs[1], inputs[2]

	// Validate input types
	if input.shape.DType != dtypes.Float32 {
		return nil, errors.Errorf("QuantizedDot: input must be Float32, got %s", input.shape.DType)
	}
	if packedWeights.shape.DType != dtypes.Uint8 {
		return nil, errors.Errorf("QuantizedDot: packedWeights must be Uint8, got %s", packedWeights.shape.DType)
	}
	if scales.shape.DType != dtypes.Float32 {
		return nil, errors.Errorf("QuantizedDot: scales must be Float32, got %s", scales.shape.DType)
	}

	// Validate input shape
	if input.shape.Rank() != 2 {
		return nil, errors.Errorf("QuantizedDot: input must be 2D [M, K], got rank %d", input.shape.Rank())
	}
	if input.shape.Dimensions[0] != M || input.shape.Dimensions[1] != K {
		return nil, errors.Errorf("QuantizedDot: input shape mismatch, expected [%d, %d], got %v",
			M, K, input.shape.Dimensions)
	}

	// Validate groupSize (must be done before division)
	if groupSize <= 0 {
		return nil, errors.Errorf("QuantizedDot: groupSize must be positive, got %d", groupSize)
	}

	// Validate packedWeights shape (4-bit packing: 2 values per byte)
	expectedPackedSize := (K*N + 1) / 2
	if packedWeights.shape.Size() != expectedPackedSize {
		return nil, errors.Errorf("QuantizedDot: packedWeights size mismatch, expected %d bytes for [%d, %d] 4-bit weights, got %d",
			expectedPackedSize, K, N, packedWeights.shape.Size())
	}

	// Validate scales shape
	numGroups := (N + groupSize - 1) / groupSize
	if scales.shape.Rank() != 2 {
		return nil, errors.Errorf("QuantizedDot: scales must be 2D [K, numGroups], got rank %d", scales.shape.Rank())
	}
	if scales.shape.Dimensions[0] != K || scales.shape.Dimensions[1] != numGroups {
		return nil, errors.Errorf("QuantizedDot: scales shape mismatch, expected [%d, %d], got %v",
			K, numGroups, scales.shape.Dimensions)
	}

	// Create node data
	nodeData := &quantizedDotNodeData{
		M:         M,
		K:         K,
		N:         N,
		GroupSize: groupSize,
		QuantType: quantType,
	}

	// Output shape: [M, N]
	outputShape := shapes.Make(dtypes.Float32, M, N)
	node, _ := f.getOrCreateNode(backends.OpTypeQuantizedDot, outputShape, inputs, nodeData)
	return node, nil
}

// QuantizedDotInt8 performs fused dequantization and matrix multiplication for Int8 weights.
//
// Parameters:
//   - inputOp: [M, K] activation matrix (float32)
//   - quantizedWeightsOp: [K, N] int8 weights
//   - scalesOp: [K, numGroups] per-group scales (float32)
//   - M, K, N: dimensions
//   - groupSize: elements per scale group
//
// Returns: [M, N] output matrix
func (f *Function) QuantizedDotInt8(
	inputOp, quantizedWeightsOp, scalesOp backends.Value,
	M, K, N, groupSize int,
) (backends.Value, error) {
	inputs, err := f.verifyAndCastValues("QuantizedDotInt8", inputOp, quantizedWeightsOp, scalesOp)
	if err != nil {
		return nil, err
	}
	input, quantizedWeights, scales := inputs[0], inputs[1], inputs[2]

	// Validate input types
	if input.shape.DType != dtypes.Float32 {
		return nil, errors.Errorf("QuantizedDotInt8: input must be Float32, got %s", input.shape.DType)
	}
	if quantizedWeights.shape.DType != dtypes.Int8 {
		return nil, errors.Errorf("QuantizedDotInt8: quantizedWeights must be Int8, got %s", quantizedWeights.shape.DType)
	}
	if scales.shape.DType != dtypes.Float32 {
		return nil, errors.Errorf("QuantizedDotInt8: scales must be Float32, got %s", scales.shape.DType)
	}

	// Validate input shape
	if input.shape.Rank() != 2 {
		return nil, errors.Errorf("QuantizedDotInt8: input must be 2D [M, K], got rank %d", input.shape.Rank())
	}
	if input.shape.Dimensions[0] != M || input.shape.Dimensions[1] != K {
		return nil, errors.Errorf("QuantizedDotInt8: input shape mismatch, expected [%d, %d], got %v",
			M, K, input.shape.Dimensions)
	}

	// Validate groupSize (must be done before division)
	if groupSize <= 0 {
		return nil, errors.Errorf("QuantizedDotInt8: groupSize must be positive, got %d", groupSize)
	}

	// Validate quantizedWeights shape
	if quantizedWeights.shape.Rank() != 2 {
		return nil, errors.Errorf("QuantizedDotInt8: quantizedWeights must be 2D [K, N], got rank %d", quantizedWeights.shape.Rank())
	}
	if quantizedWeights.shape.Dimensions[0] != K || quantizedWeights.shape.Dimensions[1] != N {
		return nil, errors.Errorf("QuantizedDotInt8: quantizedWeights shape mismatch, expected [%d, %d], got %v",
			K, N, quantizedWeights.shape.Dimensions)
	}

	// Validate scales shape
	numGroups := (N + groupSize - 1) / groupSize
	if scales.shape.Rank() != 2 {
		return nil, errors.Errorf("QuantizedDotInt8: scales must be 2D [K, numGroups], got rank %d", scales.shape.Rank())
	}
	if scales.shape.Dimensions[0] != K || scales.shape.Dimensions[1] != numGroups {
		return nil, errors.Errorf("QuantizedDotInt8: scales shape mismatch, expected [%d, %d], got %v",
			K, numGroups, scales.shape.Dimensions)
	}

	// Create node data
	nodeData := &quantizedDotInt8NodeData{
		M:         M,
		K:         K,
		N:         N,
		GroupSize: groupSize,
	}

	// Output shape: [M, N]
	outputShape := shapes.Make(dtypes.Float32, M, N)
	node, _ := f.getOrCreateNode(backends.OpTypeQuantizedDotInt8, outputShape, inputs, nodeData)
	return node, nil
}

// NF4Values are the 16 fixed values for 4-bit NormalFloat quantization.
// These are derived from the quantiles of a standard normal distribution.
var NF4Values = [16]float32{
	-1.0,
	-0.6961928009986877,
	-0.5250730514526367,
	-0.39491748809814453,
	-0.28444138169288635,
	-0.18477343022823334,
	-0.09105003625154495,
	0.0,
	0.07958029955625534,
	0.16093020141124725,
	0.24611230194568634,
	0.33791524171829224,
	0.44070982933044434,
	0.5626170039176941,
	0.7229568362236023,
	1.0,
}

// execQuantizedDot executes fused quantized matmul for NF4/Int4.
//
// Design: This implementation processes one output column at a time, dequantizing weights
// on-the-fly to minimize memory usage (O(K) for the weight column buffer instead of O(K*N)
// for fully dequantized weights). The trade-off is that output writes are strided (column-major),
// which is less cache-friendly than row-major access. For production workloads, consider:
// - Tiled/blocked execution for better cache locality
// - SIMD optimization for the inner loops
// - Multi-threading across output columns or rows
func execQuantizedDot(backend *Backend, node *Node, inputs []*Buffer, _ []bool) (*Buffer, error) {
	input := inputs[0]         // [M, K]
	packedWeights := inputs[1] // packed uint8
	scales := inputs[2]        // [K, numGroups]

	params := node.data.(*quantizedDotNodeData)
	M, K, N := params.M, params.K, params.N
	groupSize := params.GroupSize
	numGroups := (N + groupSize - 1) / groupSize

	// Get output buffer
	output := backend.getBufferForShape(node.shape)
	outputFlat := output.flat.([]float32)

	// Get input data
	inputFlat := input.flat.([]float32)
	packed := packedWeights.flat.([]uint8)
	scalesFlat := scales.flat.([]float32)

	// Tile buffer for one weight column - reused across iterations
	weightCol := make([]float32, K)

	// Process one output column at a time
	for n := 0; n < N; n++ {
		// Dequantize column n of weights (K elements)
		groupIdx := n / groupSize

		for k := 0; k < K; k++ {
			// Weight index in row-major [K, N] layout
			weightIdx := k*N + n
			packedIdx := weightIdx / 2

			var val float32
			switch params.QuantType {
			case QuantTypeNF4:
				var quantIdx int
				if weightIdx%2 == 0 {
					quantIdx = int(packed[packedIdx] & 0x0F)
				} else {
					quantIdx = int((packed[packedIdx] >> 4) & 0x0F)
				}
				val = NF4Values[quantIdx]
			case QuantTypeInt4:
				var unsignedVal int
				if weightIdx%2 == 0 {
					unsignedVal = int(packed[packedIdx] & 0x0F)
				} else {
					unsignedVal = int((packed[packedIdx] >> 4) & 0x0F)
				}
				val = float32(unsignedVal - 8)
			}

			// Apply per-group scale: scales[k, groupIdx]
			scale := scalesFlat[k*numGroups+groupIdx]
			weightCol[k] = val * scale
		}

		// Compute dot product for each input row with this weight column
		for m := 0; m < M; m++ {
			sum := float32(0)
			inputRowStart := m * K
			for k := 0; k < K; k++ {
				sum += inputFlat[inputRowStart+k] * weightCol[k]
			}
			outputFlat[m*N+n] = sum
		}
	}

	return output, nil
}

// execQuantizedDotInt8 executes fused quantized matmul for Int8.
// See execQuantizedDot for design notes on the column-wise processing approach.
func execQuantizedDotInt8(backend *Backend, node *Node, inputs []*Buffer, _ []bool) (*Buffer, error) {
	input := inputs[0]            // [M, K]
	quantizedWeights := inputs[1] // [K, N] int8
	scales := inputs[2]           // [K, numGroups]

	params := node.data.(*quantizedDotInt8NodeData)
	M, K, N := params.M, params.K, params.N
	groupSize := params.GroupSize
	numGroups := (N + groupSize - 1) / groupSize

	// Get output buffer
	output := backend.getBufferForShape(node.shape)
	outputFlat := output.flat.([]float32)

	// Get input data
	inputFlat := input.flat.([]float32)
	quantized := quantizedWeights.flat.([]int8)
	scalesFlat := scales.flat.([]float32)

	// Tile buffer for one weight column
	weightCol := make([]float32, K)

	for n := 0; n < N; n++ {
		// Dequantize column n of weights
		groupIdx := n / groupSize

		for k := 0; k < K; k++ {
			weightIdx := k*N + n
			val := float32(quantized[weightIdx])
			scale := scalesFlat[k*numGroups+groupIdx]
			weightCol[k] = val * scale
		}

		// Compute dot products
		for m := 0; m < M; m++ {
			sum := float32(0)
			inputRowStart := m * K
			for k := 0; k < K; k++ {
				sum += inputFlat[inputRowStart+k] * weightCol[k]
			}
			outputFlat[m*N+n] = sum
		}
	}

	return output, nil
}
