// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package quantization

// This file provides graph operations for on-the-fly dequantization during forward passes.
// The strategy is "dequantize-then-GEMM": we dequantize quantized weights to the compute dtype
// just before matrix multiplication, trading some compute for significant memory savings.

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// nf4LookupTable creates a constant node containing the NF4 lookup table values.
func nf4LookupTable(g *Graph, dtype dtypes.DType) *Node {
	// Convert NF4Values to the target dtype
	values := make([]float32, 16)
	copy(values, NF4Values[:])
	table := Const(g, values)
	if dtype != dtypes.Float32 {
		table = ConvertDType(table, dtype)
	}
	return table
}

// DequantizeNF4 converts packed NF4 weights to floating point.
//
// Parameters:
//   - packedWeights: Packed uint8 tensor containing NF4 indices [packedSize]
//   - scales: Per-group scale factors [rows, numGroups]
//   - rows, cols: Original tensor dimensions
//   - groupSize: Number of elements per group
//   - computeDType: Target floating point dtype (e.g., BFloat16, Float32)
//
// Returns: Dequantized tensor [rows, cols]
func DequantizeNF4(packedWeights, scales *Node, rows, cols, groupSize int, computeDType dtypes.DType) *Node {
	g := packedWeights.Graph()
	numGroups := (cols + groupSize - 1) / groupSize
	totalElements := rows * cols

	// Create NF4 lookup table
	lut := nf4LookupTable(g, computeDType)

	// Unpack 4-bit indices from packed bytes using the built-in Unpack function
	// This returns a tensor with dtype Uint4
	unpacked := Unpack(packedWeights, dtypes.Uint4)

	// Take only the elements we need (handles odd totalElements)
	if unpacked.Shape().Size() > totalElements {
		unpacked = Slice(unpacked, AxisRange(0, totalElements))
	}

	// Convert to int32 for Gather indexing
	indices := ConvertDType(unpacked, dtypes.Int32)

	// Gather NF4 values using indices as lookup
	// indices shape: [totalElements], lut shape: [16]
	// We need to reshape indices to [totalElements, 1] for Gather
	indicesReshaped := Reshape(indices, totalElements, 1)
	dequantized := Gather(lut, indicesReshaped) // [totalElements]

	// Reshape to [rows, cols]
	dequantized = Reshape(dequantized, rows, cols)

	// Prepare scales for broadcasting: [rows, numGroups] -> [rows, cols]
	// Repeat each scale value groupSize times along the column axis
	scaledOutput := applyPerGroupScales(dequantized, scales, rows, cols, numGroups, groupSize)

	return scaledOutput
}

// DequantizeInt4 converts packed Int4 weights to floating point.
//
// Int4 values are stored as unsigned [0, 15] and shifted back to signed [-8, 7].
//
// Parameters:
//   - packedWeights: Packed uint8 tensor containing Int4 values [packedSize]
//   - scales: Per-group scale factors [rows, numGroups]
//   - rows, cols: Original tensor dimensions
//   - groupSize: Number of elements per group
//   - computeDType: Target floating point dtype
//
// Returns: Dequantized tensor [rows, cols]
func DequantizeInt4(packedWeights, scales *Node, rows, cols, groupSize int, computeDType dtypes.DType) *Node {
	g := packedWeights.Graph()
	numGroups := (cols + groupSize - 1) / groupSize
	totalElements := rows * cols

	// Unpack 4-bit values from packed bytes
	unpacked := Unpack(packedWeights, dtypes.Uint4)

	// Take only the elements we need
	if unpacked.Shape().Size() > totalElements {
		unpacked = Slice(unpacked, AxisRange(0, totalElements))
	}

	// Convert from unsigned [0, 15] to signed [-8, 7]
	// First convert to compute dtype, then subtract 8
	values := ConvertDType(unpacked, computeDType)
	eight := Scalar(g, computeDType, 8.0)
	values = Sub(values, eight)

	// Reshape to [rows, cols]
	values = Reshape(values, rows, cols)

	// Apply per-group scales
	scaledOutput := applyPerGroupScales(values, scales, rows, cols, numGroups, groupSize)

	return scaledOutput
}

// DequantizeInt8 converts Int8 weights to floating point.
//
// Parameters:
//   - quantizedWeights: Int8 tensor [rows, cols]
//   - scales: Per-group scale factors [rows, numGroups]
//   - rows, cols: Original tensor dimensions
//   - groupSize: Number of elements per group
//   - computeDType: Target floating point dtype
//
// Returns: Dequantized tensor [rows, cols]
func DequantizeInt8(quantizedWeights, scales *Node, rows, cols, groupSize int, computeDType dtypes.DType) *Node {
	numGroups := (cols + groupSize - 1) / groupSize

	// Convert int8 to compute dtype
	values := ConvertDType(quantizedWeights, computeDType)

	// Apply per-group scales
	scaledOutput := applyPerGroupScales(values, scales, rows, cols, numGroups, groupSize)

	return scaledOutput
}

// applyPerGroupScales applies per-group scale factors to dequantized values.
// This expands scales from [rows, numGroups] to [rows, cols] by repeating.
func applyPerGroupScales(values, scales *Node, rows, cols, numGroups, groupSize int) *Node {
	g := values.Graph()
	computeDType := values.DType()

	// Ensure scales are the right dtype
	if scales.DType() != computeDType {
		scales = ConvertDType(scales, computeDType)
	}

	// If groupSize evenly divides cols, we can use simple broadcasting
	if cols == numGroups*groupSize {
		// Reshape scales to [rows, numGroups, 1] and tile to [rows, numGroups, groupSize]
		scalesExpanded := Reshape(scales, rows, numGroups, 1)
		// BroadcastToDims would be ideal, but we'll use tile/repeat pattern
		scalesTiled := BroadcastToDims(scalesExpanded, rows, numGroups, groupSize)
		scalesTiled = Reshape(scalesTiled, rows, cols)
		return Mul(values, scalesTiled)
	}

	// For irregular grouping (last group may be smaller), we need to handle edge cases
	// Build scale matrix by repeating each group's scale for groupSize elements
	// This handles the case where cols is not evenly divisible by groupSize

	// Create indices for gathering scales
	// Each column index maps to its group: colIdx / groupSize
	colIndices := make([]int, cols)
	for c := 0; c < cols; c++ {
		colIndices[c] = c / groupSize
	}
	groupIndices := Const(g, colIndices) // [cols]
	groupIndices = ConvertDType(groupIndices, dtypes.Int32)

	// For each row, gather the appropriate scale for each column
	// scales shape: [rows, numGroups]
	// We need: output[r, c] = scales[r, c/groupSize]

	// Reshape scales to [rows * numGroups] for flat indexing
	scalesFlat := Reshape(scales, rows*numGroups)

	// Create row offsets: [0, numGroups, 2*numGroups, ...]
	rowOffsets := make([]int, rows)
	for r := 0; r < rows; r++ {
		rowOffsets[r] = r * numGroups
	}
	rowOffsetsNode := Const(g, rowOffsets) // [rows]
	rowOffsetsNode = ConvertDType(rowOffsetsNode, dtypes.Int32)

	// Expand rowOffsets to [rows, 1] and groupIndices to [1, cols]
	// Then add to get [rows, cols] indices into scalesFlat
	rowOffsetsExpanded := Reshape(rowOffsetsNode, rows, 1)
	groupIndicesExpanded := Reshape(groupIndices, 1, cols)

	// Broadcast and add to get final indices
	rowOffsetsExpanded = BroadcastToDims(rowOffsetsExpanded, rows, cols)
	groupIndicesExpanded = BroadcastToDims(groupIndicesExpanded, rows, cols)
	flatIndices := Add(rowOffsetsExpanded, groupIndicesExpanded) // [rows, cols]

	// Flatten indices for gather
	flatIndices = Reshape(flatIndices, rows*cols, 1)
	gatheredScales := Gather(scalesFlat, flatIndices) // [rows*cols]
	gatheredScales = Reshape(gatheredScales, rows, cols)

	return Mul(values, gatheredScales)
}

// Dequantize converts quantized weights to floating point using the provided config.
// This is the main entry point for graph-based dequantization.
//
// Parameters:
//   - packedWeights: Packed quantized data
//   - scales: Per-group scale factors
//   - originalShape: The original tensor shape before quantization
//   - config: Quantization configuration
//
// Returns: Dequantized tensor with originalShape and config.ComputeDType
func Dequantize(packedWeights, scales *Node, originalShape shapes.Shape, config *Config) *Node {
	if originalShape.Rank() != 2 {
		panic("Dequantize only supports 2D tensors")
	}

	rows := originalShape.Dimensions[0]
	cols := originalShape.Dimensions[1]
	groupSize := config.GroupSize
	computeDType := config.ComputeDType

	switch config.QuantType {
	case NF4:
		return DequantizeNF4(packedWeights, scales, rows, cols, groupSize, computeDType)
	case Int4:
		return DequantizeInt4(packedWeights, scales, rows, cols, groupSize, computeDType)
	case Int8:
		return DequantizeInt8(packedWeights, scales, rows, cols, groupSize, computeDType)
	default:
		panic("unsupported quantization type")
	}
}

// QuantizedLinear performs a linear transformation with quantized weights.
// This is equivalent to: output = input @ Dequantize(quantizedWeights)
//
// Parameters:
//   - input: Input tensor [..., inFeatures]
//   - packedWeights: Packed quantized weight data
//   - scales: Per-group scale factors
//   - inFeatures, outFeatures: Weight matrix dimensions
//   - config: Quantization configuration
//
// Returns: Output tensor [..., outFeatures]
func QuantizedLinear(input, packedWeights, scales *Node, inFeatures, outFeatures int, config *Config) *Node {
	// Dequantize weights to compute dtype
	weightShape := shapes.Make(config.ComputeDType, inFeatures, outFeatures)
	weights := Dequantize(packedWeights, scales, weightShape, config)

	// Ensure input is the same dtype
	if input.DType() != config.ComputeDType {
		input = ConvertDType(input, config.ComputeDType)
	}

	// Perform matrix multiplication
	return Dot(input, weights)
}
