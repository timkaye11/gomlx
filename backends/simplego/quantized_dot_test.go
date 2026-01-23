// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"math"
	"math/rand"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
)

// TestQuantizedDotNF4Correctness verifies QuantizedDot produces correct results for NF4.
func TestQuantizedDotNF4Correctness(t *testing.T) {
	backendIface, err := New("")
	if err != nil {
		t.Fatalf("Failed to create backend: %v", err)
	}
	backend := backendIface.(*Backend)
	defer backend.Finalize()

	M, K, N := 8, 32, 64
	groupSize := 16
	numGroups := (N + groupSize - 1) / groupSize

	// Create random input
	input := make([]float32, M*K)
	for i := range input {
		input[i] = rand.Float32()*2 - 1
	}

	// Create random weights and quantize to NF4
	weights := make([]float32, K*N)
	for i := range weights {
		weights[i] = rand.Float32()*2 - 1
	}

	// Quantize weights to NF4
	packed := make([]uint8, (K*N+1)/2)
	scales := make([]float32, K*numGroups)

	for k := 0; k < K; k++ {
		for g := 0; g < numGroups; g++ {
			startCol := g * groupSize
			endCol := min(startCol+groupSize, N)

			// Find absmax for this group
			absmax := float32(0)
			for n := startCol; n < endCol; n++ {
				val := weights[k*N+n]
				if val < 0 {
					val = -val
				}
				if val > absmax {
					absmax = val
				}
			}
			scales[k*numGroups+g] = absmax

			// Quantize values
			for n := startCol; n < endCol; n++ {
				idx := k*N + n
				val := weights[idx]
				normalized := float32(0)
				if absmax > 0 {
					normalized = val / absmax
				}
				quantIdx := findNearestNF4Index(normalized)
				packedIdx := idx / 2
				if idx%2 == 0 {
					packed[packedIdx] = (packed[packedIdx] & 0xF0) | uint8(quantIdx)
				} else {
					packed[packedIdx] = (packed[packedIdx] & 0x0F) | (uint8(quantIdx) << 4)
				}
			}
		}
	}

	// Run QuantizedDot via executor directly
	params := &quantizedDotNodeData{
		M:         M,
		K:         K,
		N:         N,
		GroupSize: groupSize,
		QuantType: QuantTypeNF4,
	}

	// Create mock node
	node := &Node{
		data: params,
	}
	node.shape.DType = dtypes.Float32
	node.shape.Dimensions = []int{M, N}

	// Create input buffers
	inputBuf := &Buffer{flat: input}
	inputBuf.shape.DType = dtypes.Float32
	inputBuf.shape.Dimensions = []int{M, K}
	packedBuf := &Buffer{flat: packed}
	packedBuf.shape.DType = dtypes.Uint8
	packedBuf.shape.Dimensions = []int{len(packed)}
	scalesBuf := &Buffer{flat: scales}
	scalesBuf.shape.DType = dtypes.Float32
	scalesBuf.shape.Dimensions = []int{K, numGroups}

	// Execute
	output, err := execQuantizedDot(backend, node, []*Buffer{inputBuf, packedBuf, scalesBuf}, nil)
	if err != nil {
		t.Fatalf("execQuantizedDot failed: %v", err)
	}
	fusedOutput := output.flat.([]float32)

	// Compute naive result: dequantize + matmul
	dequantized := make([]float32, K*N)
	for k := 0; k < K; k++ {
		for n := 0; n < N; n++ {
			idx := k*N + n
			packedIdx := idx / 2
			var quantIdx int
			if idx%2 == 0 {
				quantIdx = int(packed[packedIdx] & 0x0F)
			} else {
				quantIdx = int((packed[packedIdx] >> 4) & 0x0F)
			}
			groupIdx := n / groupSize
			scale := scales[k*numGroups+groupIdx]
			dequantized[idx] = NF4Values[quantIdx] * scale
		}
	}

	naiveOutput := make([]float32, M*N)
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			sum := float32(0)
			for k := 0; k < K; k++ {
				sum += input[m*K+k] * dequantized[k*N+n]
			}
			naiveOutput[m*N+n] = sum
		}
	}

	// Compare results
	maxDiff := float32(0)
	for i := range fusedOutput {
		diff := float32(math.Abs(float64(fusedOutput[i] - naiveOutput[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	tolerance := float32(1e-5)
	if maxDiff > tolerance {
		t.Errorf("Max difference between fused and naive: %v (tolerance: %v)", maxDiff, tolerance)
	}
}

// TestQuantizedDotInt4Correctness verifies QuantizedDot produces correct results for Int4.
func TestQuantizedDotInt4Correctness(t *testing.T) {
	backendIface, err := New("")
	if err != nil {
		t.Fatalf("Failed to create backend: %v", err)
	}
	backend := backendIface.(*Backend)
	defer backend.Finalize()

	M, K, N := 8, 32, 64
	groupSize := 16
	numGroups := (N + groupSize - 1) / groupSize

	// Create random input
	input := make([]float32, M*K)
	for i := range input {
		input[i] = rand.Float32()*2 - 1
	}

	// Create random weights and quantize to Int4
	weights := make([]float32, K*N)
	for i := range weights {
		weights[i] = rand.Float32()*2 - 1
	}

	// Quantize weights to Int4
	packed := make([]uint8, (K*N+1)/2)
	scales := make([]float32, K*numGroups)

	for k := 0; k < K; k++ {
		for g := 0; g < numGroups; g++ {
			startCol := g * groupSize
			endCol := min(startCol+groupSize, N)

			// Find absmax for this group
			absmax := float32(0)
			for n := startCol; n < endCol; n++ {
				val := weights[k*N+n]
				if val < 0 {
					val = -val
				}
				if val > absmax {
					absmax = val
				}
			}

			scale := absmax / 7.0
			if scale == 0 {
				scale = 1.0
			}
			scales[k*numGroups+g] = scale

			// Quantize values
			for n := startCol; n < endCol; n++ {
				idx := k*N + n
				val := weights[idx]
				quantVal := int(math.Round(float64(val / scale)))
				if quantVal < -8 {
					quantVal = -8
				}
				if quantVal > 7 {
					quantVal = 7
				}
				unsignedVal := uint8(quantVal + 8)
				packedIdx := idx / 2
				if idx%2 == 0 {
					packed[packedIdx] = (packed[packedIdx] & 0xF0) | unsignedVal
				} else {
					packed[packedIdx] = (packed[packedIdx] & 0x0F) | (unsignedVal << 4)
				}
			}
		}
	}

	// Run QuantizedDot via executor
	params := &quantizedDotNodeData{
		M:         M,
		K:         K,
		N:         N,
		GroupSize: groupSize,
		QuantType: QuantTypeInt4,
	}

	node := &Node{data: params}
	node.shape.DType = dtypes.Float32
	node.shape.Dimensions = []int{M, N}

	inputBuf := &Buffer{flat: input}
	inputBuf.shape.DType = dtypes.Float32
	inputBuf.shape.Dimensions = []int{M, K}
	packedBuf := &Buffer{flat: packed}
	packedBuf.shape.DType = dtypes.Uint8
	packedBuf.shape.Dimensions = []int{len(packed)}
	scalesBuf := &Buffer{flat: scales}
	scalesBuf.shape.DType = dtypes.Float32
	scalesBuf.shape.Dimensions = []int{K, numGroups}

	output, err := execQuantizedDot(backend, node, []*Buffer{inputBuf, packedBuf, scalesBuf}, nil)
	if err != nil {
		t.Fatalf("execQuantizedDot failed: %v", err)
	}
	fusedOutput := output.flat.([]float32)

	// Compute naive result
	dequantized := make([]float32, K*N)
	for k := 0; k < K; k++ {
		for n := 0; n < N; n++ {
			idx := k*N + n
			packedIdx := idx / 2
			var unsignedVal int
			if idx%2 == 0 {
				unsignedVal = int(packed[packedIdx] & 0x0F)
			} else {
				unsignedVal = int((packed[packedIdx] >> 4) & 0x0F)
			}
			groupIdx := n / groupSize
			scale := scales[k*numGroups+groupIdx]
			dequantized[idx] = float32(unsignedVal-8) * scale
		}
	}

	naiveOutput := make([]float32, M*N)
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			sum := float32(0)
			for k := 0; k < K; k++ {
				sum += input[m*K+k] * dequantized[k*N+n]
			}
			naiveOutput[m*N+n] = sum
		}
	}

	maxDiff := float32(0)
	for i := range fusedOutput {
		diff := float32(math.Abs(float64(fusedOutput[i] - naiveOutput[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	tolerance := float32(1e-5)
	if maxDiff > tolerance {
		t.Errorf("Max difference between fused and naive: %v (tolerance: %v)", maxDiff, tolerance)
	}
}

// TestQuantizedDotInt8Correctness verifies QuantizedDotInt8 produces correct results.
func TestQuantizedDotInt8Correctness(t *testing.T) {
	backendIface, err := New("")
	if err != nil {
		t.Fatalf("Failed to create backend: %v", err)
	}
	backend := backendIface.(*Backend)
	defer backend.Finalize()

	M, K, N := 8, 32, 64
	groupSize := 16
	numGroups := (N + groupSize - 1) / groupSize

	// Create random input
	input := make([]float32, M*K)
	for i := range input {
		input[i] = rand.Float32()*2 - 1
	}

	// Create random weights and quantize to Int8
	weights := make([]float32, K*N)
	for i := range weights {
		weights[i] = rand.Float32()*2 - 1
	}

	// Quantize weights to Int8
	quantized := make([]int8, K*N)
	scales := make([]float32, K*numGroups)

	for k := 0; k < K; k++ {
		for g := 0; g < numGroups; g++ {
			startCol := g * groupSize
			endCol := min(startCol+groupSize, N)

			// Find absmax for this group
			absmax := float32(0)
			for n := startCol; n < endCol; n++ {
				val := weights[k*N+n]
				if val < 0 {
					val = -val
				}
				if val > absmax {
					absmax = val
				}
			}

			scale := absmax / 127.0
			if scale == 0 {
				scale = 1.0
			}
			scales[k*numGroups+g] = scale

			// Quantize values
			for n := startCol; n < endCol; n++ {
				idx := k*N + n
				val := weights[idx]
				quantVal := int(math.Round(float64(val / scale)))
				if quantVal < -128 {
					quantVal = -128
				}
				if quantVal > 127 {
					quantVal = 127
				}
				quantized[idx] = int8(quantVal)
			}
		}
	}

	// Run QuantizedDotInt8 via executor
	params := &quantizedDotInt8NodeData{
		M:         M,
		K:         K,
		N:         N,
		GroupSize: groupSize,
	}

	node := &Node{data: params}
	node.shape.DType = dtypes.Float32
	node.shape.Dimensions = []int{M, N}

	inputBuf := &Buffer{flat: input}
	inputBuf.shape.DType = dtypes.Float32
	inputBuf.shape.Dimensions = []int{M, K}
	quantizedBuf := &Buffer{flat: quantized}
	quantizedBuf.shape.DType = dtypes.Int8
	quantizedBuf.shape.Dimensions = []int{K, N}
	scalesBuf := &Buffer{flat: scales}
	scalesBuf.shape.DType = dtypes.Float32
	scalesBuf.shape.Dimensions = []int{K, numGroups}

	output, err := execQuantizedDotInt8(backend, node, []*Buffer{inputBuf, quantizedBuf, scalesBuf}, nil)
	if err != nil {
		t.Fatalf("execQuantizedDotInt8 failed: %v", err)
	}
	fusedOutput := output.flat.([]float32)

	// Compute naive result
	dequantized := make([]float32, K*N)
	for k := 0; k < K; k++ {
		for n := 0; n < N; n++ {
			idx := k*N + n
			groupIdx := n / groupSize
			scale := scales[k*numGroups+groupIdx]
			dequantized[idx] = float32(quantized[idx]) * scale
		}
	}

	naiveOutput := make([]float32, M*N)
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			sum := float32(0)
			for k := 0; k < K; k++ {
				sum += input[m*K+k] * dequantized[k*N+n]
			}
			naiveOutput[m*N+n] = sum
		}
	}

	maxDiff := float32(0)
	for i := range fusedOutput {
		diff := float32(math.Abs(float64(fusedOutput[i] - naiveOutput[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	tolerance := float32(1e-5)
	if maxDiff > tolerance {
		t.Errorf("Max difference between fused and naive: %v (tolerance: %v)", maxDiff, tolerance)
	}
}

// findNearestNF4Index finds the index of the nearest NF4 value.
func findNearestNF4Index(normalized float32) int {
	bestIdx := 0
	bestDist := float32(math.Abs(float64(normalized - NF4Values[0])))
	for i := 1; i < 16; i++ {
		dist := float32(math.Abs(float64(normalized - NF4Values[i])))
		if dist < bestDist {
			bestDist = dist
			bestIdx = i
		}
	}
	return bestIdx
}
