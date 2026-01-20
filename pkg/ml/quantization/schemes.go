// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package quantization

import (
	"math"
	"sort"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/pkg/errors"
)

// NF4Values are the 16 fixed values for 4-bit NormalFloat quantization.
// These are derived from the quantiles of a standard normal distribution,
// optimized for normally distributed neural network weights.
//
// Reference: "QLoRA: Efficient Finetuning of Quantized LLMs"
// https://arxiv.org/abs/2305.14314
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

// Quantize converts a float tensor to a quantized representation.
func Quantize(tensor *tensors.Tensor, config *Config) (*QuantizedWeight, error) {
	if err := config.Validate(); err != nil {
		return nil, err
	}

	switch config.QuantType {
	case NF4:
		return quantizeNF4(tensor, config)
	case Int4:
		return quantizeInt4(tensor, config)
	case Int8:
		return quantizeInt8(tensor, config)
	default:
		return nil, errors.Errorf("unsupported quantization type: %s", config.QuantType)
	}
}

// quantizeNF4 performs 4-bit NormalFloat quantization.
func quantizeNF4(tensor *tensors.Tensor, config *Config) (*QuantizedWeight, error) {
	shape := tensor.Shape()
	if shape.Rank() != 2 {
		return nil, errors.Errorf("NF4 quantization requires 2D tensor, got rank %d", shape.Rank())
	}

	rows := shape.Dimensions[0]
	cols := shape.Dimensions[1]
	groupSize := config.GroupSize
	numGroups := (cols + groupSize - 1) / groupSize

	// Get float32 data
	var data []float32
	err := tensor.ConstFlatData(func(flat any) {
		data = flat.([]float32)
	})
	if err != nil {
		return nil, errors.Wrap(err, "failed to read tensor data")
	}

	// Allocate output
	packedSize := (rows * cols + 1) / 2 // Two 4-bit values per byte
	packed := make([]uint8, packedSize)
	scales := make([]float32, rows*numGroups)

	// Quantize per group
	for row := 0; row < rows; row++ {
		for g := 0; g < numGroups; g++ {
			startCol := g * groupSize
			endCol := min(startCol+groupSize, cols)

			// Find absmax for the group
			absmax := float32(0)
			for col := startCol; col < endCol; col++ {
				val := float32(math.Abs(float64(data[row*cols+col])))
				if val > absmax {
					absmax = val
				}
			}

			// Store scale (absmax)
			scales[row*numGroups+g] = absmax

			// Quantize values in the group
			for col := startCol; col < endCol; col++ {
				idx := row*cols + col
				val := data[idx]

				// Normalize to [-1, 1]
				normalized := float32(0)
				if absmax > 0 {
					normalized = val / absmax
				}

				// Find nearest NF4 value
				quantIdx := findNearestNF4Index(normalized)

				// Pack into uint8 (two 4-bit values per byte)
				packedIdx := idx / 2
				if idx%2 == 0 {
					packed[packedIdx] = (packed[packedIdx] & 0xF0) | uint8(quantIdx)
				} else {
					packed[packedIdx] = (packed[packedIdx] & 0x0F) | (uint8(quantIdx) << 4)
				}
			}
		}
	}

	// Create tensors
	packedTensor := tensors.FromFlatDataAndDimensions(packed, packedSize)
	scalesTensor := tensors.FromFlatDataAndDimensions(scales, rows, numGroups)

	return &QuantizedWeight{
		PackedData:    packedTensor,
		Scales:        scalesTensor,
		ZeroPoints:    nil, // NF4 is symmetric
		OriginalShape: shape,
		Config:        config,
	}, nil
}

// findNearestNF4Index finds the index of the nearest NF4 value.
func findNearestNF4Index(normalized float32) int {
	// Binary search for insertion point
	idx := sort.Search(16, func(i int) bool {
		return NF4Values[i] >= normalized
	})

	// Handle edge cases
	if idx == 0 {
		return 0
	}
	if idx == 16 {
		return 15
	}

	// Choose closest of idx-1 and idx
	if math.Abs(float64(normalized-NF4Values[idx-1])) < math.Abs(float64(normalized-NF4Values[idx])) {
		return idx - 1
	}
	return idx
}

// quantizeInt4 performs 4-bit symmetric integer quantization.
func quantizeInt4(tensor *tensors.Tensor, config *Config) (*QuantizedWeight, error) {
	shape := tensor.Shape()
	if shape.Rank() != 2 {
		return nil, errors.Errorf("Int4 quantization requires 2D tensor, got rank %d", shape.Rank())
	}

	rows := shape.Dimensions[0]
	cols := shape.Dimensions[1]
	groupSize := config.GroupSize
	numGroups := (cols + groupSize - 1) / groupSize

	var data []float32
	err := tensor.ConstFlatData(func(flat any) {
		data = flat.([]float32)
	})
	if err != nil {
		return nil, errors.Wrap(err, "failed to read tensor data")
	}

	packedSize := (rows * cols + 1) / 2
	packed := make([]uint8, packedSize)
	scales := make([]float32, rows*numGroups)

	for row := 0; row < rows; row++ {
		for g := 0; g < numGroups; g++ {
			startCol := g * groupSize
			endCol := min(startCol+groupSize, cols)

			// Find absmax for the group
			absmax := float32(0)
			for col := startCol; col < endCol; col++ {
				val := float32(math.Abs(float64(data[row*cols+col])))
				if val > absmax {
					absmax = val
				}
			}

			// Scale factor: maps absmax to 7 (max positive int4 value)
			scale := absmax / 7.0
			if scale == 0 {
				scale = 1.0 // Avoid division by zero
			}
			scales[row*numGroups+g] = scale

			// Quantize values in the group
			for col := startCol; col < endCol; col++ {
				idx := row*cols + col
				val := data[idx]

				// Quantize to [-8, 7]
				quantVal := int(math.Round(float64(val / scale)))
				quantVal = max(-8, min(7, quantVal))

				// Shift to unsigned [0, 15] for packing
				unsignedVal := uint8(quantVal + 8)

				// Pack into uint8
				packedIdx := idx / 2
				if idx%2 == 0 {
					packed[packedIdx] = (packed[packedIdx] & 0xF0) | unsignedVal
				} else {
					packed[packedIdx] = (packed[packedIdx] & 0x0F) | (unsignedVal << 4)
				}
			}
		}
	}

	packedTensor := tensors.FromFlatDataAndDimensions(packed, packedSize)
	scalesTensor := tensors.FromFlatDataAndDimensions(scales, rows, numGroups)

	return &QuantizedWeight{
		PackedData:    packedTensor,
		Scales:        scalesTensor,
		ZeroPoints:    nil, // Symmetric quantization, zero point is implicit (8)
		OriginalShape: shape,
		Config:        config,
	}, nil
}

// quantizeInt8 performs 8-bit symmetric integer quantization.
func quantizeInt8(tensor *tensors.Tensor, config *Config) (*QuantizedWeight, error) {
	shape := tensor.Shape()
	if shape.Rank() != 2 {
		return nil, errors.Errorf("Int8 quantization requires 2D tensor, got rank %d", shape.Rank())
	}

	rows := shape.Dimensions[0]
	cols := shape.Dimensions[1]
	groupSize := config.GroupSize
	numGroups := (cols + groupSize - 1) / groupSize

	var data []float32
	err := tensor.ConstFlatData(func(flat any) {
		data = flat.([]float32)
	})
	if err != nil {
		return nil, errors.Wrap(err, "failed to read tensor data")
	}

	quantized := make([]int8, rows*cols)
	scales := make([]float32, rows*numGroups)

	for row := 0; row < rows; row++ {
		for g := 0; g < numGroups; g++ {
			startCol := g * groupSize
			endCol := min(startCol+groupSize, cols)

			// Find absmax for the group
			absmax := float32(0)
			for col := startCol; col < endCol; col++ {
				val := float32(math.Abs(float64(data[row*cols+col])))
				if val > absmax {
					absmax = val
				}
			}

			// Scale factor: maps absmax to 127 (max positive int8 value)
			scale := absmax / 127.0
			if scale == 0 {
				scale = 1.0
			}
			scales[row*numGroups+g] = scale

			// Quantize values
			for col := startCol; col < endCol; col++ {
				idx := row*cols + col
				val := data[idx]
				quantVal := int(math.Round(float64(val / scale)))
				quantVal = max(-128, min(127, quantVal))
				quantized[idx] = int8(quantVal)
			}
		}
	}

	quantizedTensor := tensors.FromFlatDataAndDimensions(quantized, rows, cols)
	scalesTensor := tensors.FromFlatDataAndDimensions(scales, rows, numGroups)

	return &QuantizedWeight{
		PackedData:    quantizedTensor,
		Scales:        scalesTensor,
		ZeroPoints:    nil,
		OriginalShape: shape,
		Config:        config,
	}, nil
}

// DequantizeToFloat32 converts a quantized weight back to float32.
// This is primarily for testing and verification.
func DequantizeToFloat32(qw *QuantizedWeight) (*tensors.Tensor, error) {
	switch qw.Config.QuantType {
	case NF4:
		return dequantizeNF4ToFloat32(qw)
	case Int4:
		return dequantizeInt4ToFloat32(qw)
	case Int8:
		return dequantizeInt8ToFloat32(qw)
	default:
		return nil, errors.Errorf("unsupported quantization type: %s", qw.Config.QuantType)
	}
}

func dequantizeNF4ToFloat32(qw *QuantizedWeight) (*tensors.Tensor, error) {
	rows := qw.OriginalShape.Dimensions[0]
	cols := qw.OriginalShape.Dimensions[1]
	groupSize := qw.Config.GroupSize
	numGroups := (cols + groupSize - 1) / groupSize

	var packed []uint8
	err := qw.PackedData.ConstFlatData(func(flat any) {
		packed = flat.([]uint8)
	})
	if err != nil {
		return nil, err
	}

	var scales []float32
	err = qw.Scales.ConstFlatData(func(flat any) {
		scales = flat.([]float32)
	})
	if err != nil {
		return nil, err
	}

	output := make([]float32, rows*cols)

	for row := 0; row < rows; row++ {
		for g := 0; g < numGroups; g++ {
			scale := scales[row*numGroups+g]
			startCol := g * groupSize
			endCol := min(startCol+groupSize, cols)

			for col := startCol; col < endCol; col++ {
				idx := row*cols + col
				packedIdx := idx / 2

				var quantIdx int
				if idx%2 == 0 {
					quantIdx = int(packed[packedIdx] & 0x0F)
				} else {
					quantIdx = int((packed[packedIdx] >> 4) & 0x0F)
				}

				output[idx] = NF4Values[quantIdx] * scale
			}
		}
	}

	return tensors.FromFlatDataAndDimensions(output, rows, cols), nil
}

func dequantizeInt4ToFloat32(qw *QuantizedWeight) (*tensors.Tensor, error) {
	rows := qw.OriginalShape.Dimensions[0]
	cols := qw.OriginalShape.Dimensions[1]
	groupSize := qw.Config.GroupSize
	numGroups := (cols + groupSize - 1) / groupSize

	var packed []uint8
	err := qw.PackedData.ConstFlatData(func(flat any) {
		packed = flat.([]uint8)
	})
	if err != nil {
		return nil, err
	}

	var scales []float32
	err = qw.Scales.ConstFlatData(func(flat any) {
		scales = flat.([]float32)
	})
	if err != nil {
		return nil, err
	}

	output := make([]float32, rows*cols)

	for row := 0; row < rows; row++ {
		for g := 0; g < numGroups; g++ {
			scale := scales[row*numGroups+g]
			startCol := g * groupSize
			endCol := min(startCol+groupSize, cols)

			for col := startCol; col < endCol; col++ {
				idx := row*cols + col
				packedIdx := idx / 2

				var unsignedVal int
				if idx%2 == 0 {
					unsignedVal = int(packed[packedIdx] & 0x0F)
				} else {
					unsignedVal = int((packed[packedIdx] >> 4) & 0x0F)
				}

				// Convert back to signed [-8, 7]
				signedVal := unsignedVal - 8
				output[idx] = float32(signedVal) * scale
			}
		}
	}

	return tensors.FromFlatDataAndDimensions(output, rows, cols), nil
}

func dequantizeInt8ToFloat32(qw *QuantizedWeight) (*tensors.Tensor, error) {
	rows := qw.OriginalShape.Dimensions[0]
	cols := qw.OriginalShape.Dimensions[1]
	groupSize := qw.Config.GroupSize
	numGroups := (cols + groupSize - 1) / groupSize

	var quantized []int8
	err := qw.PackedData.ConstFlatData(func(flat any) {
		quantized = flat.([]int8)
	})
	if err != nil {
		return nil, err
	}

	var scales []float32
	err = qw.Scales.ConstFlatData(func(flat any) {
		scales = flat.([]float32)
	})
	if err != nil {
		return nil, err
	}

	output := make([]float32, rows*cols)

	for row := 0; row < rows; row++ {
		for g := 0; g < numGroups; g++ {
			scale := scales[row*numGroups+g]
			startCol := g * groupSize
			endCol := min(startCol+groupSize, cols)

			for col := startCol; col < endCol; col++ {
				idx := row*cols + col
				output[idx] = float32(quantized[idx]) * scale
			}
		}
	}

	return tensors.FromFlatDataAndDimensions(output, rows, cols), nil
}

// ComputeQuantizationError computes the mean squared error between original and dequantized values.
func ComputeQuantizationError(original *tensors.Tensor, qw *QuantizedWeight) (float64, error) {
	dequantized, err := DequantizeToFloat32(qw)
	if err != nil {
		return 0, err
	}

	var origData, dequantData []float32
	err = original.ConstFlatData(func(flat any) {
		origData = flat.([]float32)
	})
	if err != nil {
		return 0, err
	}
	err = dequantized.ConstFlatData(func(flat any) {
		dequantData = flat.([]float32)
	})
	if err != nil {
		return 0, err
	}

	var sumSquaredError float64
	for i := range origData {
		diff := float64(origData[i] - dequantData[i])
		sumSquaredError += diff * diff
	}

	return sumSquaredError / float64(len(origData)), nil
}

// CreateQuantizedShape returns the shape for a quantized tensor given the original shape.
func CreateQuantizedShape(originalShape shapes.Shape, config *Config) shapes.Shape {
	totalElements := originalShape.Size()
	switch config.QuantType {
	case NF4, Int4:
		// Two 4-bit values per byte
		packedSize := (totalElements + 1) / 2
		return shapes.Make(dtypes.Uint8, packedSize)
	case Int8:
		return shapes.Make(dtypes.Int8, originalShape.Dimensions...)
	default:
		return originalShape
	}
}
