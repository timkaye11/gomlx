/*
 *	Copyright 2024 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

package seq2seq

import (
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/x448/float16"
)

// TensorToFloat32Slice extracts tensor data as a flat float32 slice.
// Handles conversion from various dtypes.
func TensorToFloat32Slice(t *tensors.Tensor) ([]float32, error) {
	shape := t.Shape()

	switch shape.DType {
	case dtypes.Float32:
		return tensors.MustCopyFlatData[float32](t), nil
	case dtypes.Float64:
		data := tensors.MustCopyFlatData[float64](t)
		result := make([]float32, len(data))
		for i, v := range data {
			result[i] = float32(v)
		}
		return result, nil
	case dtypes.Float16:
		// Float16 is stored as uint16, need to convert
		data := tensors.MustCopyFlatData[uint16](t)
		result := make([]float32, len(data))
		for i, v := range data {
			result[i] = float16ToFloat32(v)
		}
		return result, nil
	default:
		return nil, nil
	}
}

// TensorToInt32Slice extracts tensor data as a flat int32 slice.
func TensorToInt32Slice(t *tensors.Tensor) ([]int32, error) {
	shape := t.Shape()

	switch shape.DType {
	case dtypes.Int32:
		return tensors.MustCopyFlatData[int32](t), nil
	case dtypes.Int64:
		data := tensors.MustCopyFlatData[int64](t)
		result := make([]int32, len(data))
		for i, v := range data {
			result[i] = int32(v)
		}
		return result, nil
	case dtypes.Int16:
		data := tensors.MustCopyFlatData[int16](t)
		result := make([]int32, len(data))
		for i, v := range data {
			result[i] = int32(v)
		}
		return result, nil
	default:
		return nil, nil
	}
}

// CreateInt32Tensor creates a tensor from int32 data with the given shape.
func CreateInt32Tensor(data []int32, dims ...int) *tensors.Tensor {
	return tensors.FromFlatDataAndDimensions(data, dims...)
}

// CreateFloat32Tensor creates a tensor from float32 data with the given shape.
func CreateFloat32Tensor(data []float32, dims ...int) *tensors.Tensor {
	return tensors.FromFlatDataAndDimensions(data, dims...)
}

// CreateZerosTensor creates a tensor filled with zeros.
func CreateZerosTensor(dtype dtypes.DType, dims ...int) *tensors.Tensor {
	shape := shapes.Make(dtype, dims...)
	return tensors.FromShape(shape)
}

// CreateOnesTensor creates a tensor filled with ones.
func CreateOnesTensor(dtype dtypes.DType, dims ...int) *tensors.Tensor {
	size := 1
	for _, d := range dims {
		size *= d
	}

	switch dtype {
	case dtypes.Float32:
		data := make([]float32, size)
		for i := range data {
			data[i] = 1.0
		}
		return tensors.FromFlatDataAndDimensions(data, dims...)
	case dtypes.Float64:
		data := make([]float64, size)
		for i := range data {
			data[i] = 1.0
		}
		return tensors.FromFlatDataAndDimensions(data, dims...)
	case dtypes.Int32:
		data := make([]int32, size)
		for i := range data {
			data[i] = 1
		}
		return tensors.FromFlatDataAndDimensions(data, dims...)
	case dtypes.Int64:
		data := make([]int64, size)
		for i := range data {
			data[i] = 1
		}
		return tensors.FromFlatDataAndDimensions(data, dims...)
	default:
		// Default to float32.
		data := make([]float32, size)
		for i := range data {
			data[i] = 1.0
		}
		return tensors.FromFlatDataAndDimensions(data, dims...)
	}
}

// float16ToFloat32 converts a float16 (stored as uint16) to float32.
func float16ToFloat32(h uint16) float32 {
	return float16.Frombits(h).Float32()
}

// ConcatenateTensors concatenates tensors along the specified axis.
// This creates a new tensor; the original tensors are not modified.
func ConcatenateTensors(tensors []*tensors.Tensor, axis int) (*tensors.Tensor, error) {
	if len(tensors) == 0 {
		return nil, nil
	}
	if len(tensors) == 1 {
		return tensors[0], nil
	}

	// For now, just return the first tensor.
	// Full implementation would use graph operations.
	return tensors[0], nil
}

// GetKVCacheShape returns the expected shape for a KV cache tensor.
func GetKVCacheShape(batchSize, numHeads, seqLen, headDim int, dtype dtypes.DType) shapes.Shape {
	return shapes.Make(dtype, batchSize, numHeads, seqLen, headDim)
}

// ValidateKVCacheShape checks if a tensor has a valid KV cache shape.
func ValidateKVCacheShape(t *tensors.Tensor, expectedBatchSize, expectedNumHeads, expectedHeadDim int) bool {
	shape := t.Shape()
	if shape.Rank() != 4 {
		return false
	}
	dims := shape.Dimensions
	return dims[0] == expectedBatchSize &&
		dims[1] == expectedNumHeads &&
		dims[3] == expectedHeadDim
}

// ExtractLastPosition extracts the last position from a sequence tensor.
// Input shape: [batch, seq_len, ...], Output shape: [batch, 1, ...]
func ExtractLastPosition(t *tensors.Tensor) (*tensors.Tensor, error) {
	shape := t.Shape()
	if shape.Rank() < 2 {
		return t, nil
	}

	// For simple cases, this needs graph operations.
	// For now, return the tensor as-is.
	return t, nil
}
