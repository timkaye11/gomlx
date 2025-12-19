// Copyright 2025 The GoMLX Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build noasm || !arm64

package simplego

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"unsafe"
)

// execNormalizedDotGeneralInt8ToInt8 is a fallback implementation for int8×int8→int8
// matrix multiplication on non-ARM64 platforms. Uses scalar operations with cache tiling.
// It accumulates in int32 internally to avoid overflow, then saturates to int8.
func execNormalizedDotGeneralInt8ToInt8(lhs, rhs, output *Buffer, params *dotGeneralNodeData, batchStartIdx, batchEndIdx int) {
	lhsFlat := lhs.flat.([]int8)
	rhsFlat := rhs.flat.([]int8)
	outputFlat := output.flat.([]int8)

	contractingSize := params.contractingSize
	lhsCrossSize := params.lhsCrossSize
	rhsCrossSize := params.rhsCrossSize

	lhsBatchStride := lhsCrossSize * contractingSize
	rhsBatchStride := rhsCrossSize * contractingSize
	outputBatchStride := lhsCrossSize * rhsCrossSize

	// Block size of 64 for cache-efficient tiled matrix multiplication.
	// This creates 64x64 tiles that fit well in L1 cache (~32KB on most CPUs).
	const blockSize = 64

	// Create a temporary int32 accumulator for each output element in the current batch slice
	// to avoid repeated saturation during partial accumulation.
	accumulator := make([]int32, outputBatchStride)

	for batchIdx := batchStartIdx; batchIdx < batchEndIdx; batchIdx++ {
		lhsBaseIdx := batchIdx * lhsBatchStride
		rhsBaseIdx := batchIdx * rhsBatchStride
		outputBaseIdx := batchIdx * outputBatchStride

		// Reset accumulator for this batch
		for i := range accumulator {
			accumulator[i] = 0
		}

		for outerIdxLhsCross := 0; outerIdxLhsCross < lhsCrossSize; outerIdxLhsCross += blockSize {
			lhsCrossBlockEnd := min(outerIdxLhsCross+blockSize, lhsCrossSize)

			for outerIdxRhsCross := 0; outerIdxRhsCross < rhsCrossSize; outerIdxRhsCross += blockSize {
				rhsCrossBlockEnd := min(outerIdxRhsCross+blockSize, rhsCrossSize)

				for outerIdxContracting := 0; outerIdxContracting < contractingSize; outerIdxContracting += blockSize {
					contractingBlockEnd := min(outerIdxContracting+blockSize, contractingSize)

					for idxLhsCross := outerIdxLhsCross; idxLhsCross < lhsCrossBlockEnd; idxLhsCross++ {
						lhsRowStartIdx := lhsBaseIdx + idxLhsCross*contractingSize
						accRowStartIdx := idxLhsCross * rhsCrossSize

						for idxRhsCross := outerIdxRhsCross; idxRhsCross < rhsCrossBlockEnd; idxRhsCross++ {
							rhsColStartIdx := rhsBaseIdx + idxRhsCross*contractingSize
							sum := accumulator[accRowStartIdx+idxRhsCross]

							// Scalar implementation - accumulate in int32
							for idxContracting := outerIdxContracting; idxContracting < contractingBlockEnd; idxContracting++ {
								lhsVal := int32(lhsFlat[lhsRowStartIdx+idxContracting])
								rhsVal := int32(rhsFlat[rhsColStartIdx+idxContracting])
								sum += lhsVal * rhsVal
							}

							accumulator[accRowStartIdx+idxRhsCross] = sum
						}
					}
				}
			}
		}

		// Saturate and copy to output
		for i := 0; i < outputBatchStride; i++ {
			outputFlat[outputBaseIdx+i] = saturateInt32ToInt8(accumulator[i])
		}
	}
}

// execNormalizedDotGeneralUint8ToUint8 is a fallback implementation for uint8×uint8→uint8
// Also handles mixed int8/uint8 cases by treating everything as unsigned.
// It accumulates in int32 internally to avoid overflow, then saturates to uint8.
// Uses cache tiling for better performance on large matrices.
func execNormalizedDotGeneralUint8ToUint8(lhs, rhs, output *Buffer, params *dotGeneralNodeData, batchStartIdx, batchEndIdx int) {
	// Handle both uint8 and int8 inputs by converting to uint8 view
	var lhsFlat, rhsFlat []uint8

	// Convert lhs to uint8 view
	switch lhs.shape.DType {
	case dtypes.Uint8:
		lhsFlat = lhs.flat.([]uint8)
	case dtypes.Int8:
		// Reinterpret int8 as uint8 (same bit pattern, different interpretation)
		int8Flat := lhs.flat.([]int8)
		lhsFlat = unsafe.Slice((*uint8)(unsafe.Pointer(&int8Flat[0])), len(int8Flat))
	}

	// Convert rhs to uint8 view
	switch rhs.shape.DType {
	case dtypes.Uint8:
		rhsFlat = rhs.flat.([]uint8)
	case dtypes.Int8:
		int8Flat := rhs.flat.([]int8)
		rhsFlat = unsafe.Slice((*uint8)(unsafe.Pointer(&int8Flat[0])), len(int8Flat))
	}

	outputFlat := output.flat.([]uint8)

	contractingSize := params.contractingSize
	lhsCrossSize := params.lhsCrossSize
	rhsCrossSize := params.rhsCrossSize

	lhsBatchStride := lhsCrossSize * contractingSize
	rhsBatchStride := rhsCrossSize * contractingSize
	outputBatchStride := lhsCrossSize * rhsCrossSize

	// Block size of 64 for cache-efficient tiled matrix multiplication.
	const blockSize = 64

	// Create a temporary int32 accumulator for each output element in the current batch slice
	accumulator := make([]int32, outputBatchStride)

	for batchIdx := batchStartIdx; batchIdx < batchEndIdx; batchIdx++ {
		lhsBaseIdx := batchIdx * lhsBatchStride
		rhsBaseIdx := batchIdx * rhsBatchStride
		outputBaseIdx := batchIdx * outputBatchStride

		// Reset accumulator for this batch
		for i := range accumulator {
			accumulator[i] = 0
		}

		for outerIdxLhsCross := 0; outerIdxLhsCross < lhsCrossSize; outerIdxLhsCross += blockSize {
			lhsCrossBlockEnd := min(outerIdxLhsCross+blockSize, lhsCrossSize)

			for outerIdxRhsCross := 0; outerIdxRhsCross < rhsCrossSize; outerIdxRhsCross += blockSize {
				rhsCrossBlockEnd := min(outerIdxRhsCross+blockSize, rhsCrossSize)

				for outerIdxContracting := 0; outerIdxContracting < contractingSize; outerIdxContracting += blockSize {
					contractingBlockEnd := min(outerIdxContracting+blockSize, contractingSize)

					for idxLhsCross := outerIdxLhsCross; idxLhsCross < lhsCrossBlockEnd; idxLhsCross++ {
						lhsRowStartIdx := lhsBaseIdx + idxLhsCross*contractingSize
						accRowStartIdx := idxLhsCross * rhsCrossSize

						for idxRhsCross := outerIdxRhsCross; idxRhsCross < rhsCrossBlockEnd; idxRhsCross++ {
							rhsColStartIdx := rhsBaseIdx + idxRhsCross*contractingSize
							sum := accumulator[accRowStartIdx+idxRhsCross]

							// Scalar implementation - accumulate in int32
							for idxContracting := outerIdxContracting; idxContracting < contractingBlockEnd; idxContracting++ {
								lhsVal := int32(lhsFlat[lhsRowStartIdx+idxContracting])
								rhsVal := int32(rhsFlat[rhsColStartIdx+idxContracting])
								sum += lhsVal * rhsVal
							}

							accumulator[accRowStartIdx+idxRhsCross] = sum
						}
					}
				}
			}
		}

		// Saturate and copy to output
		for i := 0; i < outputBatchStride; i++ {
			outputFlat[outputBaseIdx+i] = saturateInt32ToUint8(accumulator[i])
		}
	}
}
