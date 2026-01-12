// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

//go:generate go run ../../internal/cmd/alternates_generator -base=dotgeneral_small_matmul_alt_base.go -tags=bf16,f16

//alt:base import (
//alt:base _ "github.com/gomlx/gomlx/pkg/core/dtypes/bfloat16"
//alt:base _ "github.com/x448/float16"
//alt:base )
//alt:bf16 import	"github.com/gomlx/gomlx/pkg/core/dtypes/bfloat16"
import "github.com/x448/float16" //alt:f16

// execDotGeneralSmallMatMul* executes matrix multiplication without transpose.
//
// BFloat16/Float16 variants read from native dtype inputs, accumulate in float32,
// and write to float32 output buffer. The caller handles conversion back to native dtype.
//
//alt:base func execDotGeneralSmallMatMulFloat32(_ *Backend, lhs, rhs *Buffer, params *dotGeneralNodeData, output *Buffer) {
//alt:bf16 func execDotGeneralSmallMatMulBFloat16(_ *Backend, lhs, rhs *Buffer, params *dotGeneralNodeData, output *Buffer) {
func execDotGeneralSmallMatMulFloat16(_ *Backend, lhs, rhs *Buffer, params *dotGeneralNodeData, output *Buffer) { //alt:f16

	//alt:base lhsFlat := lhs.flat.([]float32)
	//alt:base rhsFlat := rhs.flat.([]float32)
	//alt:base outputFlat := output.flat.([]float32)
	//alt:bf16 lhsFlat := lhs.flat.([]bfloat16.BFloat16)
	//alt:bf16 rhsFlat := rhs.flat.([]bfloat16.BFloat16)
	//alt:bf16 outputFlat := output.flat.([]float32)
	lhsFlat := lhs.flat.([]float16.Float16) //alt:f16
	rhsFlat := rhs.flat.([]float16.Float16) //alt:f16
	outputFlat := output.flat.([]float32)   //alt:f16

	batchSize := params.batchSize
	lhsCrossSize := params.lhsCrossSize
	rhsCrossSize := params.rhsCrossSize
	contractingSize := params.contractingSize

	lhsBatchStride := lhsCrossSize * contractingSize
	rhsBatchStride := contractingSize * rhsCrossSize
	outputBatchStride := lhsCrossSize * rhsCrossSize
	rhsColStride := rhsCrossSize

	for batchIdx := range batchSize {
		lhsBaseIdx := batchIdx * lhsBatchStride
		rhsBaseIdx := batchIdx * rhsBatchStride
		outputBaseIdx := batchIdx * outputBatchStride

		for m := range lhsCrossSize {
			lhsRowStart := lhsBaseIdx + m*contractingSize
			outputRowStart := outputBaseIdx + m*rhsCrossSize

			for n := range rhsCrossSize {
				rhsColStart := rhsBaseIdx + n
				var sum float32

				k := 0
				for ; k+3 < contractingSize; k += 4 {
					//alt:base sum += lhsFlat[lhsRowStart+k]*rhsFlat[rhsColStart+k*rhsColStride] +
					//alt:base lhsFlat[lhsRowStart+k+1]*rhsFlat[rhsColStart+(k+1)*rhsColStride] +
					//alt:base lhsFlat[lhsRowStart+k+2]*rhsFlat[rhsColStart+(k+2)*rhsColStride] +
					//alt:base lhsFlat[lhsRowStart+k+3]*rhsFlat[rhsColStart+(k+3)*rhsColStride]
					//alt:bf16 sum += lhsFlat[lhsRowStart+k].Float32()*rhsFlat[rhsColStart+k*rhsColStride].Float32() +
					//alt:bf16 lhsFlat[lhsRowStart+k+1].Float32()*rhsFlat[rhsColStart+(k+1)*rhsColStride].Float32() +
					//alt:bf16 lhsFlat[lhsRowStart+k+2].Float32()*rhsFlat[rhsColStart+(k+2)*rhsColStride].Float32() +
					//alt:bf16 lhsFlat[lhsRowStart+k+3].Float32()*rhsFlat[rhsColStart+(k+3)*rhsColStride].Float32()
					sum += lhsFlat[lhsRowStart+k].Float32()*rhsFlat[rhsColStart+k*rhsColStride].Float32() + //alt:f16
						lhsFlat[lhsRowStart+k+1].Float32()*rhsFlat[rhsColStart+(k+1)*rhsColStride].Float32() + //alt:f16
						lhsFlat[lhsRowStart+k+2].Float32()*rhsFlat[rhsColStart+(k+2)*rhsColStride].Float32() + //alt:f16
						lhsFlat[lhsRowStart+k+3].Float32()*rhsFlat[rhsColStart+(k+3)*rhsColStride].Float32() //alt:f16
				}
				for ; k < contractingSize; k++ {
					//alt:base sum += lhsFlat[lhsRowStart+k] * rhsFlat[rhsColStart+k*rhsColStride]
					//alt:bf16 sum += lhsFlat[lhsRowStart+k].Float32() * rhsFlat[rhsColStart+k*rhsColStride].Float32()
					sum += lhsFlat[lhsRowStart+k].Float32() * rhsFlat[rhsColStart+k*rhsColStride].Float32() //alt:f16
				}

				outputFlat[outputRowStart+n] = sum
			}
		}
	}
}
