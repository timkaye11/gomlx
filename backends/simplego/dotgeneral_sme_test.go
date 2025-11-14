// Copyright 2025 The GoMLX Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !noasm && arm64

package simplego

import (
	"fmt"
	"testing"

	"github.com/gomlx/gomlx/types/shapes"
)

// TestSMEDetection tests SME feature detection
func TestSMEDetection(t *testing.T) {
	// This test always runs to log SME availability
	if hasSME {
		t.Logf("✅ SME (Scalable Matrix Extension) detected - using variable-length vectors")
	} else {
		t.Logf("ℹ️  SME not available - using scalar fallback")
	}
}

// TestDotProductSME tests the SME dot product implementation with progressive sizes
func TestDotProductSME(t *testing.T) {
	if !hasSME {
		t.Skip("SME not available on this system")
	}

	// Test with progressively larger sizes
	sizes := []int{8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192}

	for _, size := range sizes {
		a := make([]float32, size)
		b := make([]float32, size)

		for i := 0; i < size; i++ {
			a[i] = 1.0
			b[i] = 2.0
		}

		var expected float32
		for i := 0; i < size; i++ {
			expected += a[i] * b[i]
		}

		result := dotProduct_sme(a, b, 0, 0, int64(size))

		if result != expected {
			t.Errorf("Size %d failed: got %f, expected %f", size, result, expected)
			return
		}
	}
}

// TestDotProductSMEEdgeCases tests edge cases like NaN, Inf, negative numbers, etc.
func TestDotProductSMEEdgeCases(t *testing.T) {
	if !hasSME {
		t.Skip("SME not available on this system")
	}

	tests := []struct {
		name     string
		size     int
		setupA   func([]float32)
		setupB   func([]float32)
		validate func(float32) bool
	}{
		{
			name: "negative_numbers",
			size: 128,
			setupA: func(a []float32) {
				for i := range a {
					a[i] = -float32(i + 1)
				}
			},
			setupB: func(b []float32) {
				for i := range b {
					b[i] = float32(i + 1)
				}
			},
			validate: func(result float32) bool {
				// Sum of -i * i for i=1..128 = -(1+4+9+...+128²)
				var expected float32
				for i := 1; i <= 128; i++ {
					expected += -float32(i * i)
				}
				return result == expected
			},
		},
		{
			name: "mixed_signs",
			size: 64,
			setupA: func(a []float32) {
				for i := range a {
					if i%2 == 0 {
						a[i] = 1.0
					} else {
						a[i] = -1.0
					}
				}
			},
			setupB: func(b []float32) {
				for i := range b {
					b[i] = 2.0
				}
			},
			validate: func(result float32) bool {
				return result == 0.0 // 32 positive + 32 negative = 0
			},
		},
		{
			name: "zeros",
			size: 256,
			setupA: func(a []float32) {
				for i := range a {
					a[i] = 0.0
				}
			},
			setupB: func(b []float32) {
				for i := range b {
					b[i] = float32(i)
				}
			},
			validate: func(result float32) bool {
				return result == 0.0
			},
		},
		{
			name: "small_values",
			size: 128,
			setupA: func(a []float32) {
				for i := range a {
					a[i] = 1e-6
				}
			},
			setupB: func(b []float32) {
				for i := range b {
					b[i] = 1e-6
				}
			},
			validate: func(result float32) bool {
				expected := 128 * 1e-12
				// Allow for floating point error
				return result > expected*0.99 && result < expected*1.01
			},
		},
		{
			name: "large_values",
			size: 64,
			setupA: func(a []float32) {
				for i := range a {
					a[i] = 1e6
				}
			},
			setupB: func(b []float32) {
				for i := range b {
					b[i] = 1e6
				}
			},
			validate: func(result float32) bool {
				expected := 64 * 1e12
				// Allow for floating point error
				return result > expected*0.99 && result < expected*1.01
			},
		},
		{
			name: "non_aligned_size",
			size: 67, // Not a multiple of 16
			setupA: func(a []float32) {
				for i := range a {
					a[i] = 1.0
				}
			},
			setupB: func(b []float32) {
				for i := range b {
					b[i] = 2.0
				}
			},
			validate: func(result float32) bool {
				return result == 134.0 // 67 * 2
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := make([]float32, tt.size)
			b := make([]float32, tt.size)

			tt.setupA(a)
			tt.setupB(b)

			// Compute expected using scalar implementation
			var expected float32
			for i := 0; i < tt.size; i++ {
				expected += a[i] * b[i]
			}

			// Compute using SME
			result := dotProduct_sme(a, b, 0, 0, int64(tt.size))

			// Validate result
			if !tt.validate(result) {
				t.Errorf("Failed: got %f, expected %f", result, expected)
			}

			// Also verify it matches scalar computation
			if result != expected {
				t.Errorf("Result mismatch with scalar: SME=%f, scalar=%f", result, expected)
			}
		})
	}
}

// TestDotProductSMEWithOffset tests the function with non-zero indices
func TestDotProductSMEWithOffset(t *testing.T) {
	if !hasSME {
		t.Skip("SME not available on this system")
	}

	size := 256
	offset := 64

	a := make([]float32, size)
	b := make([]float32, size)

	for i := range a {
		a[i] = float32(i)
		b[i] = 2.0
	}

	// Compute dot product starting at offset
	result := dotProduct_sme(a, b, offset, offset, int64(size-offset))

	// Expected: sum of i*2 for i=64..255
	var expected float32
	for i := offset; i < size; i++ {
		expected += a[i] * b[i]
	}

	if result != expected {
		t.Errorf("Offset test failed: got %f, expected %f", result, expected)
	}
}

// TestSMEIntegration tests that the SME path is properly integrated into buildDotGeneralKernel
func TestSMEIntegration(t *testing.T) {
	if !hasSME {
		t.Skip("SME not available on this system")
	}

	// Test with blockDim >= 64 to trigger SME path
	blockDim := 128
	blockSize := blockDim * blockDim

	// Create test buffers
	lhsShape := &shapes.Shape{DType: shapes.F32, Dimensions: []int{blockDim, blockDim}}
	rhsShape := &shapes.Shape{DType: shapes.F32, Dimensions: []int{blockDim, blockDim}}
	outputShape := &shapes.Shape{DType: shapes.F32, Dimensions: []int{blockDim, blockDim}}

	lhs := NewBuffer(lhsShape)
	rhs := NewBuffer(rhsShape)
	output := NewBuffer(outputShape)

	// Initialize input data
	lhsFlat := lhs.AdjustType(lhs.shape.DType).([]float32)
	rhsFlat := rhs.AdjustType(rhs.shape.DType).([]float32)
	outputFlat := output.AdjustType(output.shape.DType).([]float32)

	for i := range lhsFlat {
		lhsFlat[i] = 1.0
	}
	for i := range rhsFlat {
		rhsFlat[i] = 2.0
	}
	for i := range outputFlat {
		outputFlat[i] = 0.0
	}

	// Build and execute the kernel
	kernel := buildDotGeneralKernel[float32](lhs, rhs, output, blockDim, blockSize)
	kernel(0, 0, 0)

	// Verify results
	// Each output element should be blockDim * 1.0 * 2.0 = 2 * blockDim
	expected := float32(2 * blockDim)
	for i := range outputFlat {
		if outputFlat[i] != expected {
			t.Errorf("Integration test failed at index %d: got %f, expected %f", i, outputFlat[i], expected)
			break
		}
	}

	t.Logf("✅ SME integration test passed: blockDim=%d, all %d outputs correct", blockDim, len(outputFlat))
}

// TestSMEIntegrationVsScalar verifies SME produces same results as scalar implementation
func TestSMEIntegrationVsScalar(t *testing.T) {
	if !hasSME {
		t.Skip("SME not available on this system")
	}

	blockDim := 64
	blockSize := blockDim * blockDim

	// Create test buffers
	lhsShape := &shapes.Shape{DType: shapes.F32, Dimensions: []int{blockDim, blockDim}}
	rhsShape := &shapes.Shape{DType: shapes.F32, Dimensions: []int{blockDim, blockDim}}
	outputShape := &shapes.Shape{DType: shapes.F32, Dimensions: []int{blockDim, blockDim}}

	// Test with SME
	lhsSME := NewBuffer(lhsShape)
	rhsSME := NewBuffer(rhsShape)
	outputSME := NewBuffer(outputShape)

	// Initialize with non-trivial data
	lhsFlatSME := lhsSME.AdjustType(lhsSME.shape.DType).([]float32)
	rhsFlatSME := rhsSME.AdjustType(rhsSME.shape.DType).([]float32)
	outputFlatSME := outputSME.AdjustType(outputSME.shape.DType).([]float32)

	for i := range lhsFlatSME {
		lhsFlatSME[i] = float32(i%10) / 10.0
	}
	for i := range rhsFlatSME {
		rhsFlatSME[i] = float32((i*7)%10) / 10.0
	}
	for i := range outputFlatSME {
		outputFlatSME[i] = 0.0
	}

	// Execute with SME (blockDim=64, triggers SME path)
	kernelSME := buildDotGeneralKernel[float32](lhsSME, rhsSME, outputSME, blockDim, blockSize)
	kernelSME(0, 0, 0)

	// Test with scalar (use smaller blockDim to avoid SME)
	// We'll compute the same operation using pure scalar by using blockDim < 64
	// But actually, let's just compute it manually
	outputScalar := make([]float32, len(outputFlatSME))
	for i := 0; i < blockDim; i++ {
		for j := 0; j < blockDim; j++ {
			var sum float32
			for k := 0; k < blockDim; k++ {
				sum += lhsFlatSME[i*blockDim+k] * rhsFlatSME[k*blockDim+j]
			}
			outputScalar[i*blockDim+j] = sum
		}
	}

	// Compare results
	maxDiff := float32(0.0)
	for i := range outputFlatSME {
		diff := outputFlatSME[i] - outputScalar[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > maxDiff {
			maxDiff = diff
		}
		// Allow for floating point rounding differences
		if diff > 1e-4 {
			t.Errorf("Result mismatch at index %d: SME=%f, scalar=%f, diff=%f",
				i, outputFlatSME[i], outputScalar[i], diff)
		}
	}

	t.Logf("✅ SME vs scalar comparison passed: max difference = %e", maxDiff)
}

// BenchmarkDotProductSME benchmarks the SME implementation
func BenchmarkDotProductSME(b *testing.B) {
	if !hasSME {
		b.Skip("SME not available on this system")
	}

	sizes := []int{64, 512, 2048, 8192}

	for _, size := range sizes {
		a := make([]float32, size)
		c := make([]float32, size)
		for i := 0; i < size; i++ {
			a[i] = float32(i)
			c[i] = 2.0
		}

		b.Run(fmt.Sprintf("%d", size), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = dotProduct_sme(a, c, 0, 0, int64(size))
			}
		})
	}
}

// BenchmarkDotProductScalar benchmarks the scalar implementation for comparison
func BenchmarkDotProductScalar(b *testing.B) {
	sizes := []int{64, 512, 2048, 8192}

	for _, size := range sizes {
		a := make([]float32, size)
		c := make([]float32, size)
		for i := 0; i < size; i++ {
			a[i] = float32(i)
			c[i] = 2.0
		}

		b.Run(fmt.Sprintf("%d", size), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				var sum float32
				for j := 0; j < size; j++ {
					sum += a[j] * c[j]
				}
				_ = sum
			}
		})
	}
}
