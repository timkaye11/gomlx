// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package quantization_test

import (
	"math"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/quantization"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewConfig(t *testing.T) {
	config := quantization.NewConfig()

	assert.Equal(t, quantization.NF4, config.QuantType)
	assert.Equal(t, quantization.PerGroup, config.Granularity)
	assert.Equal(t, 64, config.GroupSize)
	assert.Equal(t, dtypes.BFloat16, config.ComputeDType)
}

func TestConfigSetters(t *testing.T) {
	config := quantization.NewConfig().
		SetQuantType(quantization.Int8).
		SetGranularity(quantization.PerChannel).
		SetGroupSize(32).
		SetComputeDType(dtypes.Float16)

	assert.Equal(t, quantization.Int8, config.QuantType)
	assert.Equal(t, quantization.PerChannel, config.Granularity)
	assert.Equal(t, 32, config.GroupSize)
	assert.Equal(t, dtypes.Float16, config.ComputeDType)
}

func TestConfigValidation(t *testing.T) {
	// Valid config
	config := quantization.NewConfig()
	err := config.Validate()
	require.NoError(t, err)

	// Invalid group size
	config = quantization.NewConfig()
	config.GroupSize = 0
	err = config.Validate()
	require.Error(t, err)

	// Invalid compute dtype (non-float)
	config = quantization.NewConfig()
	config.ComputeDType = dtypes.Int32
	err = config.Validate()
	require.Error(t, err)
}

func TestConfigBitsPerWeight(t *testing.T) {
	config := quantization.NewConfig()

	config.SetQuantType(quantization.NF4)
	assert.Equal(t, 4, config.BitsPerWeight())

	config.SetQuantType(quantization.Int4)
	assert.Equal(t, 4, config.BitsPerWeight())

	config.SetQuantType(quantization.Int8)
	assert.Equal(t, 8, config.BitsPerWeight())
}

func TestNF4Values(t *testing.T) {
	// NF4 should have 16 values
	assert.Equal(t, 16, len(quantization.NF4Values))

	// Values should be sorted
	for i := 1; i < len(quantization.NF4Values); i++ {
		assert.Greater(t, quantization.NF4Values[i], quantization.NF4Values[i-1],
			"NF4 values should be sorted in ascending order")
	}

	// First and last should be -1 and 1
	assert.Equal(t, float32(-1.0), quantization.NF4Values[0])
	assert.Equal(t, float32(1.0), quantization.NF4Values[15])

	// Should contain 0
	found := false
	for _, v := range quantization.NF4Values {
		if v == 0 {
			found = true
			break
		}
	}
	assert.True(t, found, "NF4 values should contain 0")
}

func TestQuantizeNF4(t *testing.T) {
	// Create a small test tensor
	rows, cols := 4, 8
	data := make([]float32, rows*cols)
	for i := range data {
		// Values roughly in [-1, 1] range
		data[i] = float32(i%8-4) / 4.0
	}
	tensor := tensors.FromFlatDataAndDimensions(data, rows, cols)

	config := quantization.NewConfig().SetQuantType(quantization.NF4).SetGroupSize(4)

	qw, err := quantization.Quantize(tensor, config)
	require.NoError(t, err)
	require.NotNil(t, qw)

	// Check that packed data was created
	assert.NotNil(t, qw.PackedData)
	assert.NotNil(t, qw.Scales)

	// Check original shape is preserved
	assert.Equal(t, shapes.Make(dtypes.Float32, rows, cols), qw.OriginalShape)

	// Packed data should be roughly half the size (2 values per byte)
	packedSize := qw.PackedData.Shape().Size()
	expectedPackedSize := (rows * cols + 1) / 2
	assert.Equal(t, expectedPackedSize, packedSize)

	// Scales should have shape [rows, numGroups]
	numGroups := (cols + config.GroupSize - 1) / config.GroupSize
	assert.Equal(t, []int{rows, numGroups}, qw.Scales.Shape().Dimensions)
}

func TestQuantizeInt4(t *testing.T) {
	rows, cols := 4, 8
	data := make([]float32, rows*cols)
	for i := range data {
		data[i] = float32(i%8-4) * 0.5
	}
	tensor := tensors.FromFlatDataAndDimensions(data, rows, cols)

	config := quantization.NewConfig().SetQuantType(quantization.Int4).SetGroupSize(4)

	qw, err := quantization.Quantize(tensor, config)
	require.NoError(t, err)
	require.NotNil(t, qw)

	assert.NotNil(t, qw.PackedData)
	assert.NotNil(t, qw.Scales)
	assert.Equal(t, shapes.Make(dtypes.Float32, rows, cols), qw.OriginalShape)
}

func TestQuantizeInt8(t *testing.T) {
	rows, cols := 4, 8
	data := make([]float32, rows*cols)
	for i := range data {
		data[i] = float32(i%8-4) * 0.5
	}
	tensor := tensors.FromFlatDataAndDimensions(data, rows, cols)

	config := quantization.NewConfig().SetQuantType(quantization.Int8).SetGroupSize(4)

	qw, err := quantization.Quantize(tensor, config)
	require.NoError(t, err)
	require.NotNil(t, qw)

	assert.NotNil(t, qw.PackedData)
	assert.NotNil(t, qw.Scales)
	assert.Equal(t, shapes.Make(dtypes.Float32, rows, cols), qw.OriginalShape)

	// Int8 is not packed, so size should match original
	assert.Equal(t, rows*cols, qw.PackedData.Shape().Size())
}

func TestDequantizeNF4(t *testing.T) {
	rows, cols := 4, 8
	data := make([]float32, rows*cols)
	for i := range data {
		data[i] = float32(i%8-4) / 4.0
	}
	tensor := tensors.FromFlatDataAndDimensions(data, rows, cols)

	config := quantization.NewConfig().SetQuantType(quantization.NF4).SetGroupSize(4)

	qw, err := quantization.Quantize(tensor, config)
	require.NoError(t, err)

	dequantized, err := quantization.DequantizeToFloat32(qw)
	require.NoError(t, err)

	// Shape should match original
	assert.Equal(t, tensor.Shape(), dequantized.Shape())

	// Check quantization error is reasonable
	mse, err := quantization.ComputeQuantizationError(tensor, qw)
	require.NoError(t, err)
	assert.Less(t, mse, 0.1, "MSE should be reasonably low for NF4 quantization")
}

func TestDequantizeInt4(t *testing.T) {
	rows, cols := 4, 8
	data := make([]float32, rows*cols)
	for i := range data {
		data[i] = float32(i%8-4) * 0.5
	}
	tensor := tensors.FromFlatDataAndDimensions(data, rows, cols)

	config := quantization.NewConfig().SetQuantType(quantization.Int4).SetGroupSize(4)

	qw, err := quantization.Quantize(tensor, config)
	require.NoError(t, err)

	dequantized, err := quantization.DequantizeToFloat32(qw)
	require.NoError(t, err)

	assert.Equal(t, tensor.Shape(), dequantized.Shape())

	mse, err := quantization.ComputeQuantizationError(tensor, qw)
	require.NoError(t, err)
	assert.Less(t, mse, 0.1, "MSE should be reasonably low for Int4 quantization")
}

func TestDequantizeInt8(t *testing.T) {
	rows, cols := 4, 8
	data := make([]float32, rows*cols)
	for i := range data {
		data[i] = float32(i%8-4) * 0.5
	}
	tensor := tensors.FromFlatDataAndDimensions(data, rows, cols)

	config := quantization.NewConfig().SetQuantType(quantization.Int8).SetGroupSize(4)

	qw, err := quantization.Quantize(tensor, config)
	require.NoError(t, err)

	dequantized, err := quantization.DequantizeToFloat32(qw)
	require.NoError(t, err)

	assert.Equal(t, tensor.Shape(), dequantized.Shape())

	mse, err := quantization.ComputeQuantizationError(tensor, qw)
	require.NoError(t, err)
	// Int8 should have lower error than 4-bit
	assert.Less(t, mse, 0.01, "MSE should be very low for Int8 quantization")
}

func TestQuantizationRoundTrip(t *testing.T) {
	// Test that quantize->dequantize preserves data approximately
	rows, cols := 8, 16
	data := make([]float32, rows*cols)
	for i := range data {
		// Normally distributed-ish values
		data[i] = float32(math.Sin(float64(i)*0.1) * 0.5)
	}
	tensor := tensors.FromFlatDataAndDimensions(data, rows, cols)

	tests := []struct {
		name     string
		quantType quantization.QuantType
		maxMSE   float64
	}{
		{"NF4", quantization.NF4, 0.05},
		{"Int4", quantization.Int4, 0.05},
		{"Int8", quantization.Int8, 0.001},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := quantization.NewConfig().SetQuantType(tt.quantType).SetGroupSize(8)

			qw, err := quantization.Quantize(tensor, config)
			require.NoError(t, err)

			mse, err := quantization.ComputeQuantizationError(tensor, qw)
			require.NoError(t, err)

			assert.Less(t, mse, tt.maxMSE,
				"MSE %f exceeds max %f for %s", mse, tt.maxMSE, tt.name)
		})
	}
}

func TestCompressionRatio(t *testing.T) {
	rows, cols := 64, 64
	data := make([]float32, rows*cols)
	for i := range data {
		data[i] = float32(i) / float32(rows*cols)
	}
	tensor := tensors.FromFlatDataAndDimensions(data, rows, cols)

	// NF4/Int4 should achieve ~8x compression (4 bits vs 32 bits)
	config := quantization.NewConfig().SetQuantType(quantization.NF4).SetGroupSize(64)
	qw, err := quantization.Quantize(tensor, config)
	require.NoError(t, err)

	ratio := qw.CompressionRatio()
	// Should be roughly 6-8x depending on scale overhead
	assert.Greater(t, ratio, 5.0, "NF4 compression ratio should be > 5x")
	assert.Less(t, ratio, 10.0, "NF4 compression ratio should be < 10x")

	// Int8 should achieve ~4x compression
	config = quantization.NewConfig().SetQuantType(quantization.Int8).SetGroupSize(64)
	qw, err = quantization.Quantize(tensor, config)
	require.NoError(t, err)

	ratio = qw.CompressionRatio()
	assert.Greater(t, ratio, 3.0, "Int8 compression ratio should be > 3x")
	assert.Less(t, ratio, 5.0, "Int8 compression ratio should be < 5x")
}

func TestQuantizeRequires2D(t *testing.T) {
	// 1D tensor should fail
	tensor1D := tensors.FromFlatDataAndDimensions([]float32{1, 2, 3, 4}, 4)
	config := quantization.NewConfig()

	_, err := quantization.Quantize(tensor1D, config)
	require.Error(t, err)

	// 3D tensor should fail
	tensor3D := tensors.FromFlatDataAndDimensions(make([]float32, 8), 2, 2, 2)
	_, err = quantization.Quantize(tensor3D, config)
	require.Error(t, err)
}

func TestCreateQuantizedShape(t *testing.T) {
	originalShape := shapes.Make(dtypes.Float32, 64, 128)

	// NF4/Int4 should pack 2 values per byte
	config := quantization.NewConfig().SetQuantType(quantization.NF4)
	packedShape := quantization.CreateQuantizedShape(originalShape, config)
	expectedSize := (64 * 128 + 1) / 2
	assert.Equal(t, expectedSize, packedShape.Size())
	assert.Equal(t, dtypes.Uint8, packedShape.DType)

	// Int8 should preserve element count
	config = quantization.NewConfig().SetQuantType(quantization.Int8)
	packedShape = quantization.CreateQuantizedShape(originalShape, config)
	assert.Equal(t, 64*128, packedShape.Size())
	assert.Equal(t, dtypes.Int8, packedShape.DType)
}
