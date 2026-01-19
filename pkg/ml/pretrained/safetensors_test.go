// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package pretrained

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseSafeTensorsDType(t *testing.T) {
	tests := []struct {
		input    string
		expected dtypes.DType
		wantErr  bool
	}{
		{"F32", dtypes.Float32, false},
		{"FLOAT32", dtypes.Float32, false},
		{"F64", dtypes.Float64, false},
		{"FLOAT64", dtypes.Float64, false},
		{"F16", dtypes.Float16, false},
		{"FLOAT16", dtypes.Float16, false},
		{"BF16", dtypes.BFloat16, false},
		{"BFLOAT16", dtypes.BFloat16, false},
		{"I32", dtypes.Int32, false},
		{"INT32", dtypes.Int32, false},
		{"I64", dtypes.Int64, false},
		{"INT64", dtypes.Int64, false},
		{"I16", dtypes.Int16, false},
		{"I8", dtypes.Int8, false},
		{"U32", dtypes.Uint32, false},
		{"U64", dtypes.Uint64, false},
		{"U16", dtypes.Uint16, false},
		{"U8", dtypes.Uint8, false},
		{"BOOL", dtypes.Bool, false},
		{"INVALID", dtypes.InvalidDType, true},
		{"", dtypes.InvalidDType, true},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got, err := parseSafeTensorsDType(tt.input)
			if tt.wantErr {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
				assert.Equal(t, tt.expected, got)
			}
		})
	}
}

func TestSafeTensorsHeaderUnmarshal(t *testing.T) {
	headerJSON := `{
		"__metadata__": {"format": "pt", "version": "1.0"},
		"tensor1": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]},
		"tensor2": {"dtype": "I64", "shape": [4], "data_offsets": [24, 56]}
	}`

	var header safeTensorsHeader
	err := json.Unmarshal([]byte(headerJSON), &header)
	require.NoError(t, err)

	assert.Equal(t, "pt", header.Metadata["format"])
	assert.Equal(t, "1.0", header.Metadata["version"])

	assert.Len(t, header.Tensors, 2)

	tensor1 := header.Tensors["tensor1"]
	assert.Equal(t, "F32", tensor1.DType)
	assert.Equal(t, []int{2, 3}, tensor1.Shape)
	assert.Equal(t, [2]int64{0, 24}, tensor1.Offsets)

	tensor2 := header.Tensors["tensor2"]
	assert.Equal(t, "I64", tensor2.DType)
	assert.Equal(t, []int{4}, tensor2.Shape)
	assert.Equal(t, [2]int64{24, 56}, tensor2.Offsets)
}

func TestSafeTensorsFileMethods(t *testing.T) {
	// Create a minimal valid SafeTensors structure
	header := map[string]interface{}{
		"__metadata__": map[string]string{"test": "value"},
		"tensor1": map[string]interface{}{
			"dtype":        "F32",
			"shape":        []int{2},
			"data_offsets": []int64{0, 8},
		},
	}
	headerBytes, err := json.Marshal(header)
	require.NoError(t, err)

	var data []byte
	headerSizeBytes := make([]byte, 8)
	binary.LittleEndian.PutUint64(headerSizeBytes, uint64(len(headerBytes)))
	data = append(data, headerSizeBytes...)
	data = append(data, headerBytes...)

	// Add tensor data: 2 float32 = 8 bytes
	tensorData := make([]byte, 8)
	binary.LittleEndian.PutUint32(tensorData[0:4], math.Float32bits(1.5))
	binary.LittleEndian.PutUint32(tensorData[4:8], math.Float32bits(2.5))
	data = append(data, tensorData...)

	st, err := ParseSafeTensors(data)
	require.NoError(t, err)

	// Test TensorNames
	names := st.TensorNames()
	assert.Len(t, names, 1)

	// Test GetTensorInfo
	info, ok := st.GetTensorInfo("tensor1")
	require.True(t, ok)
	assert.Equal(t, "tensor1", info.Name)
	assert.Equal(t, dtypes.Float32, info.DType)

	// Test Close (should not panic on nil file)
	err = st.Close()
	assert.NoError(t, err)
}

func TestLoadSafeTensorsFile(t *testing.T) {
	// Create a temporary SafeTensors file for testing
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "test.safetensors")

	// Create minimal valid SafeTensors content
	header := map[string]interface{}{
		"weights": map[string]interface{}{
			"dtype":        "F32",
			"shape":        []int{3},
			"data_offsets": []int64{0, 12},
		},
	}
	headerBytes, err := json.Marshal(header)
	require.NoError(t, err)

	var fileData []byte
	headerSizeBytes := make([]byte, 8)
	binary.LittleEndian.PutUint64(headerSizeBytes, uint64(len(headerBytes)))
	fileData = append(fileData, headerSizeBytes...)
	fileData = append(fileData, headerBytes...)

	// Add tensor data: 3 float32 values = 12 bytes
	tensorBytes := make([]byte, 12)
	binary.LittleEndian.PutUint32(tensorBytes[0:4], math.Float32bits(1.0))
	binary.LittleEndian.PutUint32(tensorBytes[4:8], math.Float32bits(2.0))
	binary.LittleEndian.PutUint32(tensorBytes[8:12], math.Float32bits(3.0))
	fileData = append(fileData, tensorBytes...)

	err = os.WriteFile(filePath, fileData, 0644)
	require.NoError(t, err)

	// Test LoadSafeTensors
	st, err := LoadSafeTensors(filePath)
	require.NoError(t, err)
	require.NotNil(t, st)
	defer st.Close()

	// Verify tensor info
	info, ok := st.GetTensorInfo("weights")
	require.True(t, ok)
	assert.Equal(t, dtypes.Float32, info.DType)
	assert.Equal(t, []int{3}, info.Shape.Dimensions)

	// Load and verify tensor data
	tensor, err := st.LoadTensor("weights")
	require.NoError(t, err)
	require.NotNil(t, tensor)

	// Verify tensor values
	values := tensor.Value().([]float32)
	assert.InDelta(t, 1.0, values[0], 0.001)
	assert.InDelta(t, 2.0, values[1], 0.001)
	assert.InDelta(t, 3.0, values[2], 0.001)
}

func TestOpenSafeTensorsFile(t *testing.T) {
	// Create a temporary SafeTensors file for testing
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "test.safetensors")

	// Create minimal valid SafeTensors content
	header := map[string]interface{}{
		"weights": map[string]interface{}{
			"dtype":        "F32",
			"shape":        []int{2},
			"data_offsets": []int64{0, 8},
		},
	}
	headerBytes, err := json.Marshal(header)
	require.NoError(t, err)

	var fileData []byte
	headerSizeBytes := make([]byte, 8)
	binary.LittleEndian.PutUint64(headerSizeBytes, uint64(len(headerBytes)))
	fileData = append(fileData, headerSizeBytes...)
	fileData = append(fileData, headerBytes...)

	// Add tensor data: 2 float32 values = 8 bytes
	tensorBytes := make([]byte, 8)
	binary.LittleEndian.PutUint32(tensorBytes[0:4], math.Float32bits(1.0))
	binary.LittleEndian.PutUint32(tensorBytes[4:8], math.Float32bits(2.0))
	fileData = append(fileData, tensorBytes...)

	err = os.WriteFile(filePath, fileData, 0644)
	require.NoError(t, err)

	// Test OpenSafeTensors (lazy loading)
	st, err := OpenSafeTensors(filePath)
	require.NoError(t, err)
	require.NotNil(t, st)
	defer st.Close()

	// Verify we can still load tensors from the opened file
	tensor, err := st.LoadTensor("weights")
	require.NoError(t, err)
	require.NotNil(t, tensor)

	values := tensor.Value().([]float32)
	assert.InDelta(t, 1.0, values[0], 0.001)
	assert.InDelta(t, 2.0, values[1], 0.001)
}

func TestLoadSafeTensorsMissing(t *testing.T) {
	_, err := LoadSafeTensors("/nonexistent/path/file.safetensors")
	require.Error(t, err)
}

func TestOpenSafeTensorsMissing(t *testing.T) {
	_, err := OpenSafeTensors("/nonexistent/path/file.safetensors")
	require.Error(t, err)
}

func TestLoadTensorMissing(t *testing.T) {
	// Create minimal SafeTensors
	header := map[string]interface{}{
		"tensor1": map[string]interface{}{
			"dtype":        "F32",
			"shape":        []int{1},
			"data_offsets": []int64{0, 4},
		},
	}
	headerBytes, _ := json.Marshal(header)

	var data []byte
	headerSizeBytes := make([]byte, 8)
	binary.LittleEndian.PutUint64(headerSizeBytes, uint64(len(headerBytes)))
	data = append(data, headerSizeBytes...)
	data = append(data, headerBytes...)
	data = append(data, make([]byte, 4)...) // dummy data

	st, err := ParseSafeTensors(data)
	require.NoError(t, err)

	_, err = st.LoadTensor("nonexistent")
	require.Error(t, err)
}

func TestLoadAllTensors(t *testing.T) {
	// Create SafeTensors with multiple tensors
	header := map[string]interface{}{
		"tensor1": map[string]interface{}{
			"dtype":        "F32",
			"shape":        []int{2},
			"data_offsets": []int64{0, 8},
		},
		"tensor2": map[string]interface{}{
			"dtype":        "I32",
			"shape":        []int{2},
			"data_offsets": []int64{8, 16},
		},
	}
	headerBytes, err := json.Marshal(header)
	require.NoError(t, err)

	var data []byte
	headerSizeBytes := make([]byte, 8)
	binary.LittleEndian.PutUint64(headerSizeBytes, uint64(len(headerBytes)))
	data = append(data, headerSizeBytes...)
	data = append(data, headerBytes...)

	// Add tensor data: 2 float32 + 2 int32 = 16 bytes
	tensorData := make([]byte, 16)
	binary.LittleEndian.PutUint32(tensorData[0:4], math.Float32bits(1.0))
	binary.LittleEndian.PutUint32(tensorData[4:8], math.Float32bits(2.0))
	binary.LittleEndian.PutUint32(tensorData[8:12], 10)
	binary.LittleEndian.PutUint32(tensorData[12:16], 20)
	data = append(data, tensorData...)

	st, err := ParseSafeTensors(data)
	require.NoError(t, err)

	tensors, err := st.LoadAllTensors()
	require.NoError(t, err)
	assert.Len(t, tensors, 2)

	// Verify tensor1
	t1, ok := tensors["tensor1"]
	require.True(t, ok)
	values1 := t1.Value().([]float32)
	assert.InDelta(t, 1.0, values1[0], 0.001)
	assert.InDelta(t, 2.0, values1[1], 0.001)

	// Verify tensor2
	t2, ok := tensors["tensor2"]
	require.True(t, ok)
	values2 := t2.Value().([]int32)
	assert.Equal(t, int32(10), values2[0])
	assert.Equal(t, int32(20), values2[1])
}

func TestParseSafeTensorsTooSmall(t *testing.T) {
	// Data too small (less than 8 bytes)
	data := make([]byte, 5)
	_, err := ParseSafeTensors(data)
	require.Error(t, err)
}

func TestParseSafeTensorsTruncated(t *testing.T) {
	// Header says 100 bytes but only has 10
	data := make([]byte, 20)
	binary.LittleEndian.PutUint64(data[:8], 100) // header claims 100 bytes
	_, err := ParseSafeTensors(data)
	require.Error(t, err)
}
