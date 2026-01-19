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

func TestLoadTensorEmptyShape(t *testing.T) {
	tests := []struct {
		name  string
		shape []int
	}{
		{"empty_1d", []int{0}},
		{"empty_2d_first", []int{0, 5}},
		{"empty_2d_second", []int{5, 0}},
		{"empty_2d_both", []int{0, 0}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create SafeTensors with empty shape tensor
			header := map[string]interface{}{
				"empty_tensor": map[string]interface{}{
					"dtype":        "F32",
					"shape":        tt.shape,
					"data_offsets": []int64{0, 0}, // Empty tensor has no data
				},
			}
			headerBytes, err := json.Marshal(header)
			require.NoError(t, err)

			var data []byte
			headerSizeBytes := make([]byte, 8)
			binary.LittleEndian.PutUint64(headerSizeBytes, uint64(len(headerBytes)))
			data = append(data, headerSizeBytes...)
			data = append(data, headerBytes...)
			// No tensor data needed for empty tensor

			st, err := ParseSafeTensors(data)
			require.NoError(t, err)

			// Verify tensor info
			info, ok := st.GetTensorInfo("empty_tensor")
			require.True(t, ok)
			assert.Equal(t, tt.shape, info.Shape.Dimensions)

			// Load the empty tensor
			tensor, err := st.LoadTensor("empty_tensor")
			require.NoError(t, err)
			require.NotNil(t, tensor)
			assert.Equal(t, 0, tensor.Shape().Size())
		})
	}
}

func TestLoadTensorFloat16(t *testing.T) {
	// Create SafeTensors with F16 dtype
	header := map[string]interface{}{
		"f16_tensor": map[string]interface{}{
			"dtype":        "F16",
			"shape":        []int{3},
			"data_offsets": []int64{0, 6}, // 3 float16 values = 6 bytes
		},
	}
	headerBytes, err := json.Marshal(header)
	require.NoError(t, err)

	var data []byte
	headerSizeBytes := make([]byte, 8)
	binary.LittleEndian.PutUint64(headerSizeBytes, uint64(len(headerBytes)))
	data = append(data, headerSizeBytes...)
	data = append(data, headerBytes...)

	// Add Float16 tensor data: 3 float16 values = 6 bytes
	// Using known float16 bit patterns
	tensorData := make([]byte, 6)
	binary.LittleEndian.PutUint16(tensorData[0:2], 0x3C00) // 1.0 in float16
	binary.LittleEndian.PutUint16(tensorData[2:4], 0x4000) // 2.0 in float16
	binary.LittleEndian.PutUint16(tensorData[4:6], 0x4200) // 3.0 in float16
	data = append(data, tensorData...)

	st, err := ParseSafeTensors(data)
	require.NoError(t, err)

	// Verify tensor info
	info, ok := st.GetTensorInfo("f16_tensor")
	require.True(t, ok)
	assert.Equal(t, dtypes.Float16, info.DType)
	assert.Equal(t, []int{3}, info.Shape.Dimensions)

	// Load and verify tensor
	tensor, err := st.LoadTensor("f16_tensor")
	require.NoError(t, err)
	require.NotNil(t, tensor)
	assert.Equal(t, 3, tensor.Shape().Size())
}

func TestLoadTensorBFloat16(t *testing.T) {
	// Create SafeTensors with BF16 dtype
	header := map[string]interface{}{
		"bf16_tensor": map[string]interface{}{
			"dtype":        "BF16",
			"shape":        []int{2},
			"data_offsets": []int64{0, 4}, // 2 bfloat16 values = 4 bytes
		},
	}
	headerBytes, err := json.Marshal(header)
	require.NoError(t, err)

	var data []byte
	headerSizeBytes := make([]byte, 8)
	binary.LittleEndian.PutUint64(headerSizeBytes, uint64(len(headerBytes)))
	data = append(data, headerSizeBytes...)
	data = append(data, headerBytes...)

	// Add BFloat16 tensor data: 2 bfloat16 values = 4 bytes
	// Using known bfloat16 bit patterns
	tensorData := make([]byte, 4)
	binary.LittleEndian.PutUint16(tensorData[0:2], 0x3F80) // 1.0 in bfloat16
	binary.LittleEndian.PutUint16(tensorData[2:4], 0x4000) // 2.0 in bfloat16
	data = append(data, tensorData...)

	st, err := ParseSafeTensors(data)
	require.NoError(t, err)

	// Verify tensor info
	info, ok := st.GetTensorInfo("bf16_tensor")
	require.True(t, ok)
	assert.Equal(t, dtypes.BFloat16, info.DType)
	assert.Equal(t, []int{2}, info.Shape.Dimensions)

	// Load and verify tensor
	tensor, err := st.LoadTensor("bf16_tensor")
	require.NoError(t, err)
	require.NotNil(t, tensor)
	assert.Equal(t, 2, tensor.Shape().Size())
}

func TestLoadTensorInt8Int16(t *testing.T) {
	// Test I8 dtype
	t.Run("i8_tensor", func(t *testing.T) {
		header := map[string]interface{}{
			"i8_tensor": map[string]interface{}{
				"dtype":        "I8",
				"shape":        []int{4},
				"data_offsets": []int64{0, 4},
			},
		}
		headerBytes, err := json.Marshal(header)
		require.NoError(t, err)

		var data []byte
		headerSizeBytes := make([]byte, 8)
		binary.LittleEndian.PutUint64(headerSizeBytes, uint64(len(headerBytes)))
		data = append(data, headerSizeBytes...)
		data = append(data, headerBytes...)

		// Add I8 tensor data: convert signed int8 values to bytes
		i8Values := []int8{-128, -1, 0, 127} // Min, negative, zero, max
		tensorData := make([]byte, 4)
		for i, v := range i8Values {
			tensorData[i] = byte(v)
		}
		data = append(data, tensorData...)

		st, err := ParseSafeTensors(data)
		require.NoError(t, err)

		// Verify tensor info
		info, ok := st.GetTensorInfo("i8_tensor")
		require.True(t, ok)
		assert.Equal(t, dtypes.Int8, info.DType)
		assert.Equal(t, []int{4}, info.Shape.Dimensions)

		// Load and verify tensor
		tensor, err := st.LoadTensor("i8_tensor")
		require.NoError(t, err)
		require.NotNil(t, tensor)

		// Verify values
		values := tensor.Value().([]int8)
		assert.Equal(t, int8(-128), values[0])
		assert.Equal(t, int8(-1), values[1])
		assert.Equal(t, int8(0), values[2])
		assert.Equal(t, int8(127), values[3])
	})

	// Test I16 dtype
	t.Run("i16_tensor", func(t *testing.T) {
		header := map[string]interface{}{
			"i16_tensor": map[string]interface{}{
				"dtype":        "I16",
				"shape":        []int{2},
				"data_offsets": []int64{0, 4},
			},
		}
		headerBytes, err := json.Marshal(header)
		require.NoError(t, err)

		var data []byte
		headerSizeBytes := make([]byte, 8)
		binary.LittleEndian.PutUint64(headerSizeBytes, uint64(len(headerBytes)))
		data = append(data, headerSizeBytes...)
		data = append(data, headerBytes...)

		// Add I16 tensor data: convert signed int16 values to uint16
		i16Values := []int16{-1000, 1000}
		tensorData := make([]byte, 4)
		for i, v := range i16Values {
			binary.LittleEndian.PutUint16(tensorData[i*2:(i+1)*2], uint16(v))
		}
		data = append(data, tensorData...)

		st, err := ParseSafeTensors(data)
		require.NoError(t, err)

		// Verify tensor info
		info, ok := st.GetTensorInfo("i16_tensor")
		require.True(t, ok)
		assert.Equal(t, dtypes.Int16, info.DType)
		assert.Equal(t, []int{2}, info.Shape.Dimensions)

		// Load and verify tensor
		tensor, err := st.LoadTensor("i16_tensor")
		require.NoError(t, err)
		require.NotNil(t, tensor)

		// Verify values
		values := tensor.Value().([]int16)
		assert.Equal(t, int16(-1000), values[0])
		assert.Equal(t, int16(1000), values[1])
	})
}

func TestLoadTensorUint8Uint16(t *testing.T) {
	tests := []struct {
		name       string
		dtype      string
		shape      []int
		dataSize   int
		dtypeValue dtypes.DType
		values     []byte
	}{
		{
			name:       "u8_tensor",
			dtype:      "U8",
			shape:      []int{3},
			dataSize:   3,
			dtypeValue: dtypes.Uint8,
			values:     []byte{0, 128, 255}, // Min, middle, max
		},
		{
			name:       "u16_tensor",
			dtype:      "U16",
			shape:      []int{2},
			dataSize:   4,
			dtypeValue: dtypes.Uint16,
			values:     nil, // Will be set below
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			header := map[string]interface{}{
				tt.name: map[string]interface{}{
					"dtype":        tt.dtype,
					"shape":        tt.shape,
					"data_offsets": []int64{0, int64(tt.dataSize)},
				},
			}
			headerBytes, err := json.Marshal(header)
			require.NoError(t, err)

			var data []byte
			headerSizeBytes := make([]byte, 8)
			binary.LittleEndian.PutUint64(headerSizeBytes, uint64(len(headerBytes)))
			data = append(data, headerSizeBytes...)
			data = append(data, headerBytes...)

			// Add tensor data
			if tt.dtype == "U8" {
				data = append(data, tt.values...)
			} else if tt.dtype == "U16" {
				tensorData := make([]byte, 4)
				binary.LittleEndian.PutUint16(tensorData[0:2], 0)
				binary.LittleEndian.PutUint16(tensorData[2:4], 65535)
				data = append(data, tensorData...)
			}

			st, err := ParseSafeTensors(data)
			require.NoError(t, err)

			// Verify tensor info
			info, ok := st.GetTensorInfo(tt.name)
			require.True(t, ok)
			assert.Equal(t, tt.dtypeValue, info.DType)
			assert.Equal(t, tt.shape, info.Shape.Dimensions)

			// Load and verify tensor
			tensor, err := st.LoadTensor(tt.name)
			require.NoError(t, err)
			require.NotNil(t, tensor)
		})
	}
}

func TestLoadTensorBool(t *testing.T) {
	// Create SafeTensors with Bool dtype
	header := map[string]interface{}{
		"bool_tensor": map[string]interface{}{
			"dtype":        "BOOL",
			"shape":        []int{5},
			"data_offsets": []int64{0, 5}, // 5 bool values = 5 bytes
		},
	}
	headerBytes, err := json.Marshal(header)
	require.NoError(t, err)

	var data []byte
	headerSizeBytes := make([]byte, 8)
	binary.LittleEndian.PutUint64(headerSizeBytes, uint64(len(headerBytes)))
	data = append(data, headerSizeBytes...)
	data = append(data, headerBytes...)

	// Add Bool tensor data: true, false, true, false, true
	tensorData := []byte{1, 0, 1, 0, 1}
	data = append(data, tensorData...)

	st, err := ParseSafeTensors(data)
	require.NoError(t, err)

	// Verify tensor info
	info, ok := st.GetTensorInfo("bool_tensor")
	require.True(t, ok)
	assert.Equal(t, dtypes.Bool, info.DType)
	assert.Equal(t, []int{5}, info.Shape.Dimensions)

	// Load and verify tensor
	tensor, err := st.LoadTensor("bool_tensor")
	require.NoError(t, err)
	require.NotNil(t, tensor)

	// Verify values
	values := tensor.Value().([]bool)
	expected := []bool{true, false, true, false, true}
	assert.Equal(t, expected, values)
}

func TestParseSafeTensorsNegativeOffsets(t *testing.T) {
	// Create SafeTensors with negative data offsets
	header := map[string]interface{}{
		"bad_tensor": map[string]interface{}{
			"dtype":        "F32",
			"shape":        []int{2},
			"data_offsets": []int64{-10, 8}, // Negative start offset
		},
	}
	headerBytes, err := json.Marshal(header)
	require.NoError(t, err)

	var data []byte
	headerSizeBytes := make([]byte, 8)
	binary.LittleEndian.PutUint64(headerSizeBytes, uint64(len(headerBytes)))
	data = append(data, headerSizeBytes...)
	data = append(data, headerBytes...)
	data = append(data, make([]byte, 8)...) // Dummy data

	st, err := ParseSafeTensors(data)
	require.NoError(t, err) // Parsing succeeds

	// Loading should fail due to invalid offsets
	_, err = st.LoadTensor("bad_tensor")
	require.Error(t, err)
}

func TestParseSafeTensorsOverlappingData(t *testing.T) {
	// Create SafeTensors with overlapping tensor data regions
	header := map[string]interface{}{
		"tensor1": map[string]interface{}{
			"dtype":        "F32",
			"shape":        []int{2},
			"data_offsets": []int64{0, 8},
		},
		"tensor2": map[string]interface{}{
			"dtype":        "F32",
			"shape":        []int{2},
			"data_offsets": []int64{4, 12}, // Overlaps with tensor1 (starts at 4, but tensor1 ends at 8)
		},
	}
	headerBytes, err := json.Marshal(header)
	require.NoError(t, err)

	var data []byte
	headerSizeBytes := make([]byte, 8)
	binary.LittleEndian.PutUint64(headerSizeBytes, uint64(len(headerBytes)))
	data = append(data, headerSizeBytes...)
	data = append(data, headerBytes...)

	// Add 12 bytes of tensor data (enough for overlapping regions)
	tensorData := make([]byte, 12)
	for i := 0; i < 12; i += 4 {
		binary.LittleEndian.PutUint32(tensorData[i:i+4], math.Float32bits(float32(i)))
	}
	data = append(data, tensorData...)

	// SafeTensors format allows overlapping regions (though unusual)
	// The parser should still succeed
	st, err := ParseSafeTensors(data)
	require.NoError(t, err)

	// Both tensors should be loadable (they share some data)
	t1, err := st.LoadTensor("tensor1")
	require.NoError(t, err)
	require.NotNil(t, t1)

	t2, err := st.LoadTensor("tensor2")
	require.NoError(t, err)
	require.NotNil(t, t2)
}
