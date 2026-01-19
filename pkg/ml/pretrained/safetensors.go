// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package pretrained provides utilities for loading pre-trained model weights
// from common formats like SafeTensors.

package pretrained

import (
	"encoding/binary"
	"encoding/json"
	"io"
	"os"
	"strings"
	"unsafe"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/dtypes/bfloat16"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/pkg/errors"
	"github.com/x448/float16"
)

// SafeTensorsFile represents a parsed SafeTensors file.
type SafeTensorsFile struct {
	// Metadata contains optional metadata from the file header.
	Metadata map[string]string

	// Tensors maps tensor names to their metadata.
	Tensors map[string]*SafeTensorInfo

	// dataOffset is the byte offset where tensor data begins.
	dataOffset int64

	// file is the underlying file handle (if opened from file).
	file *os.File

	// data is the raw tensor data (if loaded into memory).
	data []byte
}

// SafeTensorInfo contains metadata about a single tensor in a SafeTensors file.
type SafeTensorInfo struct {
	// Name is the tensor name.
	Name string

	// DType is the data type of the tensor.
	DType dtypes.DType

	// Shape is the shape of the tensor.
	Shape shapes.Shape

	// DataStart is the byte offset (from start of data section) where this tensor begins.
	DataStart int64

	// DataEnd is the byte offset (from start of data section) where this tensor ends.
	DataEnd int64
}

// safeTensorsHeader is the JSON header format used in SafeTensors files.
type safeTensorsHeader struct {
	Metadata map[string]string                 `json:"__metadata__"`
	Tensors  map[string]safeTensorHeaderEntry  `json:"-"`
}

type safeTensorHeaderEntry struct {
	DType   string  `json:"dtype"`
	Shape   []int   `json:"shape"`
	Offsets [2]int64 `json:"data_offsets"`
}

// UnmarshalJSON implements custom JSON unmarshaling for the header.
func (h *safeTensorsHeader) UnmarshalJSON(data []byte) error {
	// First, unmarshal into a generic map
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}

	h.Tensors = make(map[string]safeTensorHeaderEntry)

	for key, value := range raw {
		if key == "__metadata__" {
			var meta map[string]string
			if err := json.Unmarshal(value, &meta); err != nil {
				return errors.Wrapf(err, "failed to parse __metadata__")
			}
			h.Metadata = meta
		} else {
			var entry safeTensorHeaderEntry
			if err := json.Unmarshal(value, &entry); err != nil {
				return errors.Wrapf(err, "failed to parse tensor %q", key)
			}
			h.Tensors[key] = entry
		}
	}

	return nil
}

// OpenSafeTensors opens a SafeTensors file and parses its header.
// The file is kept open for lazy tensor loading. Call Close() when done.
func OpenSafeTensors(filePath string) (*SafeTensorsFile, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to open SafeTensors file %q", filePath)
	}

	st, err := parseSafeTensorsHeader(file)
	if err != nil {
		file.Close()
		return nil, err
	}

	st.file = file
	return st, nil
}

// LoadSafeTensors loads a SafeTensors file entirely into memory.
// This is faster for accessing many tensors but uses more memory.
func LoadSafeTensors(filePath string) (*SafeTensorsFile, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to read SafeTensors file %q", filePath)
	}

	return ParseSafeTensors(data)
}

// ParseSafeTensors parses SafeTensors data from a byte slice.
func ParseSafeTensors(data []byte) (*SafeTensorsFile, error) {
	if len(data) < 8 {
		return nil, errors.New("SafeTensors file too small: missing header size")
	}

	headerSize := binary.LittleEndian.Uint64(data[:8])
	if uint64(len(data)) < 8+headerSize {
		return nil, errors.Errorf("SafeTensors file truncated: expected %d bytes for header, got %d",
			headerSize, len(data)-8)
	}

	headerData := data[8 : 8+headerSize]
	var header safeTensorsHeader
	if err := json.Unmarshal(headerData, &header); err != nil {
		return nil, errors.Wrapf(err, "failed to parse SafeTensors header")
	}

	st := &SafeTensorsFile{
		Metadata:   header.Metadata,
		Tensors:    make(map[string]*SafeTensorInfo),
		dataOffset: int64(8 + headerSize),
		data:       data,
	}

	for name, entry := range header.Tensors {
		dtype, err := parseSafeTensorsDType(entry.DType)
		if err != nil {
			return nil, errors.Wrapf(err, "tensor %q", name)
		}

		st.Tensors[name] = &SafeTensorInfo{
			Name:      name,
			DType:     dtype,
			Shape:     shapes.Make(dtype, entry.Shape...),
			DataStart: entry.Offsets[0],
			DataEnd:   entry.Offsets[1],
		}
	}

	return st, nil
}

// parseSafeTensorsHeader reads and parses the header from a reader.
func parseSafeTensorsHeader(r io.Reader) (*SafeTensorsFile, error) {
	// Read header size (8 bytes, little-endian uint64)
	var headerSize uint64
	if err := binary.Read(r, binary.LittleEndian, &headerSize); err != nil {
		return nil, errors.Wrap(err, "failed to read SafeTensors header size")
	}

	// Read header JSON
	headerData := make([]byte, headerSize)
	if _, err := io.ReadFull(r, headerData); err != nil {
		return nil, errors.Wrap(err, "failed to read SafeTensors header")
	}

	var header safeTensorsHeader
	if err := json.Unmarshal(headerData, &header); err != nil {
		return nil, errors.Wrapf(err, "failed to parse SafeTensors header")
	}

	st := &SafeTensorsFile{
		Metadata:   header.Metadata,
		Tensors:    make(map[string]*SafeTensorInfo),
		dataOffset: int64(8 + headerSize),
	}

	for name, entry := range header.Tensors {
		dtype, err := parseSafeTensorsDType(entry.DType)
		if err != nil {
			return nil, errors.Wrapf(err, "tensor %q", name)
		}

		st.Tensors[name] = &SafeTensorInfo{
			Name:      name,
			DType:     dtype,
			Shape:     shapes.Make(dtype, entry.Shape...),
			DataStart: entry.Offsets[0],
			DataEnd:   entry.Offsets[1],
		}
	}

	return st, nil
}

// parseSafeTensorsDType converts a SafeTensors dtype string to GoMLX dtype.
func parseSafeTensorsDType(dtype string) (dtypes.DType, error) {
	switch strings.ToUpper(dtype) {
	case "F64", "FLOAT64":
		return dtypes.Float64, nil
	case "F32", "FLOAT32":
		return dtypes.Float32, nil
	case "F16", "FLOAT16":
		return dtypes.Float16, nil
	case "BF16", "BFLOAT16":
		return dtypes.BFloat16, nil
	case "I64", "INT64":
		return dtypes.Int64, nil
	case "I32", "INT32":
		return dtypes.Int32, nil
	case "I16", "INT16":
		return dtypes.Int16, nil
	case "I8", "INT8":
		return dtypes.Int8, nil
	case "U64", "UINT64":
		return dtypes.Uint64, nil
	case "U32", "UINT32":
		return dtypes.Uint32, nil
	case "U16", "UINT16":
		return dtypes.Uint16, nil
	case "U8", "UINT8":
		return dtypes.Uint8, nil
	case "BOOL":
		return dtypes.Bool, nil
	default:
		return dtypes.InvalidDType, errors.Errorf("unknown SafeTensors dtype: %q", dtype)
	}
}

// Close closes the underlying file if opened with OpenSafeTensors.
func (st *SafeTensorsFile) Close() error {
	if st.file != nil {
		return st.file.Close()
	}
	return nil
}

// TensorNames returns all tensor names in the file.
func (st *SafeTensorsFile) TensorNames() []string {
	names := make([]string, 0, len(st.Tensors))
	for name := range st.Tensors {
		names = append(names, name)
	}
	return names
}

// GetTensorInfo returns information about a tensor without loading its data.
func (st *SafeTensorsFile) GetTensorInfo(name string) (*SafeTensorInfo, bool) {
	info, ok := st.Tensors[name]
	return info, ok
}

// LoadTensor loads a tensor by name into a GoMLX tensor.
func (st *SafeTensorsFile) LoadTensor(name string) (*tensors.Tensor, error) {
	info, ok := st.Tensors[name]
	if !ok {
		return nil, errors.Errorf("tensor %q not found in SafeTensors file", name)
	}

	// Validate data offsets BEFORE allocating memory to prevent OOM attacks
	if info.DataStart < 0 {
		return nil, errors.Errorf("tensor %q has negative DataStart offset: %d", name, info.DataStart)
	}
	if info.DataEnd < info.DataStart {
		return nil, errors.Errorf("tensor %q has invalid offsets: DataEnd (%d) < DataStart (%d)",
			name, info.DataEnd, info.DataStart)
	}

	dataSize := info.DataEnd - info.DataStart

	if st.data != nil {
		// Validate bounds for in-memory data
		start := st.dataOffset + info.DataStart
		end := st.dataOffset + info.DataEnd
		if end > int64(len(st.data)) {
			return nil, errors.Errorf("tensor %q data extends beyond file (offset %d + end %d > length %d)",
				name, st.dataOffset, info.DataEnd, len(st.data))
		}
		if start < 0 || end < 0 {
			return nil, errors.Errorf("tensor %q has invalid computed offsets: start=%d, end=%d",
				name, start, end)
		}
	}

	// Safe to allocate now that we've validated offsets
	tensorData := make([]byte, dataSize)

	if st.data != nil {
		// Data is in memory
		start := st.dataOffset + info.DataStart
		end := st.dataOffset + info.DataEnd
		copy(tensorData, st.data[start:end])
	} else if st.file != nil {
		// Read from file
		_, err := st.file.Seek(st.dataOffset+info.DataStart, io.SeekStart)
		if err != nil {
			return nil, errors.Wrapf(err, "failed to seek to tensor %q data", name)
		}
		_, err = io.ReadFull(st.file, tensorData)
		if err != nil {
			return nil, errors.Wrapf(err, "failed to read tensor %q data", name)
		}
	} else {
		return nil, errors.New("SafeTensorsFile has no data source")
	}

	// Create tensor from raw bytes
	return tensorFromRawBytes(info.Shape, tensorData)
}

// tensorFromRawBytes creates a tensor from raw bytes according to the shape's dtype.
func tensorFromRawBytes(shape shapes.Shape, data []byte) (*tensors.Tensor, error) {
	dtype := shape.DType
	elemSize := dtype.Size()
	expectedSize := shape.Size() * elemSize

	if len(data) != expectedSize {
		return nil, errors.Errorf("data size mismatch: expected %d bytes for %v, got %d",
			expectedSize, shape, len(data))
	}

	// Create a slice of the appropriate type from the raw bytes
	switch dtype {
	case dtypes.Float64:
		return tensorFromBytes[float64](shape, data), nil
	case dtypes.Float32:
		return tensorFromBytes[float32](shape, data), nil
	case dtypes.Float16:
		// Float16 is stored as uint16 internally
		return tensorFromBytesF16(shape, data), nil
	case dtypes.BFloat16:
		// BFloat16 is stored as uint16 internally
		return tensorFromBytesBF16(shape, data), nil
	case dtypes.Int64:
		return tensorFromBytes[int64](shape, data), nil
	case dtypes.Int32:
		return tensorFromBytes[int32](shape, data), nil
	case dtypes.Int16:
		return tensorFromBytes[int16](shape, data), nil
	case dtypes.Int8:
		return tensorFromBytes[int8](shape, data), nil
	case dtypes.Uint64:
		return tensorFromBytes[uint64](shape, data), nil
	case dtypes.Uint32:
		return tensorFromBytes[uint32](shape, data), nil
	case dtypes.Uint16:
		return tensorFromBytes[uint16](shape, data), nil
	case dtypes.Uint8:
		return tensorFromBytes[uint8](shape, data), nil
	case dtypes.Bool:
		return tensorFromBytesBool(shape, data), nil
	default:
		return nil, errors.Errorf("unsupported dtype for raw bytes: %v", dtype)
	}
}

// tensorFromBytes creates a tensor from raw bytes for a specific Go type.
// Note: SafeTensors format uses little-endian encoding. This function assumes
// the system is little-endian. On big-endian systems, byte swapping would be required.
func tensorFromBytes[T dtypes.Supported](shape shapes.Shape, data []byte) *tensors.Tensor {
	numElements := shape.Size()
	copied := make([]T, numElements)

	// Handle empty tensors (zero elements) gracefully
	if numElements == 0 || len(data) == 0 {
		return tensors.FromFlatDataAndDimensions(copied, shape.Dimensions...)
	}

	// Use unsafe to reinterpret bytes as the target type slice
	slice := unsafe.Slice((*T)(unsafe.Pointer(&data[0])), numElements)
	// Make a copy to avoid issues with the original byte slice
	copy(copied, slice)
	return tensors.FromFlatDataAndDimensions(copied, shape.Dimensions...)
}

// tensorFromBytesF16 creates a Float16 tensor from raw bytes.
// Note: SafeTensors format uses little-endian encoding. This function assumes
// the system is little-endian. On big-endian systems, byte swapping would be required.
func tensorFromBytesF16(shape shapes.Shape, data []byte) *tensors.Tensor {
	numElements := shape.Size()
	f16Slice := make([]float16.Float16, numElements)

	// Handle empty tensors (zero elements) gracefully
	if numElements == 0 || len(data) == 0 {
		return tensors.FromFlatDataAndDimensions(f16Slice, shape.Dimensions...)
	}

	// Float16 is stored as uint16 internally
	u16Slice := unsafe.Slice((*uint16)(unsafe.Pointer(&data[0])), numElements)
	for i, v := range u16Slice {
		f16Slice[i] = float16.Frombits(v)
	}
	return tensors.FromFlatDataAndDimensions(f16Slice, shape.Dimensions...)
}

// tensorFromBytesBF16 creates a BFloat16 tensor from raw bytes.
// Note: SafeTensors format uses little-endian encoding. This function assumes
// the system is little-endian. On big-endian systems, byte swapping would be required.
func tensorFromBytesBF16(shape shapes.Shape, data []byte) *tensors.Tensor {
	numElements := shape.Size()
	bf16Slice := make([]bfloat16.BFloat16, numElements)

	// Handle empty tensors (zero elements) gracefully
	if numElements == 0 || len(data) == 0 {
		return tensors.FromFlatDataAndDimensions(bf16Slice, shape.Dimensions...)
	}

	// BFloat16 is stored as uint16 internally
	u16Slice := unsafe.Slice((*uint16)(unsafe.Pointer(&data[0])), numElements)
	for i, v := range u16Slice {
		bf16Slice[i] = bfloat16.FromBits(v)
	}
	return tensors.FromFlatDataAndDimensions(bf16Slice, shape.Dimensions...)
}

// tensorFromBytesBool creates a Bool tensor from raw bytes.
func tensorFromBytesBool(shape shapes.Shape, data []byte) *tensors.Tensor {
	numElements := shape.Size()
	boolSlice := make([]bool, numElements)
	for i := 0; i < numElements; i++ {
		boolSlice[i] = data[i] != 0
	}
	return tensors.FromFlatDataAndDimensions(boolSlice, shape.Dimensions...)
}

// LoadAllTensors loads all tensors into a map.
func (st *SafeTensorsFile) LoadAllTensors() (map[string]*tensors.Tensor, error) {
	result := make(map[string]*tensors.Tensor, len(st.Tensors))
	for name := range st.Tensors {
		tensor, err := st.LoadTensor(name)
		if err != nil {
			return nil, err
		}
		result[name] = tensor
	}
	return result, nil
}
