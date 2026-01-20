// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package quantization

// This file provides utilities for packing and unpacking 4-bit integer values.
// Two 4-bit values are stored per byte for memory efficiency.
//
// Packing layout: [high nibble][low nibble] = [value1 << 4 | value0]
// - Even indices (0, 2, 4, ...) go into the low nibble (bits 0-3)
// - Odd indices (1, 3, 5, ...) go into the high nibble (bits 4-7)

// PackInt4 packs two 4-bit values into a single byte.
// low goes into bits 0-3, high goes into bits 4-7.
func PackInt4(low, high uint8) uint8 {
	return (low & 0x0F) | ((high & 0x0F) << 4)
}

// UnpackInt4 extracts two 4-bit values from a packed byte.
// Returns (low, high) where low is bits 0-3 and high is bits 4-7.
func UnpackInt4(packed uint8) (low, high uint8) {
	low = packed & 0x0F
	high = (packed >> 4) & 0x0F
	return low, high
}

// PackInt4Slice packs a slice of 4-bit values into a byte slice.
// Each pair of consecutive values is packed into a single byte.
// If the input has an odd length, the last value's high nibble is zero.
func PackInt4Slice(values []uint8) []uint8 {
	packedLen := (len(values) + 1) / 2
	packed := make([]uint8, packedLen)

	for i := 0; i < len(values); i += 2 {
		low := values[i]
		high := uint8(0)
		if i+1 < len(values) {
			high = values[i+1]
		}
		packed[i/2] = PackInt4(low, high)
	}

	return packed
}

// UnpackInt4Slice unpacks a byte slice into 4-bit values.
// count specifies the number of 4-bit values to extract.
func UnpackInt4Slice(packed []uint8, count int) []uint8 {
	values := make([]uint8, count)

	for i := 0; i < count; i++ {
		packedIdx := i / 2
		if packedIdx >= len(packed) {
			break
		}

		if i%2 == 0 {
			values[i] = packed[packedIdx] & 0x0F
		} else {
			values[i] = (packed[packedIdx] >> 4) & 0x0F
		}
	}

	return values
}

// PackSignedInt4Slice packs signed 4-bit values [-8, 7] into bytes.
// Values are shifted to unsigned [0, 15] before packing.
func PackSignedInt4Slice(values []int8) []uint8 {
	unsigned := make([]uint8, len(values))
	for i, v := range values {
		// Shift from [-8, 7] to [0, 15]
		unsigned[i] = uint8(v + 8)
	}
	return PackInt4Slice(unsigned)
}

// UnpackSignedInt4Slice unpacks bytes into signed 4-bit values [-8, 7].
// Values are shifted from unsigned [0, 15] back to signed.
func UnpackSignedInt4Slice(packed []uint8, count int) []int8 {
	unsigned := UnpackInt4Slice(packed, count)
	signed := make([]int8, len(unsigned))
	for i, v := range unsigned {
		// Shift from [0, 15] to [-8, 7]
		signed[i] = int8(v) - 8
	}
	return signed
}

// PackedSize returns the number of bytes needed to pack n 4-bit values.
func PackedSize(n int) int {
	return (n + 1) / 2
}

// GetPackedValue extracts a single 4-bit value from a packed byte slice.
// index is the logical index of the 4-bit value (0, 1, 2, ...).
func GetPackedValue(packed []uint8, index int) uint8 {
	packedIdx := index / 2
	if packedIdx >= len(packed) {
		return 0
	}

	if index%2 == 0 {
		return packed[packedIdx] & 0x0F
	}
	return (packed[packedIdx] >> 4) & 0x0F
}

// SetPackedValue sets a single 4-bit value in a packed byte slice.
// index is the logical index of the 4-bit value (0, 1, 2, ...).
func SetPackedValue(packed []uint8, index int, value uint8) {
	packedIdx := index / 2
	if packedIdx >= len(packed) {
		return
	}

	value = value & 0x0F // Ensure only 4 bits
	if index%2 == 0 {
		packed[packedIdx] = (packed[packedIdx] & 0xF0) | value
	} else {
		packed[packedIdx] = (packed[packedIdx] & 0x0F) | (value << 4)
	}
}

// PackNF4Indices packs NF4 quantization indices (0-15) into bytes.
// This is an alias for PackInt4Slice since NF4 indices are already unsigned 4-bit.
func PackNF4Indices(indices []uint8) []uint8 {
	return PackInt4Slice(indices)
}

// UnpackNF4Indices unpacks bytes into NF4 quantization indices (0-15).
func UnpackNF4Indices(packed []uint8, count int) []uint8 {
	return UnpackInt4Slice(packed, count)
}
