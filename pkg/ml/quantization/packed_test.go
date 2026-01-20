// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package quantization

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestPackInt4(t *testing.T) {
	tests := []struct {
		low, high uint8
		expected  uint8
	}{
		{0, 0, 0x00},
		{15, 15, 0xFF},
		{0, 15, 0xF0},
		{15, 0, 0x0F},
		{5, 10, 0xA5},
		{10, 5, 0x5A},
	}

	for _, tt := range tests {
		result := PackInt4(tt.low, tt.high)
		assert.Equal(t, tt.expected, result,
			"PackInt4(%d, %d) = 0x%X, want 0x%X", tt.low, tt.high, result, tt.expected)
	}
}

func TestUnpackInt4(t *testing.T) {
	tests := []struct {
		packed       uint8
		expectedLow  uint8
		expectedHigh uint8
	}{
		{0x00, 0, 0},
		{0xFF, 15, 15},
		{0xF0, 0, 15},
		{0x0F, 15, 0},
		{0xA5, 5, 10},
		{0x5A, 10, 5},
	}

	for _, tt := range tests {
		low, high := UnpackInt4(tt.packed)
		assert.Equal(t, tt.expectedLow, low,
			"UnpackInt4(0x%X) low = %d, want %d", tt.packed, low, tt.expectedLow)
		assert.Equal(t, tt.expectedHigh, high,
			"UnpackInt4(0x%X) high = %d, want %d", tt.packed, high, tt.expectedHigh)
	}
}

func TestPackUnpackRoundTrip(t *testing.T) {
	for low := uint8(0); low < 16; low++ {
		for high := uint8(0); high < 16; high++ {
			packed := PackInt4(low, high)
			gotLow, gotHigh := UnpackInt4(packed)
			assert.Equal(t, low, gotLow, "roundtrip failed for low=%d, high=%d", low, high)
			assert.Equal(t, high, gotHigh, "roundtrip failed for low=%d, high=%d", low, high)
		}
	}
}

func TestPackInt4Slice(t *testing.T) {
	tests := []struct {
		values   []uint8
		expected []uint8
	}{
		{[]uint8{0, 0}, []uint8{0x00}},
		{[]uint8{15, 15}, []uint8{0xFF}},
		{[]uint8{1, 2, 3, 4}, []uint8{0x21, 0x43}},
		{[]uint8{0, 1, 2, 3, 4, 5}, []uint8{0x10, 0x32, 0x54}},
		// Odd length - last byte's high nibble is 0
		{[]uint8{5}, []uint8{0x05}},
		{[]uint8{1, 2, 3}, []uint8{0x21, 0x03}},
	}

	for _, tt := range tests {
		result := PackInt4Slice(tt.values)
		assert.Equal(t, tt.expected, result, "PackInt4Slice(%v)", tt.values)
	}
}

func TestUnpackInt4Slice(t *testing.T) {
	tests := []struct {
		packed   []uint8
		count    int
		expected []uint8
	}{
		{[]uint8{0x00}, 2, []uint8{0, 0}},
		{[]uint8{0xFF}, 2, []uint8{15, 15}},
		{[]uint8{0x21, 0x43}, 4, []uint8{1, 2, 3, 4}},
		{[]uint8{0x10, 0x32, 0x54}, 6, []uint8{0, 1, 2, 3, 4, 5}},
		// Partial unpack
		{[]uint8{0x21, 0x43}, 3, []uint8{1, 2, 3}},
		{[]uint8{0x21}, 1, []uint8{1}},
	}

	for _, tt := range tests {
		result := UnpackInt4Slice(tt.packed, tt.count)
		assert.Equal(t, tt.expected, result,
			"UnpackInt4Slice(%v, %d)", tt.packed, tt.count)
	}
}

func TestPackInt4SliceRoundTrip(t *testing.T) {
	// Test with various lengths
	for length := 1; length <= 16; length++ {
		values := make([]uint8, length)
		for i := range values {
			values[i] = uint8(i % 16)
		}

		packed := PackInt4Slice(values)
		unpacked := UnpackInt4Slice(packed, length)

		assert.Equal(t, values, unpacked,
			"roundtrip failed for length=%d", length)
	}
}

func TestPackSignedInt4Slice(t *testing.T) {
	tests := []struct {
		values   []int8
		expected []uint8
	}{
		// -8 -> 0, 7 -> 15
		{[]int8{0, 0}, []uint8{0x88}},          // 0+8=8 -> 0x88
		{[]int8{-8, 7}, []uint8{0xF0}},         // -8+8=0, 7+8=15 -> 0xF0
		{[]int8{-1, 1}, []uint8{0x97}},         // -1+8=7, 1+8=9 -> 0x97
		{[]int8{-4, 4, 0, -2}, []uint8{0xC4, 0x68}}, // -4+8=4, 4+8=12, 0+8=8, -2+8=6
	}

	for _, tt := range tests {
		result := PackSignedInt4Slice(tt.values)
		assert.Equal(t, tt.expected, result, "PackSignedInt4Slice(%v)", tt.values)
	}
}

func TestUnpackSignedInt4Slice(t *testing.T) {
	tests := []struct {
		packed   []uint8
		count    int
		expected []int8
	}{
		{[]uint8{0x88}, 2, []int8{0, 0}},
		{[]uint8{0xF0}, 2, []int8{-8, 7}},
		{[]uint8{0x97}, 2, []int8{-1, 1}},
		{[]uint8{0xC4, 0x68}, 4, []int8{-4, 4, 0, -2}},
	}

	for _, tt := range tests {
		result := UnpackSignedInt4Slice(tt.packed, tt.count)
		assert.Equal(t, tt.expected, result,
			"UnpackSignedInt4Slice(%v, %d)", tt.packed, tt.count)
	}
}

func TestPackedSize(t *testing.T) {
	tests := []struct {
		n        int
		expected int
	}{
		{0, 0},
		{1, 1},
		{2, 1},
		{3, 2},
		{4, 2},
		{5, 3},
		{100, 50},
		{101, 51},
	}

	for _, tt := range tests {
		result := PackedSize(tt.n)
		assert.Equal(t, tt.expected, result, "PackedSize(%d)", tt.n)
	}
}

func TestGetPackedValue(t *testing.T) {
	packed := []uint8{0x21, 0x43, 0x65}

	// Values are: 1, 2, 3, 4, 5, 6
	expected := []uint8{1, 2, 3, 4, 5, 6}
	for i, exp := range expected {
		result := GetPackedValue(packed, i)
		assert.Equal(t, exp, result, "GetPackedValue at index %d", i)
	}

	// Out of bounds should return 0
	assert.Equal(t, uint8(0), GetPackedValue(packed, 10))
}

func TestSetPackedValue(t *testing.T) {
	packed := make([]uint8, 3)

	// Set values: 1, 2, 3, 4, 5, 6
	for i := 0; i < 6; i++ {
		SetPackedValue(packed, i, uint8(i+1))
	}

	assert.Equal(t, []uint8{0x21, 0x43, 0x65}, packed)

	// Overwrite some values
	SetPackedValue(packed, 0, 15)
	SetPackedValue(packed, 3, 0)
	assert.Equal(t, []uint8{0x2F, 0x03, 0x65}, packed)
}

func TestPackNF4Indices(t *testing.T) {
	// NF4 indices are just unsigned 4-bit values (0-15)
	indices := []uint8{0, 7, 8, 15}
	packed := PackNF4Indices(indices)
	unpacked := UnpackNF4Indices(packed, 4)

	assert.Equal(t, indices, unpacked)
}
