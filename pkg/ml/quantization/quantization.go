// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package quantization provides weight quantization for memory-efficient model storage.
//
// This package implements various quantization schemes used for Parameter-Efficient
// Fine-Tuning (PEFT), particularly QLoRA which uses 4-bit NormalFloat (NF4) quantization.
//
// Supported quantization types:
//   - NF4 (4-bit Normal Float): Optimal for normally distributed weights
//   - Int4 (4-bit Integer): Symmetric quantization
//   - Int8 (8-bit Integer): Higher precision quantization
//
// Basic usage:
//
//	config := quantization.NewConfig().SetQuantType(quantization.NF4).SetGroupSize(64)
//	quantized, err := quantization.Quantize(tensor, config)
//	dequantized := quantization.Dequantize(g, quantized)
package quantization

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/pkg/errors"
)

// QuantType identifies the quantization method.
type QuantType int

const (
	// NF4 is 4-bit Normal Float quantization.
	// Uses 16 fixed values derived from the normal distribution quantiles.
	// Optimal for normally distributed weights (common in neural networks).
	NF4 QuantType = iota

	// Int4 is 4-bit symmetric integer quantization.
	// Maps values to [-8, 7] range using per-group scaling.
	Int4

	// Int8 is 8-bit symmetric integer quantization.
	// Higher precision than 4-bit methods, maps to [-128, 127].
	Int8
)

// String returns the string representation of a QuantType.
func (qt QuantType) String() string {
	switch qt {
	case NF4:
		return "nf4"
	case Int4:
		return "int4"
	case Int8:
		return "int8"
	default:
		return "unknown"
	}
}

// Granularity specifies the scale computation granularity.
type Granularity int

const (
	// PerTensor uses a single scale for the entire tensor.
	PerTensor Granularity = iota

	// PerChannel uses one scale per output channel.
	PerChannel

	// PerGroup uses one scale per group of elements (most common for QLoRA).
	PerGroup
)

// Config holds the configuration for quantization.
type Config struct {
	// QuantType specifies the quantization method.
	QuantType QuantType

	// Granularity specifies how scales are computed.
	Granularity Granularity

	// GroupSize is the number of elements per scale (for PerGroup granularity).
	// Common values: 32, 64, 128. Default is 64.
	GroupSize int

	// ComputeDType is the dtype used for dequantized computation.
	// Typically BFloat16 or Float16 for QLoRA, Float32 for full precision.
	ComputeDType dtypes.DType
}

// NewConfig creates a Config with sensible defaults for QLoRA.
//
// Default values:
//   - QuantType: NF4
//   - Granularity: PerGroup
//   - GroupSize: 64
//   - ComputeDType: BFloat16
func NewConfig() *Config {
	return &Config{
		QuantType:    NF4,
		Granularity:  PerGroup,
		GroupSize:    64,
		ComputeDType: dtypes.BFloat16,
	}
}

// SetQuantType sets the quantization type.
func (c *Config) SetQuantType(qt QuantType) *Config {
	c.QuantType = qt
	return c
}

// SetGranularity sets the scale computation granularity.
func (c *Config) SetGranularity(g Granularity) *Config {
	c.Granularity = g
	return c
}

// SetGroupSize sets the group size for PerGroup quantization.
func (c *Config) SetGroupSize(size int) *Config {
	if size <= 0 {
		panic("group size must be > 0")
	}
	c.GroupSize = size
	return c
}

// SetComputeDType sets the computation dtype for dequantization.
func (c *Config) SetComputeDType(dtype dtypes.DType) *Config {
	c.ComputeDType = dtype
	return c
}

// Validate checks if the configuration is valid.
func (c *Config) Validate() error {
	if c.QuantType < NF4 || c.QuantType > Int8 {
		return errors.Errorf("invalid quantization type: %d", c.QuantType)
	}
	if c.Granularity == PerGroup && c.GroupSize <= 0 {
		return errors.Errorf("group size must be > 0 for PerGroup granularity, got %d", c.GroupSize)
	}
	if !c.ComputeDType.IsFloat() {
		return errors.Errorf("compute dtype must be floating point, got %s", c.ComputeDType)
	}
	return nil
}

// BitsPerWeight returns the number of bits per weight for this quantization type.
func (c *Config) BitsPerWeight() int {
	switch c.QuantType {
	case NF4, Int4:
		return 4
	case Int8:
		return 8
	default:
		return 0
	}
}

// QuantizedWeight holds a quantized weight tensor with its metadata.
type QuantizedWeight struct {
	// PackedData contains the quantized values.
	// For 4-bit quantization, two values are packed per byte.
	PackedData *tensors.Tensor

	// Scales contains the per-group scaling factors.
	// Shape depends on granularity:
	//   - PerTensor: [1]
	//   - PerChannel: [outFeatures]
	//   - PerGroup: [outFeatures, numGroups]
	Scales *tensors.Tensor

	// ZeroPoints contains the zero-point offsets (nil for symmetric quantization like NF4).
	// Same shape as Scales.
	ZeroPoints *tensors.Tensor

	// OriginalShape is the shape of the original unquantized tensor.
	OriginalShape shapes.Shape

	// Config is the quantization configuration used.
	Config *Config
}

// NumGroups returns the number of groups for per-group quantization.
func (qw *QuantizedWeight) NumGroups() int {
	if qw.Config.Granularity != PerGroup {
		return 1
	}
	inFeatures := qw.OriginalShape.Dimensions[0]
	return (inFeatures + qw.Config.GroupSize - 1) / qw.Config.GroupSize
}

// CompressionRatio returns the memory compression ratio compared to float32.
func (qw *QuantizedWeight) CompressionRatio() float64 {
	originalBytes := qw.OriginalShape.Size() * 4 // float32 = 4 bytes
	var quantizedBytes int

	// Packed data bytes
	packedShape := qw.PackedData.Shape()
	quantizedBytes += packedShape.Size()

	// Scales bytes (float16 typically)
	if qw.Scales != nil {
		quantizedBytes += qw.Scales.Shape().Size() * 2 // assume float16
	}

	// Zero points bytes (if present)
	if qw.ZeroPoints != nil {
		quantizedBytes += qw.ZeroPoints.Shape().Size() * 2
	}

	return float64(originalBytes) / float64(quantizedBytes)
}

// MemoryBytes returns the total memory used by this quantized weight.
func (qw *QuantizedWeight) MemoryBytes() int {
	bytes := qw.PackedData.Shape().Size()
	if qw.Scales != nil {
		bytes += qw.Scales.Shape().Size() * qw.Scales.Shape().DType.Size()
	}
	if qw.ZeroPoints != nil {
		bytes += qw.ZeroPoints.Shape().Size() * qw.ZeroPoints.Shape().DType.Size()
	}
	return bytes
}
