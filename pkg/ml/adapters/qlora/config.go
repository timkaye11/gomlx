// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package qlora

import (
	"github.com/gomlx/gomlx/pkg/ml/adapters"
	"github.com/gomlx/gomlx/pkg/ml/adapters/lora"
	"github.com/gomlx/gomlx/pkg/ml/quantization"
)

// Config holds the configuration for QLoRA.
// It embeds LoRA configuration and adds quantization settings.
type Config struct {
	// LoRA configuration (rank, alpha, dropout, etc.)
	*lora.Config

	// QuantConfig specifies how base weights are quantized.
	QuantConfig *quantization.Config
}

// NewConfig creates a QLoRA configuration with sensible defaults.
//
// Default values:
//   - Rank: 8
//   - Alpha: 16.0
//   - Dropout: 0.0
//   - QuantType: NF4
//   - GroupSize: 64
//   - ComputeDType: BFloat16
func NewConfig() *Config {
	return &Config{
		Config:      lora.NewConfig(),
		QuantConfig: quantization.NewConfig(),
	}
}

// Type returns the adapter type (QLoRA).
func (c *Config) Type() adapters.AdapterType {
	return adapters.AdapterTypeQLoRA
}

// Validate checks if the configuration is valid.
func (c *Config) Validate() error {
	if err := c.Config.Validate(); err != nil {
		return err
	}
	if c.QuantConfig != nil {
		if err := c.QuantConfig.Validate(); err != nil {
			return err
		}
	}
	return nil
}

// Clone creates a deep copy of the configuration.
func (c *Config) Clone() adapters.AdapterConfig {
	clone := &Config{
		Config: c.Config.Clone().(*lora.Config),
	}
	if c.QuantConfig != nil {
		clone.QuantConfig = &quantization.Config{
			QuantType:    c.QuantConfig.QuantType,
			Granularity:  c.QuantConfig.Granularity,
			GroupSize:    c.QuantConfig.GroupSize,
			ComputeDType: c.QuantConfig.ComputeDType,
		}
	}
	return clone
}

// SetEnabled enables or disables the adapter.
func (c *Config) SetEnabled(enabled bool) adapters.AdapterConfig {
	c.Config.SetEnabled(enabled)
	return c
}

// SetTargetModules sets the target modules for adaptation.
func (c *Config) SetTargetModules(modules ...string) adapters.AdapterConfig {
	c.Config.SetTargetModules(modules...)
	return c
}

// SetQuantType sets the quantization type.
func (c *Config) SetQuantType(qt quantization.QuantType) *Config {
	c.QuantConfig.QuantType = qt
	return c
}

// SetGroupSize sets the group size for per-group quantization.
func (c *Config) SetGroupSize(size int) *Config {
	c.QuantConfig.SetGroupSize(size)
	return c
}

// Setters that return *Config for chaining

// SetRank sets the LoRA rank.
func (c *Config) SetRank(rank int) *Config {
	c.Config.SetRank(rank)
	return c
}

// SetAlpha sets the LoRA scaling factor.
func (c *Config) SetAlpha(alpha float64) *Config {
	c.Config.SetAlpha(alpha)
	return c
}

// SetDropout sets the dropout probability.
func (c *Config) SetDropout(dropout float64) *Config {
	c.Config.SetDropout(dropout)
	return c
}
