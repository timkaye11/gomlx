// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package lora

import (
	"github.com/gomlx/gomlx/pkg/ml/adapters"
)

// Config holds the configuration for LoRA (Low-Rank Adaptation).
//
// LoRA decomposes weight updates into low-rank matrices:
//
//	deltaW = scale * (A @ B)
//
// where A has shape [inputDim, rank] and B has shape [rank, outputDim].
// The scale factor is alpha/rank.
type Config struct {
	adapters.BaseConfig

	// Rank is the rank of the low-rank decomposition (r in the paper).
	// Higher rank = more parameters = more capacity.
	// Typical values: 4, 8, 16, 32, 64.
	rank int

	// Alpha is the scaling factor. The LoRA output is scaled by alpha/rank.
	// This allows tuning the magnitude of LoRA updates relative to pretrained weights.
	// Typical values: 16, 32, or same as rank.
	alpha float64

	// Dropout is the dropout rate applied to the LoRA path (before scaling).
	// Set to 0 to disable.
	dropout float64
}

// NewConfig creates a new LoRA configuration with sensible defaults.
//
// Default values:
//   - Rank: 8
//   - Alpha: 16.0
//   - Dropout: 0.0
//   - Enabled: true
func NewConfig() *Config {
	return &Config{
		BaseConfig: adapters.NewBaseConfig(),
		rank:       8,
		alpha:      16.0,
		dropout:    0.0,
	}
}

// Type returns the adapter type.
func (c *Config) Type() adapters.AdapterType {
	return adapters.AdapterTypeLoRA
}

// Validate checks if the configuration is valid.
func (c *Config) Validate() error {
	adapters.ValidateRank(c.rank)
	adapters.ValidateAlpha(c.alpha)
	adapters.ValidateDropout(c.dropout)
	return nil
}

// Clone creates a deep copy of the configuration.
func (c *Config) Clone() adapters.AdapterConfig {
	clone := *c
	clone.BaseConfig = c.BaseConfig.Clone()
	return &clone
}

// SetEnabled sets the enabled state and returns the config for chaining.
func (c *Config) SetEnabled(enabled bool) adapters.AdapterConfig {
	c.BaseConfig.SetEnabled(enabled)
	return c
}

// SetTargetModules sets the target modules and returns the config for chaining.
func (c *Config) SetTargetModules(modules ...string) adapters.AdapterConfig {
	c.BaseConfig.SetTargetModules(modules...)
	return c
}

// Rank returns the low-rank decomposition rank.
func (c *Config) Rank() int {
	return c.rank
}

// SetRank sets the rank and returns the config for chaining.
func (c *Config) SetRank(rank int) *Config {
	adapters.ValidateRank(rank)
	c.rank = rank
	return c
}

// Alpha returns the scaling factor alpha.
func (c *Config) Alpha() float64 {
	return c.alpha
}

// SetAlpha sets the alpha value and returns the config for chaining.
func (c *Config) SetAlpha(alpha float64) *Config {
	adapters.ValidateAlpha(alpha)
	c.alpha = alpha
	return c
}

// Dropout returns the dropout rate.
func (c *Config) Dropout() float64 {
	return c.dropout
}

// SetDropout sets the dropout rate and returns the config for chaining.
func (c *Config) SetDropout(dropout float64) *Config {
	adapters.ValidateDropout(dropout)
	c.dropout = dropout
	return c
}

// Scale returns the scaling factor (alpha / rank).
func (c *Config) Scale() float64 {
	if c.rank == 0 {
		return 0
	}
	return c.alpha / float64(c.rank)
}
