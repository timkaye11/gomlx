// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package lora

import (
	"math"

	"github.com/gomlx/gomlx/pkg/ml/adapters"
)

// Config holds the configuration for LoRA (Low-Rank Adaptation).
//
// LoRA decomposes weight updates into low-rank matrices:
//
//	deltaW = scale * (A @ B)
//
// where A has shape [inputDim, rank] and B has shape [rank, outputDim].
// The scale factor is alpha/rank (or alpha/sqrt(rank) for rsLoRA).
type Config struct {
	adapters.BaseConfig

	// Rank is the rank of the low-rank decomposition (r in the paper).
	// Higher rank = more parameters = more capacity.
	// Typical values: 4, 8, 16, 32, 64.
	rank int

	// Alpha is the scaling factor. The LoRA output is scaled by alpha/rank
	// (or alpha/sqrt(rank) if UseRSLoRA is true).
	// This allows tuning the magnitude of LoRA updates relative to pretrained weights.
	// Typical values: 16, 32, or same as rank.
	alpha float64

	// Dropout is the dropout rate applied to the LoRA path (before scaling).
	// Set to 0 to disable.
	dropout float64

	// UseRSLoRA enables Rank-Stabilized LoRA scaling.
	// When true, scale = alpha / sqrt(rank) instead of alpha / rank.
	// This provides more stable training across different rank values.
	// Reference: "A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA"
	// Default: false
	UseRSLoRA bool

	// UseLoRAPlus enables LoRA+ which uses different learning rates for A and B matrices.
	// When true, the B matrix uses learningRate * LoRAPlusRatio.
	// Reference: "LoRA+: Efficient Low Rank Adaptation of Large Models"
	// Default: false
	UseLoRAPlus bool

	// LoRAPlusRatio is the learning rate ratio for B matrix relative to A matrix.
	// Only used when UseLoRAPlus is true.
	// B_lr = A_lr * LoRAPlusRatio
	// Typical value: 16.0 (recommended in the LoRA+ paper)
	// Default: 16.0
	LoRAPlusRatio float64
}

// NewConfig creates a new LoRA configuration with sensible defaults.
//
// Default values:
//   - Rank: 8
//   - Alpha: 16.0
//   - Dropout: 0.0
//   - Enabled: true
//   - UseLoRAPlus: false
//   - LoRAPlusRatio: 16.0 (default when enabled)
func NewConfig() *Config {
	return &Config{
		BaseConfig:    adapters.NewBaseConfig(),
		rank:          8,
		alpha:         16.0,
		dropout:       0.0,
		LoRAPlusRatio: 16.0, // Default ratio when LoRA+ is enabled
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

// Scale returns the scaling factor.
// Returns alpha / rank for standard LoRA, or alpha / sqrt(rank) for rsLoRA.
func (c *Config) Scale() float64 {
	if c.rank == 0 {
		return 0
	}
	if c.UseRSLoRA {
		return c.alpha / math.Sqrt(float64(c.rank))
	}
	return c.alpha / float64(c.rank)
}

// SetUseRSLoRA enables or disables Rank-Stabilized LoRA scaling.
func (c *Config) SetUseRSLoRA(use bool) *Config {
	c.UseRSLoRA = use
	return c
}

// SetUseLoRAPlus enables or disables LoRA+ (different learning rates for A/B).
// When enabled, B matrix will use lr * LoRAPlusRatio while A uses lr.
func (c *Config) SetUseLoRAPlus(use bool) *Config {
	c.UseLoRAPlus = use
	return c
}

// SetLoRAPlusRatio sets the learning rate ratio for LoRA+ B matrix.
// B_lr = A_lr * ratio. Default is 16.0.
func (c *Config) SetLoRAPlusRatio(ratio float64) *Config {
	if ratio <= 0 {
		panic("LoRA+ ratio must be positive")
	}
	c.LoRAPlusRatio = ratio
	return c
}
