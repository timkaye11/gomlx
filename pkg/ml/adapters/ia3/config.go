// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package ia3

import (
	"github.com/gomlx/gomlx/pkg/ml/adapters"
)

// Config holds the configuration for IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations).
//
// IA³ is a parameter-efficient fine-tuning method that uses learned rescaling vectors
// instead of low-rank matrix decomposition. It rescales key, value, and feedforward
// activations with element-wise multiplication:
//
//   - Key: K' = l_k * K
//   - Value: V' = l_v * V
//   - FFN intermediate: h' = l_ff * h
//
// Where l_k, l_v, l_ff are learned vectors.
//
// IA³ has significantly fewer parameters than LoRA (only scaling vectors, not matrices),
// making it extremely lightweight while still achieving competitive performance.
//
// Reference: "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning"
// https://arxiv.org/abs/2205.05638
type Config struct {
	adapters.BaseConfig

	// InitScale is the initial value for the scaling vectors.
	// The paper recommends initializing to 1.0 so the adapter starts as identity.
	// Default: 1.0
	initScale float64

	// Dropout rate applied to the scaling vectors during training.
	// Set to 0 to disable.
	// Default: 0.0
	dropout float64

	// FeedForwardModules specifies which modules in the FFN should be adapted.
	// In transformers, this is typically the intermediate layer before the down projection.
	// Default: ["mlp", "feed_forward", "fc1"]
	feedForwardModules []string

	// AttentionModules specifies which attention modules should be adapted.
	// Typically key and value projections.
	// Default: ["key", "value", "k_proj", "v_proj"]
	attentionModules []string
}

// NewConfig creates a new IA³ configuration with sensible defaults.
//
// Default values:
//   - InitScale: 1.0 (start as identity transform)
//   - Dropout: 0.0
//   - Enabled: true
func NewConfig() *Config {
	return &Config{
		BaseConfig: adapters.NewBaseConfig(),
		initScale:  1.0,
		dropout:    0.0,
		feedForwardModules: []string{
			"mlp", "feed_forward", "fc1", "intermediate",
		},
		attentionModules: []string{
			"key", "value", "k_proj", "v_proj",
		},
	}
}

// Type returns the adapter type.
func (c *Config) Type() adapters.AdapterType {
	return adapters.AdapterTypeIA3
}

// Validate checks if the configuration is valid.
func (c *Config) Validate() error {
	adapters.ValidateDropout(c.dropout)
	if c.initScale == 0 {
		panic("IA³ init scale cannot be 0")
	}
	return nil
}

// Clone creates a deep copy of the configuration.
func (c *Config) Clone() adapters.AdapterConfig {
	clone := *c
	clone.BaseConfig = c.BaseConfig.Clone()
	if c.feedForwardModules != nil {
		clone.feedForwardModules = make([]string, len(c.feedForwardModules))
		copy(clone.feedForwardModules, c.feedForwardModules)
	}
	if c.attentionModules != nil {
		clone.attentionModules = make([]string, len(c.attentionModules))
		copy(clone.attentionModules, c.attentionModules)
	}
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

// InitScale returns the initial value for scaling vectors.
func (c *Config) InitScale() float64 {
	return c.initScale
}

// SetInitScale sets the initial scale value and returns the config for chaining.
func (c *Config) SetInitScale(scale float64) *Config {
	if scale == 0 {
		panic("IA³ init scale cannot be 0")
	}
	c.initScale = scale
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

// FeedForwardModules returns the FFN module patterns to adapt.
func (c *Config) FeedForwardModules() []string {
	return c.feedForwardModules
}

// SetFeedForwardModules sets the FFN module patterns and returns the config for chaining.
func (c *Config) SetFeedForwardModules(modules ...string) *Config {
	c.feedForwardModules = modules
	return c
}

// AttentionModules returns the attention module patterns to adapt.
func (c *Config) AttentionModules() []string {
	return c.attentionModules
}

// SetAttentionModules sets the attention module patterns and returns the config for chaining.
func (c *Config) SetAttentionModules(modules ...string) *Config {
	c.attentionModules = modules
	return c
}
