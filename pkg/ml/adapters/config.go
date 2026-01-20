// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package adapters provides a unified interface for parameter-efficient fine-tuning (PEFT) methods.
//
// This package implements various adapter types including LoRA, QLoRA, AdaLoRA, and DoRA,
// allowing efficient fine-tuning of large models on consumer hardware.
//
// Basic usage:
//
//	config := lora.NewConfig().SetRank(8).SetAlpha(16)
//	manager, err := adapters.GetAdapterModel(ctx, config, "my_adapter")
//	// Now layers using DenseWithAdapters will apply the adapter
package adapters

import (
	"github.com/gomlx/gomlx/internal/exceptions"
)

// AdapterType identifies the type of adapter.
type AdapterType string

const (
	// AdapterTypeLoRA is standard Low-Rank Adaptation.
	AdapterTypeLoRA AdapterType = "lora"

	// AdapterTypeQLoRA is Quantized LoRA with 4-bit base weights.
	AdapterTypeQLoRA AdapterType = "qlora"

	// AdapterTypeAdaLoRA is Adaptive LoRA with dynamic rank allocation.
	AdapterTypeAdaLoRA AdapterType = "adalora"

	// AdapterTypeDoRA is Weight-Decomposed LoRA with magnitude/direction decomposition.
	AdapterTypeDoRA AdapterType = "dora"

	// AdapterTypeIA3 is Infused Adapter by Inhibiting and Amplifying Inner Activations.
	// IA³ uses learned rescaling vectors instead of low-rank matrices.
	AdapterTypeIA3 AdapterType = "ia3"
)

// AdapterConfig is the base interface for all adapter configurations.
// Each adapter type implements this interface with method-specific fields.
type AdapterConfig interface {
	// Type returns the adapter type identifier.
	Type() AdapterType

	// Validate checks if the configuration is valid.
	// Returns an error if any configuration values are invalid.
	Validate() error

	// Clone creates a deep copy of the configuration.
	Clone() AdapterConfig

	// Enabled returns whether the adapter is enabled.
	Enabled() bool

	// SetEnabled toggles the adapter on/off and returns the config for chaining.
	SetEnabled(enabled bool) AdapterConfig

	// TargetModules returns the list of module patterns to apply the adapter to.
	// Empty means apply to all compatible layers.
	TargetModules() []string

	// SetTargetModules sets which modules to target and returns the config for chaining.
	SetTargetModules(modules ...string) AdapterConfig
}

// BaseConfig provides common fields and methods for all adapter configurations.
// Embed this in specific adapter configs to get default implementations.
type BaseConfig struct {
	enabled       bool
	targetModules []string
}

// NewBaseConfig creates a BaseConfig with sensible defaults.
func NewBaseConfig() BaseConfig {
	return BaseConfig{
		enabled:       true,
		targetModules: nil, // nil means all compatible layers
	}
}

// Enabled returns whether the adapter is enabled.
func (c *BaseConfig) Enabled() bool {
	return c.enabled
}

// SetEnabled sets the enabled state.
func (c *BaseConfig) SetEnabled(enabled bool) {
	c.enabled = enabled
}

// TargetModules returns the target module patterns.
func (c *BaseConfig) TargetModules() []string {
	return c.targetModules
}

// SetTargetModules sets the target module patterns.
func (c *BaseConfig) SetTargetModules(modules ...string) {
	c.targetModules = modules
}

// Clone creates a copy of the BaseConfig.
func (c *BaseConfig) Clone() BaseConfig {
	clone := *c
	if c.targetModules != nil {
		clone.targetModules = make([]string, len(c.targetModules))
		copy(clone.targetModules, c.targetModules)
	}
	return clone
}

// ValidateRank checks if a rank value is valid.
func ValidateRank(rank int) {
	if rank <= 0 {
		exceptions.Panicf("adapter rank must be > 0, got %d", rank)
	}
}

// ValidateAlpha checks if an alpha value is reasonable.
func ValidateAlpha(alpha float64) {
	if alpha <= 0 {
		exceptions.Panicf("adapter alpha must be > 0, got %f", alpha)
	}
}

// ValidateDropout checks if a dropout value is valid.
func ValidateDropout(dropout float64) {
	if dropout < 0 || dropout >= 1 {
		exceptions.Panicf("adapter dropout must be in [0, 1), got %f", dropout)
	}
}
