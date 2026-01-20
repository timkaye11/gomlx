// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package dora

import (
	"github.com/gomlx/gomlx/pkg/ml/adapters"
	"github.com/gomlx/gomlx/pkg/ml/adapters/lora"
	"github.com/pkg/errors"
)

// Config holds the configuration for DoRA.
// It embeds LoRA configuration and adds magnitude-specific settings.
type Config struct {
	// LoRA configuration (rank, alpha, dropout, etc.)
	*lora.Config

	// TrainMagnitude indicates whether the magnitude vector is trainable.
	// Default is true.
	TrainMagnitude bool

	// InitMagnitudeFromPretrained indicates whether to initialize magnitude
	// from the pretrained weight column norms. Default is true.
	// If false, magnitude is initialized to 1.
	InitMagnitudeFromPretrained bool

	// MagnitudeRegularization is the L2 regularization strength on magnitude changes.
	// Default is 0 (no regularization).
	MagnitudeRegularization float64
}

// NewConfig creates a DoRA configuration with sensible defaults.
//
// Default values:
//   - Rank: 8
//   - Alpha: 16.0
//   - Dropout: 0.0
//   - TrainMagnitude: true
//   - InitMagnitudeFromPretrained: true
//   - MagnitudeRegularization: 0.0
func NewConfig() *Config {
	return &Config{
		Config:                      lora.NewConfig(),
		TrainMagnitude:              true,
		InitMagnitudeFromPretrained: true,
		MagnitudeRegularization:     0.0,
	}
}

// Type returns the adapter type (DoRA).
func (c *Config) Type() adapters.AdapterType {
	return adapters.AdapterTypeDoRA
}

// Validate checks if the configuration is valid.
func (c *Config) Validate() error {
	if err := c.Config.Validate(); err != nil {
		return err
	}
	if c.MagnitudeRegularization < 0 {
		return errors.Errorf("magnitude regularization must be >= 0, got %f", c.MagnitudeRegularization)
	}
	return nil
}

// Clone creates a deep copy of the configuration.
func (c *Config) Clone() adapters.AdapterConfig {
	return &Config{
		Config:                      c.Config.Clone().(*lora.Config),
		TrainMagnitude:              c.TrainMagnitude,
		InitMagnitudeFromPretrained: c.InitMagnitudeFromPretrained,
		MagnitudeRegularization:     c.MagnitudeRegularization,
	}
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

// SetTrainMagnitude sets whether the magnitude vector is trainable.
func (c *Config) SetTrainMagnitude(train bool) *Config {
	c.TrainMagnitude = train
	return c
}

// SetInitMagnitudeFromPretrained sets whether to initialize from pretrained weights.
func (c *Config) SetInitMagnitudeFromPretrained(init bool) *Config {
	c.InitMagnitudeFromPretrained = init
	return c
}

// SetMagnitudeRegularization sets the L2 regularization on magnitude changes.
func (c *Config) SetMagnitudeRegularization(reg float64) *Config {
	if reg < 0 {
		panic("magnitude regularization must be >= 0")
	}
	c.MagnitudeRegularization = reg
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
