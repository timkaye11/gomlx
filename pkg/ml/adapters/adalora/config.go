// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package adalora

import (
	"github.com/gomlx/gomlx/pkg/ml/adapters"
	"github.com/pkg/errors"
)

// Config holds the configuration for AdaLoRA (Adaptive Low-Rank Adaptation).
//
// AdaLoRA uses SVD-like parameterization with dynamic rank allocation:
//
//	DeltaW = P @ diag(Lambda) @ Q
//
// Where:
//   - P [inputDim, rank]: Left singular vectors (orthogonalized)
//   - Lambda [rank]: Singular values (importance scores)
//   - Q [rank, outputDim]: Right singular vectors (orthogonalized)
//
// During training, low-importance singular values are pruned to meet
// a total rank budget across all adapted layers.
type Config struct {
	adapters.BaseConfig

	// InitialRank is the starting rank before pruning.
	// Should be larger than TargetRank to allow for pruning.
	// Default: 12
	initialRank int

	// TargetRank is the target average rank after pruning.
	// The budget allocation will try to reach this average.
	// Default: 8
	targetRank int

	// Alpha is the scaling factor (same as LoRA).
	// The output is scaled by alpha/initialRank.
	// Default: 16.0
	alpha float64

	// Dropout rate applied to the AdaLoRA path.
	// Default: 0.0
	dropout float64

	// Beta is the EMA (exponential moving average) factor for importance scoring.
	// Higher values = more smoothing of importance scores.
	// Default: 0.85
	Beta float64

	// WarmupSteps is the number of training steps before pruning starts.
	// During warmup, all ranks are active to collect importance scores.
	// Default: 100
	WarmupSteps int

	// BudgetingInterval is the number of steps between budget reallocation.
	// Default: 50
	BudgetingInterval int

	// FinalBudgetingStep is the step at which rank allocation becomes fixed.
	// After this step, no more pruning occurs.
	// Default: 1000
	FinalBudgetingStep int

	// OrthoRegStrength is the strength of orthogonal regularization on P and Q.
	// Encourages P^T @ P ≈ I and Q @ Q^T ≈ I.
	// Default: 0.1
	OrthoRegStrength float64

	// MinRank is the minimum rank any single layer can have after pruning.
	// Default: 1
	MinRank int
}

// NewConfig creates an AdaLoRA configuration with sensible defaults.
//
// Default values:
//   - InitialRank: 12
//   - TargetRank: 8
//   - Alpha: 16.0
//   - Dropout: 0.0
//   - Beta: 0.85
//   - WarmupSteps: 100
//   - BudgetingInterval: 50
//   - FinalBudgetingStep: 1000
//   - OrthoRegStrength: 0.1
//   - MinRank: 1
func NewConfig() *Config {
	return &Config{
		BaseConfig:         adapters.NewBaseConfig(),
		initialRank:        12,
		targetRank:         8,
		alpha:              16.0,
		dropout:            0.0,
		Beta:               0.85,
		WarmupSteps:        100,
		BudgetingInterval:  50,
		FinalBudgetingStep: 1000,
		OrthoRegStrength:   0.1,
		MinRank:            1,
	}
}

// Type returns the adapter type.
func (c *Config) Type() adapters.AdapterType {
	return adapters.AdapterTypeAdaLoRA
}

// Validate checks if the configuration is valid.
func (c *Config) Validate() error {
	if c.initialRank <= 0 {
		return errors.Errorf("initial rank must be > 0, got %d", c.initialRank)
	}
	if c.targetRank <= 0 {
		return errors.Errorf("target rank must be > 0, got %d", c.targetRank)
	}
	if c.targetRank > c.initialRank {
		return errors.Errorf("target rank (%d) cannot exceed initial rank (%d)", c.targetRank, c.initialRank)
	}
	if c.alpha <= 0 {
		return errors.Errorf("alpha must be > 0, got %f", c.alpha)
	}
	if c.dropout < 0 || c.dropout >= 1 {
		return errors.Errorf("dropout must be in [0, 1), got %f", c.dropout)
	}
	if c.Beta < 0 || c.Beta > 1 {
		return errors.Errorf("beta must be in [0, 1], got %f", c.Beta)
	}
	if c.WarmupSteps < 0 {
		return errors.Errorf("warmup steps must be >= 0, got %d", c.WarmupSteps)
	}
	if c.OrthoRegStrength < 0 {
		return errors.Errorf("ortho reg strength must be >= 0, got %f", c.OrthoRegStrength)
	}
	if c.MinRank <= 0 {
		return errors.Errorf("min rank must be > 0, got %d", c.MinRank)
	}
	if c.MinRank > c.targetRank {
		return errors.Errorf("min rank (%d) cannot exceed target rank (%d)", c.MinRank, c.targetRank)
	}
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

// InitialRank returns the initial rank before pruning.
func (c *Config) InitialRank() int {
	return c.initialRank
}

// SetInitialRank sets the initial rank.
func (c *Config) SetInitialRank(rank int) *Config {
	if rank <= 0 {
		panic("initial rank must be > 0")
	}
	c.initialRank = rank
	return c
}

// TargetRank returns the target rank after pruning.
func (c *Config) TargetRank() int {
	return c.targetRank
}

// SetTargetRank sets the target rank.
func (c *Config) SetTargetRank(rank int) *Config {
	if rank <= 0 {
		panic("target rank must be > 0")
	}
	c.targetRank = rank
	return c
}

// Alpha returns the scaling factor.
func (c *Config) Alpha() float64 {
	return c.alpha
}

// SetAlpha sets the alpha value.
func (c *Config) SetAlpha(alpha float64) *Config {
	if alpha <= 0 {
		panic("alpha must be > 0")
	}
	c.alpha = alpha
	return c
}

// Dropout returns the dropout rate.
func (c *Config) Dropout() float64 {
	return c.dropout
}

// SetDropout sets the dropout rate.
func (c *Config) SetDropout(dropout float64) *Config {
	if dropout < 0 || dropout >= 1 {
		panic("dropout must be in [0, 1)")
	}
	c.dropout = dropout
	return c
}

// Scale returns the scaling factor (alpha / initialRank).
func (c *Config) Scale() float64 {
	if c.initialRank == 0 {
		return 0
	}
	return c.alpha / float64(c.initialRank)
}

// SetBeta sets the EMA factor.
func (c *Config) SetBeta(beta float64) *Config {
	if beta < 0 || beta > 1 {
		panic("beta must be in [0, 1]")
	}
	c.Beta = beta
	return c
}

// SetWarmupSteps sets the warmup steps.
func (c *Config) SetWarmupSteps(steps int) *Config {
	if steps < 0 {
		panic("warmup steps must be >= 0")
	}
	c.WarmupSteps = steps
	return c
}

// SetBudgetingInterval sets the budgeting interval.
func (c *Config) SetBudgetingInterval(interval int) *Config {
	if interval <= 0 {
		panic("budgeting interval must be > 0")
	}
	c.BudgetingInterval = interval
	return c
}

// SetFinalBudgetingStep sets the final budgeting step.
func (c *Config) SetFinalBudgetingStep(step int) *Config {
	if step < 0 {
		panic("final budgeting step must be >= 0")
	}
	c.FinalBudgetingStep = step
	return c
}

// SetOrthoRegStrength sets the orthogonal regularization strength.
func (c *Config) SetOrthoRegStrength(strength float64) *Config {
	if strength < 0 {
		panic("ortho reg strength must be >= 0")
	}
	c.OrthoRegStrength = strength
	return c
}

// SetMinRank sets the minimum rank per layer.
func (c *Config) SetMinRank(minRank int) *Config {
	if minRank <= 0 {
		panic("min rank must be > 0")
	}
	c.MinRank = minRank
	return c
}
