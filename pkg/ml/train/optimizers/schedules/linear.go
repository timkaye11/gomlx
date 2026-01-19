// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package schedules

import (
	"github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
)

const (
	// LinearScope is the scope name for linear schedule variables.
	LinearScope = "linear"
)

// LinearConfig configures a linear learning rate decay schedule.
//
// The learning rate decreases linearly from InitialLearningRate to FinalLearningRate
// over TotalSteps training steps.
//
// If WarmupSteps > 0, the learning rate first increases linearly from FinalLearningRate
// (or 0) to InitialLearningRate during the warmup period, then decays linearly.
//
// Formula (after warmup):
//
//	progress = (step - warmup_steps) / (total_steps - warmup_steps)
//	lr = initial_lr + (final_lr - initial_lr) * progress
//	   = initial_lr * (1 - progress) + final_lr * progress
//
// This schedule is commonly used for fine-tuning transformer models like BERT,
// where linear warmup followed by linear decay is the standard approach.
type LinearConfig struct {
	baseConfig
}

// Linear creates a linear learning rate decay schedule configuration.
//
// The schedule only affects training; there is no effect during inference or evaluation.
//
// Example:
//
//	schedules.Linear(ctx, g, dtypes.Float32).
//		InitialLearningRate(1e-4).
//		FinalLearningRate(0).
//		WarmupSteps(1000).
//		TotalSteps(10000).
//		Done()
func Linear(ctx *context.Context, graph *Graph, dtype dtypes.DType) *LinearConfig {
	return &LinearConfig{
		baseConfig: newBaseConfig(ctx, graph, dtype),
	}
}

// FromContext reads configuration from context parameters.
//
// Parameters read:
//   - optimizers.ParamLearningRate: initial learning rate
//   - ParamFinalLearningRate: final learning rate (default: 0)
//   - ParamWarmupSteps: warmup steps (default: 0)
//   - ParamTotalSteps: total training steps
func (c *LinearConfig) FromContext() *LinearConfig {
	c.readCommonFromContext()
	return c
}

// InitialLearningRate sets the starting learning rate (at the end of warmup).
// If not set, reads from context parameter optimizers.ParamLearningRate.
func (c *LinearConfig) InitialLearningRate(lr float64) *LinearConfig {
	if lr < 0 {
		exceptions.Panicf("InitialLearningRate must be >= 0, got %g", lr)
	}
	c.initialLR = lr
	return c
}

// FinalLearningRate sets the ending learning rate. Default is 0.
func (c *LinearConfig) FinalLearningRate(lr float64) *LinearConfig {
	if lr < 0 {
		exceptions.Panicf("FinalLearningRate must be >= 0, got %g", lr)
	}
	c.finalLR = lr
	return c
}

// WarmupSteps sets the number of warmup steps.
// During warmup, learning rate increases linearly from FinalLearningRate to InitialLearningRate.
// Default is 0 (no warmup).
func (c *LinearConfig) WarmupSteps(steps int) *LinearConfig {
	if steps < 0 {
		exceptions.Panicf("WarmupSteps must be >= 0, got %d", steps)
	}
	c.warmupSteps = steps
	return c
}

// TotalSteps sets the total number of training steps.
// The learning rate decays over (TotalSteps - WarmupSteps) steps after warmup.
func (c *LinearConfig) TotalSteps(steps int) *LinearConfig {
	if steps < 0 {
		exceptions.Panicf("TotalSteps must be >= 0, got %d", steps)
	}
	c.totalSteps = steps
	return c
}

// Done finalizes the configuration and builds the computation graph
// for the linear learning rate schedule.
//
// If TotalSteps is 0 or not in training mode, this is a no-op.
func (c *LinearConfig) Done() {
	if !c.isEnabled() || c.totalSteps <= 0 {
		return
	}

	// Validate configuration
	if c.initialLR < 0 {
		c.initialLR = context.GetParamOr(c.ctx, optimizers.ParamLearningRate, 0.0)
	}
	if c.initialLR <= 0 {
		exceptions.Panicf("linear schedule: learning rate not configured and not set in context as %q",
			optimizers.ParamLearningRate)
		return
	}
	if c.warmupSteps >= c.totalSteps {
		exceptions.Panicf("linear schedule: warmup_steps (%d) must be < total_steps (%d)",
			c.warmupSteps, c.totalSteps)
		return
	}

	// Get current step (schedule maintains its own counter)
	step := c.getStepNode(LinearScope)

	// Calculate decay steps (total - warmup)
	// Note: getAdjustedTotalSteps() will return totalSteps - warmupSteps.
	// If this is <= 0, it means warmupSteps >= totalSteps, which is caught
	// by the validation above. This check is defensive programming.
	decaySteps := c.getAdjustedTotalSteps()
	if decaySteps <= 0 {
		exceptions.Panicf("linear schedule: invalid configuration - totalSteps (%d) must be greater than warmupSteps (%d)",
			c.totalSteps, c.warmupSteps)
	}

	// Adjusted step for decay calculation (step - warmup)
	adjustedStep := c.getAdjustedStep(step)

	// Calculate progress: clamped to [0, 1]
	// progress = max(0, min(1, adjusted_step / decay_steps))
	progress := DivScalar(adjustedStep, float64(decaySteps))
	progress = MaxScalar(progress, 0.0)
	progress = MinScalar(progress, 1.0)

	// Linear interpolation: lr = initial + (final - initial) * progress
	// = initial * (1 - progress) + final * progress
	lr := AddScalar(MulScalar(progress, c.finalLR-c.initialLR), c.initialLR)

	// Apply warmup if configured
	lr = c.applyWarmup(lr, step, c.initialLR)

	// Update the learning rate variable
	c.updateLearningRate(lr)
}
