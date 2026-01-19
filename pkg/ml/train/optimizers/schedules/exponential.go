// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package schedules

import (
	"github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
)

// Exponential decay specific parameters.
var (
	// ParamExponentialDecayRate is the multiplicative decay rate.
	// Learning rate is multiplied by this value every DecaySteps.
	ParamExponentialDecayRate = "schedule_exponential_decay_rate"

	// ParamExponentialDecaySteps is the number of steps between each decay.
	ParamExponentialDecaySteps = "schedule_exponential_decay_steps"

	// ParamExponentialStaircase enables staircase decay (discrete steps).
	ParamExponentialStaircase = "schedule_exponential_staircase"
)

const (
	// ExponentialScope is the scope name for exponential schedule variables.
	ExponentialScope = "exponential"
)

// ExponentialConfig configures an exponential learning rate decay schedule.
//
// The learning rate decreases exponentially based on the decay rate and decay steps.
//
// Formula:
//
//	lr = initial_lr * decay_rate ^ (step / decay_steps)
//
// If Staircase is enabled:
//
//	lr = initial_lr * decay_rate ^ floor(step / decay_steps)
//
// This creates discrete "staircase" drops in learning rate.
//
// Example: decay_rate=0.96, decay_steps=1000 means:
//   - At step 0: lr = initial_lr
//   - At step 1000: lr = initial_lr * 0.96
//   - At step 2000: lr = initial_lr * 0.96^2
//   - etc.
type ExponentialConfig struct {
	baseConfig
	decayRate  float64
	decaySteps int
	staircase  bool
}

// Exponential creates an exponential learning rate decay schedule configuration.
//
// The schedule only affects training; there is no effect during inference or evaluation.
//
// Example:
//
//	schedules.Exponential(ctx, g, dtypes.Float32).
//		InitialLearningRate(0.1).
//		DecayRate(0.96).
//		DecaySteps(1000).
//		Staircase(true).
//		WarmupSteps(500).
//		Done()
func Exponential(ctx *context.Context, graph *Graph, dtype dtypes.DType) *ExponentialConfig {
	return &ExponentialConfig{
		baseConfig: newBaseConfig(ctx, graph, dtype),
		decayRate:  0.96, // Common default
		decaySteps: 1000,
		staircase:  false,
	}
}

// FromContext reads configuration from context parameters.
//
// Parameters read:
//   - optimizers.ParamLearningRate: initial learning rate
//   - ParamWarmupSteps: warmup steps (default: 0)
//   - ParamExponentialDecayRate: decay rate (default: 0.96)
//   - ParamExponentialDecaySteps: decay steps (default: 1000)
//   - ParamExponentialStaircase: staircase mode (default: false)
func (c *ExponentialConfig) FromContext() *ExponentialConfig {
	c.readCommonFromContext()
	c.decayRate = context.GetParamOr(c.ctx, ParamExponentialDecayRate, c.decayRate)
	c.decaySteps = context.GetParamOr(c.ctx, ParamExponentialDecaySteps, c.decaySteps)
	c.staircase = context.GetParamOr(c.ctx, ParamExponentialStaircase, c.staircase)
	return c
}

// InitialLearningRate sets the starting learning rate (at the end of warmup).
// If not set, reads from context parameter optimizers.ParamLearningRate.
func (c *ExponentialConfig) InitialLearningRate(lr float64) *ExponentialConfig {
	if lr < 0 {
		exceptions.Panicf("InitialLearningRate must be >= 0, got %g", lr)
	}
	c.initialLR = lr
	return c
}

// DecayRate sets the multiplicative decay factor.
// The learning rate is multiplied by this value every DecaySteps.
//
// Common values:
//   - 0.96: Gentle decay (about 4% reduction per decay period)
//   - 0.9: Moderate decay (10% reduction per decay period)
//   - 0.5: Aggressive decay (halves every decay period)
//
// Default is 0.96.
func (c *ExponentialConfig) DecayRate(rate float64) *ExponentialConfig {
	if rate <= 0 || rate > 1 {
		exceptions.Panicf("DecayRate must be in (0, 1], got %g", rate)
	}
	c.decayRate = rate
	return c
}

// DecaySteps sets how many steps between each application of the decay rate.
// Default is 1000.
func (c *ExponentialConfig) DecaySteps(steps int) *ExponentialConfig {
	if steps <= 0 {
		exceptions.Panicf("DecaySteps must be > 0, got %d", steps)
	}
	c.decaySteps = steps
	return c
}

// Staircase enables discrete step decay.
// When true, the exponent is floor(step/decay_steps) instead of step/decay_steps,
// creating a staircase pattern where the learning rate stays constant for
// decay_steps iterations before dropping.
// Default is false (continuous decay).
func (c *ExponentialConfig) Staircase(enabled bool) *ExponentialConfig {
	c.staircase = enabled
	return c
}

// WarmupSteps sets the number of warmup steps.
// During warmup, learning rate increases linearly from 0 to InitialLearningRate.
// Default is 0 (no warmup).
func (c *ExponentialConfig) WarmupSteps(steps int) *ExponentialConfig {
	if steps < 0 {
		exceptions.Panicf("WarmupSteps must be >= 0, got %d", steps)
	}
	c.warmupSteps = steps
	return c
}

// Done finalizes the configuration and builds the computation graph
// for the exponential learning rate schedule.
//
// If not in training mode, this is a no-op.
func (c *ExponentialConfig) Done() {
	if !c.isEnabled() {
		return
	}

	// Validate configuration
	if c.initialLR < 0 {
		c.initialLR = context.GetParamOr(c.ctx, optimizers.ParamLearningRate, 0.0)
	}
	if c.initialLR <= 0 {
		exceptions.Panicf("exponential schedule: learning rate not configured and not set in context as %q",
			optimizers.ParamLearningRate)
		return
	}

	g := c.graph

	// Get current step (schedule maintains its own counter)
	step := c.getStepNode(ExponentialScope)

	// Adjusted step for decay calculation (step - warmup)
	adjustedStep := c.getAdjustedStep(step)
	adjustedStep = MaxScalar(adjustedStep, 0.0) // Ensure non-negative

	// Calculate exponent: step / decay_steps
	exponent := DivScalar(adjustedStep, float64(c.decaySteps))

	// Apply staircase if enabled
	if c.staircase {
		exponent = Floor(exponent)
	}

	// lr = initial_lr * decay_rate ^ exponent
	// = initial_lr * exp(log(decay_rate) * exponent)
	decayRateNode := ConvertDType(Const(g, c.decayRate), c.dtype)
	decayFactor := Pow(decayRateNode, exponent)
	lr := MulScalar(decayFactor, c.initialLR)

	// Apply warmup if configured
	lr = c.applyWarmup(lr, step, c.initialLR)

	// Update the learning rate variable
	c.updateLearningRate(lr)
}
