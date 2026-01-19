// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package schedules

import (
	"github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
)

// ParamPolynomialPower is the context parameter for polynomial decay power.
// Default is 1.0 (equivalent to linear decay).
var ParamPolynomialPower = "schedule_polynomial_power"

const (
	// PolynomialScope is the scope name for polynomial schedule variables.
	PolynomialScope = "polynomial"
)

// PolynomialConfig configures a polynomial learning rate decay schedule.
//
// The learning rate decreases following a polynomial curve from InitialLearningRate
// to FinalLearningRate over TotalSteps training steps.
//
// Formula (after warmup):
//
//	progress = (step - warmup_steps) / (total_steps - warmup_steps)
//	decay = (1 - progress) ^ power
//	lr = (initial_lr - final_lr) * decay + final_lr
//
// Common power values:
//   - power = 1.0: Linear decay (equivalent to Linear schedule)
//   - power = 2.0: Quadratic decay (slower at start, faster at end)
//   - power = 0.5: Square root decay (faster at start, slower at end)
//
// This schedule is commonly used for fine-tuning language models.
// TensorFlow and HuggingFace use polynomial decay with power=1.0 by default for BERT.
type PolynomialConfig struct {
	baseConfig
	power float64
	cycle bool // Whether to cycle the decay after reaching totalSteps
}

// Polynomial creates a polynomial learning rate decay schedule configuration.
//
// The schedule only affects training; there is no effect during inference or evaluation.
//
// Example:
//
//	schedules.Polynomial(ctx, g, dtypes.Float32).
//		InitialLearningRate(1e-4).
//		FinalLearningRate(1e-6).
//		Power(2.0).  // Quadratic decay
//		WarmupSteps(1000).
//		TotalSteps(10000).
//		Done()
func Polynomial(ctx *context.Context, graph *Graph, dtype dtypes.DType) *PolynomialConfig {
	return &PolynomialConfig{
		baseConfig: newBaseConfig(ctx, graph, dtype),
		power:      1.0, // Default to linear
		cycle:      false,
	}
}

// FromContext reads configuration from context parameters.
//
// Parameters read:
//   - optimizers.ParamLearningRate: initial learning rate
//   - ParamFinalLearningRate: final learning rate (default: 0)
//   - ParamWarmupSteps: warmup steps (default: 0)
//   - ParamTotalSteps: total training steps
//   - ParamPolynomialPower: polynomial power (default: 1.0)
func (c *PolynomialConfig) FromContext() *PolynomialConfig {
	c.readCommonFromContext()
	c.power = context.GetParamOr(c.ctx, ParamPolynomialPower, c.power)
	return c
}

// InitialLearningRate sets the starting learning rate (at the end of warmup).
// If not set, reads from context parameter optimizers.ParamLearningRate.
func (c *PolynomialConfig) InitialLearningRate(lr float64) *PolynomialConfig {
	if lr < 0 {
		exceptions.Panicf("InitialLearningRate must be >= 0, got %g", lr)
	}
	c.initialLR = lr
	return c
}

// FinalLearningRate sets the ending learning rate. Default is 0.
func (c *PolynomialConfig) FinalLearningRate(lr float64) *PolynomialConfig {
	if lr < 0 {
		exceptions.Panicf("FinalLearningRate must be >= 0, got %g", lr)
	}
	c.finalLR = lr
	return c
}

// Power sets the polynomial power for the decay curve.
//
//   - power = 1.0: Linear decay
//   - power = 2.0: Quadratic decay (slower start, faster end)
//   - power = 0.5: Square root decay (faster start, slower end)
//
// Default is 1.0.
func (c *PolynomialConfig) Power(power float64) *PolynomialConfig {
	if power <= 0 {
		exceptions.Panicf("Power must be > 0, got %g", power)
	}
	c.power = power
	return c
}

// Cycle enables cycling: after reaching totalSteps, the decay restarts.
// This can be useful for cyclic training or when training duration is uncertain.
// Default is false (no cycling).
func (c *PolynomialConfig) Cycle(enabled bool) *PolynomialConfig {
	c.cycle = enabled
	return c
}

// WarmupSteps sets the number of warmup steps.
// During warmup, learning rate increases linearly from FinalLearningRate to InitialLearningRate.
// Default is 0 (no warmup).
func (c *PolynomialConfig) WarmupSteps(steps int) *PolynomialConfig {
	if steps < 0 {
		exceptions.Panicf("WarmupSteps must be >= 0, got %d", steps)
	}
	c.warmupSteps = steps
	return c
}

// TotalSteps sets the total number of training steps.
// The learning rate decays over (TotalSteps - WarmupSteps) steps after warmup.
func (c *PolynomialConfig) TotalSteps(steps int) *PolynomialConfig {
	if steps < 0 {
		exceptions.Panicf("TotalSteps must be >= 0, got %d", steps)
	}
	c.totalSteps = steps
	return c
}

// Done finalizes the configuration and builds the computation graph
// for the polynomial learning rate schedule.
//
// If TotalSteps is 0 or not in training mode, this is a no-op.
func (c *PolynomialConfig) Done() {
	if !c.isEnabled() || c.totalSteps <= 0 {
		return
	}

	// Validate configuration
	if c.initialLR < 0 {
		c.initialLR = context.GetParamOr(c.ctx, optimizers.ParamLearningRate, 0.0)
	}
	if c.initialLR <= 0 {
		exceptions.Panicf("polynomial schedule: learning rate not configured and not set in context as %q",
			optimizers.ParamLearningRate)
		return
	}
	if c.warmupSteps >= c.totalSteps {
		exceptions.Panicf("polynomial schedule: warmup_steps (%d) must be < total_steps (%d)",
			c.warmupSteps, c.totalSteps)
		return
	}

	g := c.graph

	// Get current step (schedule maintains its own counter)
	step := c.getStepNode(PolynomialScope)

	// Calculate decay steps (total - warmup)
	decaySteps := c.getAdjustedTotalSteps()
	if decaySteps <= 0 {
		decaySteps = 1 // Avoid division by zero
	}

	// Adjusted step for decay calculation (step - warmup)
	adjustedStep := c.getAdjustedStep(step)

	// Handle cycling if enabled
	if c.cycle {
		// adjustedStep = adjustedStep % decaySteps
		decayStepsNode := Const(g, float64(decaySteps))
		decayStepsNode = ConvertDType(decayStepsNode, c.dtype)
		cycle := Div(adjustedStep, decayStepsNode)
		cycle = Floor(cycle)
		adjustedStep = Sub(adjustedStep, Mul(cycle, decayStepsNode))
	}

	// Calculate progress: clamped to [0, 1]
	progress := DivScalar(adjustedStep, float64(decaySteps))
	progress = MaxScalar(progress, 0.0)
	progress = MinScalar(progress, 1.0)

	// Polynomial decay: decay = (1 - progress) ^ power
	oneMinusProgress := SubScalar(ScalarOne(g, c.dtype), 0.0)
	oneMinusProgress = Sub(oneMinusProgress, progress)
	oneMinusProgress = MaxScalar(oneMinusProgress, 0.0) // Ensure non-negative for Pow

	var decay *Node
	if c.power == 1.0 {
		// Linear decay, avoid Pow for efficiency
		decay = oneMinusProgress
	} else {
		powerNode := ConvertDType(Const(g, c.power), c.dtype)
		decay = Pow(oneMinusProgress, powerNode)
	}

	// lr = (initial_lr - final_lr) * decay + final_lr
	lr := AddScalar(MulScalar(decay, c.initialLR-c.finalLR), c.finalLR)

	// Apply warmup if configured
	lr = c.applyWarmup(lr, step, c.initialLR)

	// Update the learning rate variable
	c.updateLearningRate(lr)
}
