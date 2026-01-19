// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package schedules

import (
	"sort"

	"github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
)

// Step decay specific parameters.
var (
	// ParamStepDecayFactor is the multiplicative factor applied at each boundary.
	ParamStepDecayFactor = "schedule_step_decay_factor"
)

const (
	// StepScope is the scope name for step schedule variables.
	StepScope = "step"
)

// StepConfig configures a step-based learning rate decay schedule.
//
// The learning rate is reduced by a multiplicative factor at specific step boundaries.
// This is useful when you know specific points in training where you want to reduce
// the learning rate (e.g., at epoch boundaries or specific iterations).
//
// Formula:
//
//	lr = initial_lr * (factor ^ num_boundaries_passed)
//
// Example with boundaries=[1000, 2000, 3000] and factor=0.1:
//   - Steps 0-999: lr = initial_lr
//   - Steps 1000-1999: lr = initial_lr * 0.1
//   - Steps 2000-2999: lr = initial_lr * 0.01
//   - Steps 3000+: lr = initial_lr * 0.001
type StepConfig struct {
	baseConfig
	boundaries []int   // Step numbers at which to reduce LR
	factor     float64 // Multiplicative factor (e.g., 0.1 = reduce by 10x)
}

// Step creates a step-based learning rate decay schedule configuration.
//
// The schedule only affects training; there is no effect during inference or evaluation.
//
// Example:
//
//	schedules.Step(ctx, g, dtypes.Float32).
//		InitialLearningRate(0.1).
//		Boundaries(10000, 20000, 30000).
//		Factor(0.1).
//		WarmupSteps(1000).
//		Done()
func Step(ctx *context.Context, graph *Graph, dtype dtypes.DType) *StepConfig {
	return &StepConfig{
		baseConfig: newBaseConfig(ctx, graph, dtype),
		boundaries: nil,
		factor:     0.1, // Default: reduce by 10x at each boundary
	}
}

// FromContext reads configuration from context parameters.
//
// Parameters read:
//   - optimizers.ParamLearningRate: initial learning rate
//   - ParamWarmupSteps: warmup steps (default: 0)
//   - ParamStepDecayFactor: decay factor (default: 0.1)
//
// Note: Boundaries cannot be read from context; they must be set explicitly.
func (c *StepConfig) FromContext() *StepConfig {
	c.readCommonFromContext()
	c.factor = context.GetParamOr(c.ctx, ParamStepDecayFactor, c.factor)
	return c
}

// InitialLearningRate sets the starting learning rate (at the end of warmup).
// If not set, reads from context parameter optimizers.ParamLearningRate.
func (c *StepConfig) InitialLearningRate(lr float64) *StepConfig {
	if lr < 0 {
		exceptions.Panicf("InitialLearningRate must be >= 0, got %g", lr)
	}
	c.initialLR = lr
	return c
}

// Boundaries sets the step numbers at which the learning rate is reduced.
// At each boundary, the learning rate is multiplied by Factor.
// Boundaries are automatically sorted.
//
// Example: Boundaries(10000, 20000, 30000) with Factor(0.1) means:
//   - reduce LR by 10x at step 10000
//   - reduce LR by 10x again at step 20000
//   - reduce LR by 10x again at step 30000
func (c *StepConfig) Boundaries(steps ...int) *StepConfig {
	for _, s := range steps {
		if s < 0 {
			exceptions.Panicf("Boundary step must be >= 0, got %d", s)
		}
	}
	c.boundaries = make([]int, len(steps))
	copy(c.boundaries, steps)
	sort.Ints(c.boundaries) // Ensure sorted order
	return c
}

// Factor sets the multiplicative factor applied at each boundary.
// The learning rate is multiplied by this value at each boundary.
//
// Common values:
//   - 0.1: Reduce by 10x at each boundary
//   - 0.5: Halve at each boundary
//   - 0.2: Reduce by 5x at each boundary
//
// Default is 0.1.
func (c *StepConfig) Factor(factor float64) *StepConfig {
	if factor <= 0 || factor > 1 {
		exceptions.Panicf("Factor must be in (0, 1], got %g", factor)
	}
	c.factor = factor
	return c
}

// WarmupSteps sets the number of warmup steps.
// During warmup, learning rate increases linearly from 0 to InitialLearningRate.
// Boundaries are adjusted to account for warmup (they refer to steps after warmup).
// Default is 0 (no warmup).
func (c *StepConfig) WarmupSteps(steps int) *StepConfig {
	if steps < 0 {
		exceptions.Panicf("WarmupSteps must be >= 0, got %d", steps)
	}
	c.warmupSteps = steps
	return c
}

// Done finalizes the configuration and builds the computation graph
// for the step learning rate schedule.
//
// If no boundaries are set or not in training mode, this is a no-op.
func (c *StepConfig) Done() {
	if !c.isEnabled() || len(c.boundaries) == 0 {
		return
	}

	// Validate configuration
	if c.initialLR < 0 {
		c.initialLR = context.GetParamOr(c.ctx, optimizers.ParamLearningRate, 0.0)
	}
	if c.initialLR <= 0 {
		exceptions.Panicf("step schedule: learning rate not configured and not set in context as %q",
			optimizers.ParamLearningRate)
		return
	}

	g := c.graph

	// Get current step (schedule maintains its own counter)
	step := c.getStepNode(StepScope)

	// Adjusted step for boundary checking (step - warmup)
	adjustedStep := c.getAdjustedStep(step)

	// Count how many boundaries have been passed
	// We use a series of comparisons and sum them up
	numBoundariesPassed := ScalarZero(g, c.dtype)
	for _, boundary := range c.boundaries {
		boundaryNode := Const(g, float64(boundary))
		boundaryNode = ConvertDType(boundaryNode, c.dtype)
		// Add 1 if adjustedStep >= boundary, else 0
		passed := Where(
			GreaterOrEqual(adjustedStep, boundaryNode),
			ScalarOne(g, c.dtype),
			ScalarZero(g, c.dtype),
		)
		numBoundariesPassed = Add(numBoundariesPassed, passed)
	}

	// lr = initial_lr * factor ^ num_boundaries_passed
	factorNode := ConvertDType(Const(g, c.factor), c.dtype)
	decayFactor := Pow(factorNode, numBoundariesPassed)
	lr := MulScalar(decayFactor, c.initialLR)

	// Apply warmup if configured
	lr = c.applyWarmup(lr, step, c.initialLR)

	// Update the learning rate variable
	c.updateLearningRate(lr)
}
