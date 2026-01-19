// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package schedules

import (
	"math"

	"github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
)

// One-cycle specific parameters.
var (
	// ParamOneCycleMaxLR is the maximum learning rate at the peak of the cycle.
	ParamOneCycleMaxLR = "schedule_onecycle_max_lr"

	// ParamOneCycleDivFactor divides max_lr to get initial_lr.
	// initial_lr = max_lr / div_factor
	ParamOneCycleDivFactor = "schedule_onecycle_div_factor"

	// ParamOneCycleFinalDivFactor divides initial_lr to get final_lr.
	// final_lr = initial_lr / final_div_factor
	ParamOneCycleFinalDivFactor = "schedule_onecycle_final_div_factor"

	// ParamOneCyclePctStart is the percentage of cycle spent in warmup phase.
	ParamOneCyclePctStart = "schedule_onecycle_pct_start"

	// ParamOneCycleAnnealStrategy is the annealing strategy: "cos" or "linear".
	ParamOneCycleAnnealStrategy = "schedule_onecycle_anneal_strategy"
)

const (
	// OneCycleScope is the scope name for one-cycle schedule variables.
	OneCycleScope = "onecycle"

	// AnnealCos uses cosine annealing.
	AnnealCos = "cos"

	// AnnealLinear uses linear annealing.
	AnnealLinear = "linear"
)

// OneCycleConfig configures the 1cycle learning rate policy.
//
// The 1cycle policy, introduced by Leslie Smith in "Super-Convergence: Very Fast Training of
// Neural Networks Using Large Learning Rates", consists of:
//
//  1. Warmup phase (pct_start of total steps): LR increases from initial_lr to max_lr
//  2. Annealing phase (remaining steps): LR decreases from max_lr to final_lr
//
// The annealing can be either cosine or linear.
//
// Default values (matching PyTorch):
//   - div_factor: 25.0 (initial_lr = max_lr / 25)
//   - final_div_factor: 10000.0 (final_lr = initial_lr / 10000)
//   - pct_start: 0.3 (30% warmup)
//   - anneal_strategy: "cos"
//
// Reference: https://arxiv.org/abs/1708.07120
type OneCycleConfig struct {
	baseConfig
	maxLR          float64
	divFactor      float64
	finalDivFactor float64
	pctStart       float64
	annealStrategy string
}

// OneCycle creates a one-cycle learning rate schedule configuration.
//
// The schedule only affects training; there is no effect during inference or evaluation.
//
// Example:
//
//	schedules.OneCycle(ctx, g, dtypes.Float32).
//		MaxLR(0.01).
//		DivFactor(25.0).
//		FinalDivFactor(10000.0).
//		PctStart(0.3).
//		TotalSteps(10000).
//		Done()
func OneCycle(ctx *context.Context, graph *Graph, dtype dtypes.DType) *OneCycleConfig {
	return &OneCycleConfig{
		baseConfig:     newBaseConfig(ctx, graph, dtype),
		maxLR:          -1, // Will be read from context or error
		divFactor:      25.0,
		finalDivFactor: 10000.0,
		pctStart:       0.3,
		annealStrategy: AnnealCos,
	}
}

// FromContext reads configuration from context parameters.
//
// Parameters read:
//   - ParamOneCycleMaxLR or optimizers.ParamLearningRate: maximum learning rate
//   - ParamTotalSteps: total training steps
//   - ParamOneCycleDivFactor: div factor (default: 25.0)
//   - ParamOneCycleFinalDivFactor: final div factor (default: 10000.0)
//   - ParamOneCyclePctStart: warmup percentage (default: 0.3)
//   - ParamOneCycleAnnealStrategy: "cos" or "linear" (default: "cos")
func (c *OneCycleConfig) FromContext() *OneCycleConfig {
	c.readCommonFromContext()
	// For one-cycle, try MaxLR param first, then fall back to ParamLearningRate
	c.maxLR = context.GetParamOr(c.ctx, ParamOneCycleMaxLR, c.maxLR)
	if c.maxLR < 0 {
		c.maxLR = context.GetParamOr(c.ctx, optimizers.ParamLearningRate, -1.0)
	}
	c.divFactor = context.GetParamOr(c.ctx, ParamOneCycleDivFactor, c.divFactor)
	c.finalDivFactor = context.GetParamOr(c.ctx, ParamOneCycleFinalDivFactor, c.finalDivFactor)
	c.pctStart = context.GetParamOr(c.ctx, ParamOneCyclePctStart, c.pctStart)
	c.annealStrategy = context.GetParamOr(c.ctx, ParamOneCycleAnnealStrategy, c.annealStrategy)
	return c
}

// MaxLR sets the maximum learning rate at the peak of the cycle.
// This is the highest learning rate reached during training.
// If not set, reads from context parameter ParamOneCycleMaxLR or optimizers.ParamLearningRate.
func (c *OneCycleConfig) MaxLR(lr float64) *OneCycleConfig {
	if lr <= 0 {
		exceptions.Panicf("MaxLR must be > 0, got %g", lr)
	}
	c.maxLR = lr
	return c
}

// DivFactor sets the division factor for initial learning rate.
//
//	initial_lr = max_lr / div_factor
//
// Default is 25.0 (initial LR is max_lr/25).
func (c *OneCycleConfig) DivFactor(factor float64) *OneCycleConfig {
	if factor <= 0 {
		exceptions.Panicf("DivFactor must be > 0, got %g", factor)
	}
	c.divFactor = factor
	return c
}

// FinalDivFactor sets the division factor for final learning rate.
//
//	final_lr = initial_lr / final_div_factor
//
// Default is 10000.0 (final LR is very small).
func (c *OneCycleConfig) FinalDivFactor(factor float64) *OneCycleConfig {
	if factor <= 0 {
		exceptions.Panicf("FinalDivFactor must be > 0, got %g", factor)
	}
	c.finalDivFactor = factor
	return c
}

// PctStart sets the percentage of the cycle spent in the warmup phase.
//
// During pct_start of total_steps, the LR increases from initial_lr to max_lr.
// After that, LR decreases from max_lr to final_lr.
//
// Default is 0.3 (30% warmup).
func (c *OneCycleConfig) PctStart(pct float64) *OneCycleConfig {
	if pct <= 0 || pct >= 1 {
		exceptions.Panicf("PctStart must be in (0, 1), got %g", pct)
	}
	c.pctStart = pct
	return c
}

// AnnealStrategy sets the annealing strategy for the decay phase.
//
//   - "cos": Cosine annealing (smooth, commonly used)
//   - "linear": Linear annealing
//
// Default is "cos".
func (c *OneCycleConfig) AnnealStrategy(strategy string) *OneCycleConfig {
	if strategy != AnnealCos && strategy != AnnealLinear {
		exceptions.Panicf("AnnealStrategy must be %q or %q, got %q", AnnealCos, AnnealLinear, strategy)
	}
	c.annealStrategy = strategy
	return c
}

// TotalSteps sets the total number of training steps.
func (c *OneCycleConfig) TotalSteps(steps int) *OneCycleConfig {
	if steps <= 0 {
		exceptions.Panicf("TotalSteps must be > 0, got %d", steps)
	}
	c.totalSteps = steps
	return c
}

// Done finalizes the configuration and builds the computation graph
// for the one-cycle learning rate schedule.
//
// If TotalSteps is 0 or not in training mode, this is a no-op.
func (c *OneCycleConfig) Done() {
	if !c.isEnabled() || c.totalSteps <= 0 {
		return
	}

	// Validate configuration
	if c.maxLR < 0 {
		c.maxLR = context.GetParamOr(c.ctx, ParamOneCycleMaxLR, -1.0)
		if c.maxLR < 0 {
			c.maxLR = context.GetParamOr(c.ctx, optimizers.ParamLearningRate, 0.0)
		}
	}
	if c.maxLR <= 0 {
		exceptions.Panicf("onecycle schedule: max learning rate not configured and not set in context as %q or %q",
			ParamOneCycleMaxLR, optimizers.ParamLearningRate)
		return
	}

	g := c.graph

	// Calculate initial and final learning rates
	initialLR := c.maxLR / c.divFactor
	finalLR := initialLR / c.finalDivFactor

	// Calculate phase boundaries
	warmupSteps := int(float64(c.totalSteps) * c.pctStart)
	annealSteps := c.totalSteps - warmupSteps

	// Get current step (schedule maintains its own counter)
	step := c.getStepNode(OneCycleScope)

	// Calculate which phase we're in and the progress within that phase
	warmupStepsNode := Const(g, float64(warmupSteps))
	warmupStepsNode = ConvertDType(warmupStepsNode, c.dtype)

	// Phase 1: Warmup (0 to warmupSteps)
	// Progress in warmup: step / warmupSteps
	warmupProgress := Div(step, warmupStepsNode)
	warmupProgress = MinScalar(warmupProgress, 1.0)

	// Warmup LR: initial_lr + (max_lr - initial_lr) * progress
	// NOTE: This implementation applies the annealing strategy to the warmup phase.
	// This differs from PyTorch's OneCycleLR, which always uses linear warmup
	// regardless of the anneal_strategy parameter. Here, when anneal_strategy="cos",
	// warmup also uses cosine annealing for a smoother acceleration.
	var warmupLR *Node
	if c.annealStrategy == AnnealCos {
		// Cosine warmup: smoother start
		cosWarmup := Cos(MulScalar(warmupProgress, math.Pi))
		cosWarmup = DivScalar(OneMinus(cosWarmup), 2.0) // (1 - cos) / 2 -> 0 to 1
		warmupLR = AddScalar(MulScalar(cosWarmup, c.maxLR-initialLR), initialLR)
	} else {
		// Linear warmup
		warmupLR = AddScalar(MulScalar(warmupProgress, c.maxLR-initialLR), initialLR)
	}

	// Phase 2: Annealing (warmupSteps to totalSteps)
	// Progress in annealing: (step - warmupSteps) / annealSteps
	annealStepsNode := ConvertDType(Const(g, float64(annealSteps)), c.dtype)
	annealProgress := Div(Sub(step, warmupStepsNode), annealStepsNode)
	annealProgress = MaxScalar(annealProgress, 0.0)
	annealProgress = MinScalar(annealProgress, 1.0)

	// Annealing LR: max_lr -> final_lr
	var annealLR *Node
	if c.annealStrategy == AnnealCos {
		// Cosine annealing
		cosAnneal := Cos(MulScalar(annealProgress, math.Pi))
		cosAnneal = DivScalar(OnePlus(cosAnneal), 2.0) // (1 + cos) / 2 -> 1 to 0
		annealLR = AddScalar(MulScalar(cosAnneal, c.maxLR-finalLR), finalLR)
	} else {
		// Linear annealing
		annealLR = AddScalar(MulScalar(annealProgress, finalLR-c.maxLR), c.maxLR)
	}

	// Choose LR based on which phase we're in
	inWarmup := LessThan(step, warmupStepsNode)
	lr := Where(inWarmup, warmupLR, annealLR)

	// Update the learning rate variable
	// For one-cycle, we set the initial value to the calculated initial LR
	c.initialLR = initialLR
	c.updateLearningRate(lr)
}
