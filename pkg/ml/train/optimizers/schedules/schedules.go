// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package schedules provides learning rate scheduling strategies for training.
//
// Each schedule type follows a builder pattern similar to the cosineschedule package:
//
//  1. Create a configuration with New*(ctx, graph, dtype)
//  2. Configure parameters using method chaining
//  3. Call Done() to build the computation graph
//
// The schedules modify the learning rate variable during graph building, and only
// affect training (no effect during inference or evaluation).
//
// Example usage with linear decay and warmup:
//
//	func MyModelGraph(ctx *context.Context, inputs []*Node) *Node {
//		g := inputs[0].Graph()
//		schedules.Linear(ctx, g, dtypes.Float32).
//			InitialLearningRate(1e-4).
//			FinalLearningRate(0).
//			WarmupSteps(1000).
//			TotalSteps(10000).
//			Done()
//		// ... rest of model
//	}
//
// Or configure via context parameters:
//
//	ctx.SetParam(schedules.ParamTotalSteps, 10000)
//	ctx.SetParam(schedules.ParamWarmupSteps, 1000)
//	ctx.SetParam(optimizers.ParamLearningRate, 1e-4)
//
//	func MyModelGraph(ctx *context.Context, inputs []*Node) *Node {
//		g := inputs[0].Graph()
//		schedules.Linear(ctx, g, dtypes.Float32).FromContext().Done()
//		// ... rest of model
//	}
package schedules

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
)

// Common hyperparameter keys used by multiple schedules.
var (
	// ParamTotalSteps is the total number of training steps.
	// Used by linear, polynomial, and other schedules to calculate decay.
	ParamTotalSteps = "schedule_total_steps"

	// ParamWarmupSteps is the number of warmup steps where learning rate
	// linearly increases from 0 (or min) to the initial learning rate.
	// Default is 0 (no warmup).
	ParamWarmupSteps = "schedule_warmup_steps"

	// ParamFinalLearningRate is the learning rate at the end of training.
	// Default is 0.0 for most schedules.
	ParamFinalLearningRate = "schedule_final_learning_rate"
)

const (
	// Scope for schedule variables, nested under optimizers.Scope.
	Scope = "schedules"
)

// baseConfig contains common fields used by all schedule configurations.
type baseConfig struct {
	ctx   *context.Context
	graph *Graph
	dtype dtypes.DType

	initialLR   float64
	finalLR     float64
	warmupSteps int
	totalSteps  int
}

// newBaseConfig creates a new base configuration with defaults.
func newBaseConfig(ctx *context.Context, graph *Graph, dtype dtypes.DType) baseConfig {
	return baseConfig{
		ctx:       ctx,
		graph:     graph,
		dtype:     dtype,
		initialLR: -1, // -1 means not set, will be read from context
		finalLR:   0,
	}
}

// readCommonFromContext reads common parameters from the context.
func (b *baseConfig) readCommonFromContext() {
	if b.initialLR < 0 {
		b.initialLR = context.GetParamOr(b.ctx, optimizers.ParamLearningRate, 0.0)
	}
	b.finalLR = context.GetParamOr(b.ctx, ParamFinalLearningRate, b.finalLR)
	b.warmupSteps = context.GetParamOr(b.ctx, ParamWarmupSteps, b.warmupSteps)
	b.totalSteps = context.GetParamOr(b.ctx, ParamTotalSteps, b.totalSteps)
}

// isEnabled returns true if training is enabled and schedule should be applied.
func (b *baseConfig) isEnabled() bool {
	return b.ctx.IsTraining(b.graph)
}

// getStepNode returns the current step as a node, incremented from the schedule's own counter.
// This is similar to how cosineschedule maintains its own step counter.
func (b *baseConfig) getStepNode(scheduleScope string) *Node {
	step := optimizers.IncrementGlobalStepGraph(
		b.ctx.In(optimizers.Scope).In(Scope).In(scheduleScope),
		b.graph,
		b.dtype,
	)
	return MinusOne(step) // Return value before increment
}

// updateLearningRate sets the learning rate variable to the computed value.
func (b *baseConfig) updateLearningRate(lr *Node) {
	lrVar := optimizers.LearningRateVarWithValue(b.ctx, b.dtype, b.initialLR)
	lrVar.SetValueGraph(lr)
}

// applyWarmup applies linear warmup to the learning rate if warmup is configured.
// It returns the adjusted learning rate node.
//
// During warmup (step < warmupSteps):
//
//	lr = minLR + (maxLR - minLR) * (step / warmupSteps)
//
// After warmup: returns the provided lr unchanged.
func (b *baseConfig) applyWarmup(lr *Node, step *Node, maxLR float64) *Node {
	if b.warmupSteps <= 0 {
		return lr
	}

	g := lr.Graph()

	// Calculate warmup learning rate
	ratio := DivScalar(step, float64(b.warmupSteps))
	ratio = MinScalar(ratio, 1.0)
	warmupLR := AddScalar(MulScalar(ratio, maxLR-b.finalLR), b.finalLR)

	// Use warmup LR when step < warmupSteps
	warmupThreshold := Const(g, float64(b.warmupSteps))
	warmupThreshold = ConvertDType(warmupThreshold, b.dtype)
	return Where(LessThan(step, warmupThreshold), warmupLR, lr)
}

// getAdjustedStep returns the step adjusted for warmup (step - warmupSteps).
// Returns the original step if no warmup is configured.
func (b *baseConfig) getAdjustedStep(step *Node) *Node {
	if b.warmupSteps <= 0 {
		return step
	}
	return SubScalar(step, float64(b.warmupSteps))
}

// getAdjustedTotalSteps returns total steps minus warmup steps.
func (b *baseConfig) getAdjustedTotalSteps() int {
	if b.warmupSteps <= 0 {
		return b.totalSteps
	}
	return b.totalSteps - b.warmupSteps
}
