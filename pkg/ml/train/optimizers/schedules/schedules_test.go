// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package schedules_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers/schedules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	_ "github.com/gomlx/gomlx/backends/default"
)

func TestLinearSchedule(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	const totalSteps = 100
	const initialLR = 1.0
	const finalLR = 0.0

	t.Run("basic", func(t *testing.T) {
		ctx := context.New().Checked(false)
		exec, err := context.NewExec(backend, ctx, func(ctx *context.Context, graph *Graph) *Node {
			ctx.SetTraining(graph, true)
			schedules.Linear(ctx, graph, dtypes.Float32).
				InitialLearningRate(initialLR).
				FinalLearningRate(finalLR).
				TotalSteps(totalSteps).
				Done()
			return optimizers.LearningRateVar(ctx, dtypes.Float32, 1e3).ValueGraph(graph)
		})
		require.NoError(t, err)

		for ii := range totalSteps {
			lrT, err := exec.Exec1()
			require.NoErrorf(t, err, "failed for step %d", ii)

			// Check step counter
			stepVar := ctx.GetVariableByScopeAndName(
				fmt.Sprintf("/%s/%s/%s", optimizers.Scope, schedules.Scope, schedules.LinearScope),
				optimizers.GlobalStepVariableName,
			)
			require.NotNil(t, stepVar, "step variable should exist")
			step := stepVar.MustValue().Value().(int64)
			assert.Equal(t, int64(ii+1), step)

			// Check linear decay: lr = initial + (final - initial) * progress
			lr := tensors.ToScalar[float32](lrT)
			progress := float64(ii) / float64(totalSteps)
			wantLR := initialLR + (finalLR-initialLR)*progress
			require.InDeltaf(t, float32(wantLR), lr, 1e-4, "step=%d, want=%g, got=%g", ii, wantLR, lr)
		}
	})

	t.Run("with warmup", func(t *testing.T) {
		ctx := context.New().Checked(false)
		const warmupSteps = 20
		exec, err := context.NewExec(backend, ctx, func(ctx *context.Context, graph *Graph) *Node {
			ctx.SetTraining(graph, true)
			schedules.Linear(ctx, graph, dtypes.Float32).
				InitialLearningRate(initialLR).
				FinalLearningRate(finalLR).
				WarmupSteps(warmupSteps).
				TotalSteps(totalSteps).
				Done()
			return optimizers.LearningRateVar(ctx, dtypes.Float32, 1e3).ValueGraph(graph)
		})
		require.NoError(t, err)

		for ii := range totalSteps {
			lrT, err := exec.Exec1()
			require.NoErrorf(t, err, "failed for step %d", ii)

			lr := tensors.ToScalar[float32](lrT)
			var wantLR float64
			if ii < warmupSteps {
				// Warmup: linear increase from finalLR to initialLR
				ratio := float64(ii) / float64(warmupSteps)
				wantLR = finalLR + (initialLR-finalLR)*ratio
			} else {
				// Decay: linear decrease from initialLR to finalLR
				progress := float64(ii-warmupSteps) / float64(totalSteps-warmupSteps)
				wantLR = initialLR + (finalLR-initialLR)*progress
			}
			require.InDeltaf(t, float32(wantLR), lr, 1e-4, "step=%d, want=%g, got=%g", ii, wantLR, lr)
		}
	})

	t.Run("from context", func(t *testing.T) {
		ctx := context.New().Checked(false)
		ctx.SetParam(optimizers.ParamLearningRate, initialLR)
		ctx.SetParam(schedules.ParamFinalLearningRate, finalLR)
		ctx.SetParam(schedules.ParamTotalSteps, totalSteps)
		ctx.SetParam(schedules.ParamWarmupSteps, 10)

		exec, err := context.NewExec(backend, ctx, func(ctx *context.Context, graph *Graph) *Node {
			ctx.SetTraining(graph, true)
			schedules.Linear(ctx, graph, dtypes.Float32).FromContext().Done()
			return optimizers.LearningRateVar(ctx, dtypes.Float32, 1e3).ValueGraph(graph)
		})
		require.NoError(t, err)

		// Just verify it runs without error
		for ii := range totalSteps {
			_, err := exec.Exec1()
			require.NoErrorf(t, err, "failed for step %d", ii)
		}
	})
}

func TestPolynomialSchedule(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	const totalSteps = 100
	const initialLR = 1.0
	const finalLR = 0.0

	t.Run("linear (power=1)", func(t *testing.T) {
		ctx := context.New().Checked(false)
		exec, err := context.NewExec(backend, ctx, func(ctx *context.Context, graph *Graph) *Node {
			ctx.SetTraining(graph, true)
			schedules.Polynomial(ctx, graph, dtypes.Float32).
				InitialLearningRate(initialLR).
				FinalLearningRate(finalLR).
				Power(1.0).
				TotalSteps(totalSteps).
				Done()
			return optimizers.LearningRateVar(ctx, dtypes.Float32, 1e3).ValueGraph(graph)
		})
		require.NoError(t, err)

		for ii := range totalSteps {
			lrT, err := exec.Exec1()
			require.NoErrorf(t, err, "failed for step %d", ii)

			lr := tensors.ToScalar[float32](lrT)
			// With power=1, polynomial decay = linear decay
			progress := float64(ii) / float64(totalSteps)
			decay := 1.0 - progress
			wantLR := (initialLR-finalLR)*decay + finalLR
			require.InDeltaf(t, float32(wantLR), lr, 1e-4, "step=%d, want=%g, got=%g", ii, wantLR, lr)
		}
	})

	t.Run("quadratic (power=2)", func(t *testing.T) {
		ctx := context.New().Checked(false)
		const power = 2.0
		exec, err := context.NewExec(backend, ctx, func(ctx *context.Context, graph *Graph) *Node {
			ctx.SetTraining(graph, true)
			schedules.Polynomial(ctx, graph, dtypes.Float32).
				InitialLearningRate(initialLR).
				FinalLearningRate(finalLR).
				Power(power).
				TotalSteps(totalSteps).
				Done()
			return optimizers.LearningRateVar(ctx, dtypes.Float32, 1e3).ValueGraph(graph)
		})
		require.NoError(t, err)

		for ii := range totalSteps {
			lrT, err := exec.Exec1()
			require.NoErrorf(t, err, "failed for step %d", ii)

			lr := tensors.ToScalar[float32](lrT)
			progress := float64(ii) / float64(totalSteps)
			decay := math.Pow(1.0-progress, power)
			wantLR := (initialLR-finalLR)*decay + finalLR
			require.InDeltaf(t, float32(wantLR), lr, 1e-4, "step=%d, want=%g, got=%g", ii, wantLR, lr)
		}
	})
}

func TestExponentialSchedule(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	const initialLR = 1.0
	const decayRate = 0.9
	const decaySteps = 10

	t.Run("continuous", func(t *testing.T) {
		ctx := context.New().Checked(false)
		exec, err := context.NewExec(backend, ctx, func(ctx *context.Context, graph *Graph) *Node {
			ctx.SetTraining(graph, true)
			schedules.Exponential(ctx, graph, dtypes.Float32).
				InitialLearningRate(initialLR).
				DecayRate(decayRate).
				DecaySteps(decaySteps).
				Staircase(false).
				Done()
			return optimizers.LearningRateVar(ctx, dtypes.Float32, 1e3).ValueGraph(graph)
		})
		require.NoError(t, err)

		for ii := range 50 {
			lrT, err := exec.Exec1()
			require.NoErrorf(t, err, "failed for step %d", ii)

			lr := tensors.ToScalar[float32](lrT)
			// Continuous decay: lr = initial * decay_rate ^ (step / decay_steps)
			exponent := float64(ii) / float64(decaySteps)
			wantLR := initialLR * math.Pow(decayRate, exponent)
			require.InDeltaf(t, float32(wantLR), lr, 1e-4, "step=%d, want=%g, got=%g", ii, wantLR, lr)
		}
	})

	t.Run("staircase", func(t *testing.T) {
		ctx := context.New().Checked(false)
		exec, err := context.NewExec(backend, ctx, func(ctx *context.Context, graph *Graph) *Node {
			ctx.SetTraining(graph, true)
			schedules.Exponential(ctx, graph, dtypes.Float32).
				InitialLearningRate(initialLR).
				DecayRate(decayRate).
				DecaySteps(decaySteps).
				Staircase(true).
				Done()
			return optimizers.LearningRateVar(ctx, dtypes.Float32, 1e3).ValueGraph(graph)
		})
		require.NoError(t, err)

		for ii := range 50 {
			lrT, err := exec.Exec1()
			require.NoErrorf(t, err, "failed for step %d", ii)

			lr := tensors.ToScalar[float32](lrT)
			// Staircase decay: lr = initial * decay_rate ^ floor(step / decay_steps)
			exponent := math.Floor(float64(ii) / float64(decaySteps))
			wantLR := initialLR * math.Pow(decayRate, exponent)
			require.InDeltaf(t, float32(wantLR), lr, 1e-4, "step=%d, want=%g, got=%g", ii, wantLR, lr)
		}
	})
}

func TestStepSchedule(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	const initialLR = 1.0
	const factor = 0.1

	t.Run("multiple boundaries", func(t *testing.T) {
		ctx := context.New().Checked(false)
		boundaries := []int{10, 20, 30}
		exec, err := context.NewExec(backend, ctx, func(ctx *context.Context, graph *Graph) *Node {
			ctx.SetTraining(graph, true)
			schedules.Step(ctx, graph, dtypes.Float32).
				InitialLearningRate(initialLR).
				Boundaries(boundaries...).
				Factor(factor).
				Done()
			return optimizers.LearningRateVar(ctx, dtypes.Float32, 1e3).ValueGraph(graph)
		})
		require.NoError(t, err)

		for ii := range 50 {
			lrT, err := exec.Exec1()
			require.NoErrorf(t, err, "failed for step %d", ii)

			lr := tensors.ToScalar[float32](lrT)

			// Count boundaries passed
			numPassed := 0
			for _, b := range boundaries {
				if ii >= b {
					numPassed++
				}
			}
			wantLR := initialLR * math.Pow(factor, float64(numPassed))
			require.InDeltaf(t, float32(wantLR), lr, 1e-4, "step=%d, numPassed=%d, want=%g, got=%g", ii, numPassed, wantLR, lr)
		}
	})
}

func TestOneCycleSchedule(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	const totalSteps = 100
	const maxLR = 1.0
	const divFactor = 25.0
	const finalDivFactor = 10000.0
	const pctStart = 0.3

	initialLR := maxLR / divFactor
	finalLR := initialLR / finalDivFactor
	warmupSteps := int(float64(totalSteps) * pctStart)

	t.Run("linear anneal", func(t *testing.T) {
		ctx := context.New().Checked(false)
		exec, err := context.NewExec(backend, ctx, func(ctx *context.Context, graph *Graph) *Node {
			ctx.SetTraining(graph, true)
			schedules.OneCycle(ctx, graph, dtypes.Float32).
				MaxLR(maxLR).
				DivFactor(divFactor).
				FinalDivFactor(finalDivFactor).
				PctStart(pctStart).
				AnnealStrategy(schedules.AnnealLinear).
				TotalSteps(totalSteps).
				Done()
			return optimizers.LearningRateVar(ctx, dtypes.Float32, 1e3).ValueGraph(graph)
		})
		require.NoError(t, err)

		for ii := range totalSteps {
			lrT, err := exec.Exec1()
			require.NoErrorf(t, err, "failed for step %d", ii)

			lr := tensors.ToScalar[float32](lrT)
			var wantLR float64

			if ii < warmupSteps {
				// Warmup phase: linear increase from initialLR to maxLR
				progress := float64(ii) / float64(warmupSteps)
				wantLR = initialLR + (maxLR-initialLR)*progress
			} else {
				// Annealing phase: linear decrease from maxLR to finalLR
				annealSteps := totalSteps - warmupSteps
				progress := float64(ii-warmupSteps) / float64(annealSteps)
				wantLR = maxLR + (finalLR-maxLR)*progress
			}
			require.InDeltaf(t, float32(wantLR), lr, 1e-3, "step=%d, want=%g, got=%g", ii, wantLR, lr)
		}
	})

	t.Run("cosine anneal", func(t *testing.T) {
		ctx := context.New().Checked(false)
		exec, err := context.NewExec(backend, ctx, func(ctx *context.Context, graph *Graph) *Node {
			ctx.SetTraining(graph, true)
			schedules.OneCycle(ctx, graph, dtypes.Float32).
				MaxLR(maxLR).
				DivFactor(divFactor).
				FinalDivFactor(finalDivFactor).
				PctStart(pctStart).
				AnnealStrategy(schedules.AnnealCos).
				TotalSteps(totalSteps).
				Done()
			return optimizers.LearningRateVar(ctx, dtypes.Float32, 1e3).ValueGraph(graph)
		})
		require.NoError(t, err)

		// Just verify it runs and produces values in expected range
		for ii := range totalSteps {
			lrT, err := exec.Exec1()
			require.NoErrorf(t, err, "failed for step %d", ii)

			lr := tensors.ToScalar[float32](lrT)
			// LR should be between finalLR and maxLR
			require.GreaterOrEqual(t, lr, float32(finalLR)-1e-6, "step=%d, lr=%g too low", ii, lr)
			require.LessOrEqual(t, lr, float32(maxLR)+1e-6, "step=%d, lr=%g too high", ii, lr)
		}
	})
}

func TestScheduleNoOpWhenNotTraining(t *testing.T) {
	backend := graphtest.BuildTestBackend()

	ctx := context.New().Checked(false)
	// Note: NOT setting training mode
	exec, err := context.NewExec(backend, ctx, func(ctx *context.Context, graph *Graph) *Node {
		// ctx.SetTraining(graph, true) -- deliberately not set
		schedules.Linear(ctx, graph, dtypes.Float32).
			InitialLearningRate(1.0).
			TotalSteps(100).
			Done()

		// Return a constant to verify exec works
		return Const(graph, float32(42.0))
	})
	require.NoError(t, err)

	result, err := exec.Exec1()
	require.NoError(t, err)
	assert.Equal(t, float32(42.0), tensors.ToScalar[float32](result))

	// Verify no schedule variables were created (schedule was a no-op)
	stepVar := ctx.GetVariableByScopeAndName(
		fmt.Sprintf("/%s/%s/%s", optimizers.Scope, schedules.Scope, schedules.LinearScope),
		optimizers.GlobalStepVariableName,
	)
	assert.Nil(t, stepVar, "no step variable should be created when not training")
}
