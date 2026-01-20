// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package adalora_test

import (
	"strings"
	"testing"

	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/adapters"
	"github.com/gomlx/gomlx/pkg/ml/adapters/adalora"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAdaLoRAConfig(t *testing.T) {
	config := adalora.NewConfig()

	// Test defaults
	assert.Equal(t, 12, config.InitialRank())
	assert.Equal(t, 8, config.TargetRank())
	assert.Equal(t, 16.0, config.Alpha())
	assert.Equal(t, 0.0, config.Dropout())
	assert.True(t, config.Enabled())
	assert.Equal(t, adapters.AdapterTypeAdaLoRA, config.Type())

	// Test AdaLoRA-specific defaults
	assert.Equal(t, 0.85, config.Beta)
	assert.Equal(t, 100, config.WarmupSteps)
	assert.Equal(t, 50, config.BudgetingInterval)
	assert.Equal(t, 1000, config.FinalBudgetingStep)
	assert.Equal(t, 0.1, config.OrthoRegStrength)
	assert.Equal(t, 1, config.MinRank)
}

func TestAdaLoRAConfigSetters(t *testing.T) {
	config := adalora.NewConfig().
		SetInitialRank(16).
		SetTargetRank(10).
		SetAlpha(32.0).
		SetDropout(0.1).
		SetBeta(0.9).
		SetWarmupSteps(200).
		SetBudgetingInterval(100).
		SetFinalBudgetingStep(2000).
		SetOrthoRegStrength(0.2).
		SetMinRank(2)

	assert.Equal(t, 16, config.InitialRank())
	assert.Equal(t, 10, config.TargetRank())
	assert.Equal(t, 32.0, config.Alpha())
	assert.Equal(t, 0.1, config.Dropout())
	assert.Equal(t, 0.9, config.Beta)
	assert.Equal(t, 200, config.WarmupSteps)
	assert.Equal(t, 100, config.BudgetingInterval)
	assert.Equal(t, 2000, config.FinalBudgetingStep)
	assert.Equal(t, 0.2, config.OrthoRegStrength)
	assert.Equal(t, 2, config.MinRank)

	// Test scale calculation: alpha / initialRank
	assert.Equal(t, 2.0, config.Scale()) // 32.0 / 16 = 2.0
}

func TestAdaLoRAConfigValidation(t *testing.T) {
	config := adalora.NewConfig()

	// Valid config should pass
	err := config.Validate()
	require.NoError(t, err)

	// Test invalid configurations
	tests := []struct {
		name    string
		setup   func(*adalora.Config)
		wantErr string
	}{
		{
			name: "target > initial",
			setup: func(c *adalora.Config) {
				c.SetInitialRank(8)
				c.SetTargetRank(16)
			},
			wantErr: "target rank (16) cannot exceed initial rank (8)",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := adalora.NewConfig()
			tt.setup(cfg)
			err := cfg.Validate()
			if tt.wantErr != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.wantErr)
			}
		})
	}

	// Test panic conditions
	assert.Panics(t, func() {
		adalora.NewConfig().SetInitialRank(0)
	})
	assert.Panics(t, func() {
		adalora.NewConfig().SetTargetRank(0)
	})
	assert.Panics(t, func() {
		adalora.NewConfig().SetAlpha(-1)
	})
	assert.Panics(t, func() {
		adalora.NewConfig().SetDropout(-0.1)
	})
	assert.Panics(t, func() {
		adalora.NewConfig().SetBeta(1.5)
	})
	assert.Panics(t, func() {
		adalora.NewConfig().SetOrthoRegStrength(-0.1)
	})
}

func TestAdaLoRAConfigClone(t *testing.T) {
	config := adalora.NewConfig().
		SetInitialRank(16).
		SetTargetRank(10).
		SetAlpha(32.0).
		SetBeta(0.9)

	clone := config.Clone()
	adaLoRAClone, ok := clone.(*adalora.Config)
	require.True(t, ok)

	// Check values are copied
	assert.Equal(t, 16, adaLoRAClone.InitialRank())
	assert.Equal(t, 10, adaLoRAClone.TargetRank())
	assert.Equal(t, 32.0, adaLoRAClone.Alpha())
	assert.Equal(t, 0.9, adaLoRAClone.Beta)

	// Verify independence
	config.SetInitialRank(8)
	assert.Equal(t, 16, adaLoRAClone.InitialRank())
}

func TestAdaLoRAAdapterCreation(t *testing.T) {
	config := adalora.NewConfig().SetInitialRank(8)

	adapter, err := adalora.New("test_adalora", config)
	require.NoError(t, err)
	require.NotNil(t, adapter)

	assert.Equal(t, "test_adalora", adapter.Name())
	assert.Equal(t, adapters.AdapterTypeAdaLoRA, adapter.Type())
	assert.False(t, adapter.IsMerged())

	// Test with nil config (should use defaults)
	adapter2, err := adalora.New("default_adalora", nil)
	require.NoError(t, err)
	require.NotNil(t, adapter2)
}

func TestAdaLoRAFactory(t *testing.T) {
	factory := adalora.NewFactory()

	assert.Equal(t, adapters.AdapterTypeAdaLoRA, factory.Type())

	config := adalora.NewConfig()
	adapter, err := factory.Create("test", config)
	require.NoError(t, err)
	assert.Equal(t, "test", adapter.Name())
}

func TestAdaLoRADense(t *testing.T) {
	backend, err := backends.New()
	require.NoError(t, err)
	ctx := context.New()

	config := adalora.NewConfig().SetInitialRank(4).SetTargetRank(2).SetAlpha(8.0)

	batchSize := 2
	inputDim := 8
	outputDim := 4

	exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, input *graph.Node) *graph.Node {
		return adalora.Dense(ctx, input, config, outputDim)
	})

	// Create test input
	inputData := make([][]float32, batchSize)
	for i := range inputData {
		inputData[i] = make([]float32, inputDim)
		for j := range inputData[i] {
			inputData[i][j] = 1.0
		}
	}

	result := exec.MustExec(inputData)[0]

	// Verify output shape
	assert.Equal(t, []int{batchSize, outputDim}, result.Shape().Dimensions)
}

func TestAdaLoRADenseWithBias(t *testing.T) {
	backend, err := backends.New()
	require.NoError(t, err)
	ctx := context.New()

	config := adalora.NewConfig().SetInitialRank(4)

	batchSize := 2
	inputDim := 8
	outputDim := 4

	exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, input *graph.Node) *graph.Node {
		return adalora.DenseWithBias(ctx, input, config, true, outputDim)
	})

	inputData := make([][]float32, batchSize)
	for i := range inputData {
		inputData[i] = make([]float32, inputDim)
		for j := range inputData[i] {
			inputData[i][j] = 1.0
		}
	}

	result := exec.MustExec(inputData)[0]

	assert.Equal(t, []int{batchSize, outputDim}, result.Shape().Dimensions)
}

func TestAdaLoRADisabled(t *testing.T) {
	backend, err := backends.New()
	require.NoError(t, err)
	ctx := context.New()

	config := adalora.NewConfig().SetInitialRank(4)
	config.SetEnabled(false)

	batchSize := 2
	inputDim := 8
	outputDim := 4

	exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, input *graph.Node) *graph.Node {
		return adalora.Dense(ctx, input, config, outputDim)
	})

	inputData := make([][]float32, batchSize)
	for i := range inputData {
		inputData[i] = make([]float32, inputDim)
		for j := range inputData[i] {
			inputData[i][j] = 1.0
		}
	}

	result := exec.MustExec(inputData)[0]

	// Should still work, just without AdaLoRA contribution
	assert.Equal(t, []int{batchSize, outputDim}, result.Shape().Dimensions)
}

func TestAdaLoRAVariables(t *testing.T) {
	backend, err := backends.New()
	require.NoError(t, err)
	ctx := context.New()

	rank := 4
	config := adalora.NewConfig().SetInitialRank(rank)

	inputDim := 8
	outputDim := 4

	// Run once to create variables
	exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, input *graph.Node) *graph.Node {
		return adalora.Dense(ctx, input, config, outputDim)
	})
	inputData := make([][]float32, 1)
	inputData[0] = make([]float32, inputDim)
	for j := range inputData[0] {
		inputData[0][j] = 1.0
	}
	_ = exec.MustExec(inputData)

	// Check that AdaLoRA variables were created
	var foundP, foundLambda, foundQ, foundMask, foundImportance, foundWeights bool
	for v := range ctx.IterVariables() {
		name := v.ParameterName()
		if strings.Contains(name, "adalora_P") {
			foundP = true
			// P should be [inputDim, rank]
			assert.Equal(t, []int{inputDim, rank}, v.Shape().Dimensions)
		}
		if strings.Contains(name, "adalora_lambda") {
			foundLambda = true
			// Lambda should be [rank]
			assert.Equal(t, []int{rank}, v.Shape().Dimensions)
		}
		if strings.Contains(name, "adalora_Q") {
			foundQ = true
			// Q should be [rank, outputDim]
			assert.Equal(t, []int{rank, outputDim}, v.Shape().Dimensions)
		}
		if strings.Contains(name, "adalora_mask") {
			foundMask = true
			// Mask should be [rank]
			assert.Equal(t, []int{rank}, v.Shape().Dimensions)
		}
		if strings.Contains(name, "adalora_importance") {
			foundImportance = true
			// Importance should be [rank]
			assert.Equal(t, []int{rank}, v.Shape().Dimensions)
		}
		// Base weights
		if strings.Contains(name, "weights") &&
			!strings.Contains(name, "adalora_P") &&
			!strings.Contains(name, "adalora_lambda") &&
			!strings.Contains(name, "adalora_Q") &&
			!strings.Contains(name, "adalora_mask") &&
			!strings.Contains(name, "adalora_importance") {
			foundWeights = true
			// Base weights should be [inputDim, outputDim]
			assert.Equal(t, []int{inputDim, outputDim}, v.Shape().Dimensions)
		}
	}

	assert.True(t, foundP, "adalora_P variable should be created")
	assert.True(t, foundLambda, "adalora_lambda variable should be created")
	assert.True(t, foundQ, "adalora_Q variable should be created")
	assert.True(t, foundMask, "adalora_mask variable should be created")
	assert.True(t, foundImportance, "adalora_importance variable should be created")
	assert.True(t, foundWeights, "base weights variable should be created")
}

func TestAdaLoRARegistration(t *testing.T) {
	// The factory should be auto-registered via init()
	factory, ok := adapters.GetFactory(adapters.AdapterTypeAdaLoRA)
	assert.True(t, ok)
	assert.NotNil(t, factory)
	assert.Equal(t, adapters.AdapterTypeAdaLoRA, factory.Type())
}

func TestBudgetState(t *testing.T) {
	backend, err := backends.New()
	require.NoError(t, err)
	ctx := context.New()

	rank := 4
	config := adalora.NewConfig().SetInitialRank(rank).SetTargetRank(2)

	inputDim := 8
	outputDim := 4

	// Create two layers to test budget allocation
	exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, input *graph.Node) *graph.Node {
		layer1 := adalora.Dense(ctx.In("layer1"), input, config, outputDim)
		layer2 := adalora.Dense(ctx.In("layer2"), layer1, config, outputDim)
		return layer2
	})

	inputData := make([][]float32, 1)
	inputData[0] = make([]float32, inputDim)
	for j := range inputData[0] {
		inputData[0][j] = 1.0
	}
	_ = exec.MustExec(inputData)

	// Create budget state
	state := adalora.NewBudgetState(ctx)

	// Should have found 2 layers
	assert.Equal(t, 2, len(state.Layers))

	// Total initial budget should be 2 * rank = 8
	assert.Equal(t, 2*rank, state.TotalActiveSingularValues)

	// Each layer should have full rank initially
	for _, layer := range state.Layers {
		assert.Equal(t, rank, layer.ActiveRank)
	}
}

func TestBudgetReallocation(t *testing.T) {
	backend, err := backends.New()
	require.NoError(t, err)
	ctx := context.New()

	rank := 4
	config := adalora.NewConfig().SetInitialRank(rank).SetTargetRank(2).SetMinRank(1)

	inputDim := 8
	outputDim := 4

	// Create a single layer
	exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, input *graph.Node) *graph.Node {
		return adalora.Dense(ctx, input, config, outputDim)
	})

	inputData := make([][]float32, 1)
	inputData[0] = make([]float32, inputDim)
	for j := range inputData[0] {
		inputData[0][j] = 1.0
	}
	_ = exec.MustExec(inputData)

	// Create budget state and set some importance scores
	state := adalora.NewBudgetState(ctx)
	require.Equal(t, 1, len(state.Layers))

	// Set varying importance (some high, some low)
	state.Layers[0].Importance = []float32{1.0, 0.1, 0.5, 0.2}

	// Reallocate to target budget of 2
	state.ReallocateBudget(ctx, 2, 1)

	// Should have pruned 2 lowest importance values (indices 1 and 3)
	assert.Equal(t, 2, state.Layers[0].ActiveRank)
	assert.Equal(t, 2, state.TotalActiveSingularValues)

	// Check mask
	assert.Equal(t, float32(1.0), state.Layers[0].Mask[0]) // highest importance, kept
	assert.Equal(t, float32(0.0), state.Layers[0].Mask[1]) // lowest importance, pruned
	assert.Equal(t, float32(1.0), state.Layers[0].Mask[2]) // medium importance, kept
	assert.Equal(t, float32(0.0), state.Layers[0].Mask[3]) // low importance, pruned
}

func TestImportanceStats(t *testing.T) {
	state := &adalora.BudgetState{
		Layers: []adalora.LayerImportance{
			{
				Importance: []float32{1.0, 2.0, 3.0, 4.0},
				Mask:       []float32{1.0, 1.0, 1.0, 1.0},
				ActiveRank: 4,
			},
		},
	}

	stats := state.GetImportanceStats()

	assert.Equal(t, float32(1.0), stats.Min)
	assert.Equal(t, float32(4.0), stats.Max)
	assert.Equal(t, float32(2.5), stats.Mean) // (1+2+3+4)/4 = 2.5
	assert.Equal(t, float32(3.0), stats.Median)
}
