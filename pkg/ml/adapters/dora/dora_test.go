// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package dora_test

import (
	"strings"
	"testing"

	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/adapters"
	"github.com/gomlx/gomlx/pkg/ml/adapters/dora"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDoRAConfig(t *testing.T) {
	config := dora.NewConfig()

	// Test defaults from LoRA
	assert.Equal(t, 8, config.Rank())
	assert.Equal(t, 16.0, config.Alpha())
	assert.Equal(t, 0.0, config.Dropout())
	assert.True(t, config.Enabled())
	assert.Equal(t, adapters.AdapterTypeDoRA, config.Type())

	// Test DoRA-specific defaults
	assert.True(t, config.TrainMagnitude)
	assert.True(t, config.InitMagnitudeFromPretrained)
	assert.Equal(t, 0.0, config.MagnitudeRegularization)
}

func TestDoRAConfigSetters(t *testing.T) {
	config := dora.NewConfig().
		SetRank(16).
		SetAlpha(32.0).
		SetDropout(0.1).
		SetTrainMagnitude(false).
		SetInitMagnitudeFromPretrained(false).
		SetMagnitudeRegularization(0.01)

	assert.Equal(t, 16, config.Rank())
	assert.Equal(t, 32.0, config.Alpha())
	assert.Equal(t, 0.1, config.Dropout())
	assert.False(t, config.TrainMagnitude)
	assert.False(t, config.InitMagnitudeFromPretrained)
	assert.Equal(t, 0.01, config.MagnitudeRegularization)

	// Test scale calculation
	assert.Equal(t, 2.0, config.Scale()) // 32.0 / 16 = 2.0
}

func TestDoRAConfigValidation(t *testing.T) {
	config := dora.NewConfig()

	// Valid config should pass
	err := config.Validate()
	require.NoError(t, err)

	// Invalid rank should panic
	assert.Panics(t, func() {
		config.SetRank(0)
	})

	// Invalid magnitude regularization should panic
	assert.Panics(t, func() {
		config.SetMagnitudeRegularization(-0.1)
	})
}

func TestDoRAConfigClone(t *testing.T) {
	config := dora.NewConfig().
		SetRank(16).
		SetAlpha(32.0).
		SetTrainMagnitude(false).
		SetMagnitudeRegularization(0.05)

	clone := config.Clone()
	doraClone, ok := clone.(*dora.Config)
	require.True(t, ok)

	// Check values are copied
	assert.Equal(t, 16, doraClone.Rank())
	assert.Equal(t, 32.0, doraClone.Alpha())
	assert.False(t, doraClone.TrainMagnitude)
	assert.Equal(t, 0.05, doraClone.MagnitudeRegularization)

	// Verify independence
	config.SetRank(8)
	assert.Equal(t, 16, doraClone.Rank())
}

func TestDoRAAdapterCreation(t *testing.T) {
	config := dora.NewConfig().SetRank(8)

	adapter, err := dora.New("test_dora", config)
	require.NoError(t, err)
	require.NotNil(t, adapter)

	assert.Equal(t, "test_dora", adapter.Name())
	assert.Equal(t, adapters.AdapterTypeDoRA, adapter.Type())
	assert.False(t, adapter.IsMerged())

	// Test with nil config (should use defaults)
	adapter2, err := dora.New("default_dora", nil)
	require.NoError(t, err)
	require.NotNil(t, adapter2)
}

func TestDoRAFactory(t *testing.T) {
	factory := dora.NewFactory()

	assert.Equal(t, adapters.AdapterTypeDoRA, factory.Type())

	config := dora.NewConfig()
	adapter, err := factory.Create("test", config)
	require.NoError(t, err)
	assert.Equal(t, "test", adapter.Name())
}

func TestDoRADense(t *testing.T) {
	backend, err := backends.New()
	require.NoError(t, err)
	ctx := context.New()

	config := dora.NewConfig().SetRank(4).SetAlpha(8.0)

	batchSize := 2
	inputDim := 8
	outputDim := 4

	exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, input *graph.Node) *graph.Node {
		return dora.Dense(ctx, input, config, outputDim)
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

func TestDoRADenseWithBias(t *testing.T) {
	backend, err := backends.New()
	require.NoError(t, err)
	ctx := context.New()

	config := dora.NewConfig().SetRank(4)

	batchSize := 2
	inputDim := 8
	outputDim := 4

	exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, input *graph.Node) *graph.Node {
		return dora.DenseWithBias(ctx, input, config, true, outputDim)
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

func TestDoRADisabled(t *testing.T) {
	backend, err := backends.New()
	require.NoError(t, err)
	ctx := context.New()

	config := dora.NewConfig().SetRank(4)
	config.SetEnabled(false)

	batchSize := 2
	inputDim := 8
	outputDim := 4

	exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, input *graph.Node) *graph.Node {
		return dora.Dense(ctx, input, config, outputDim)
	})

	inputData := make([][]float32, batchSize)
	for i := range inputData {
		inputData[i] = make([]float32, inputDim)
		for j := range inputData[i] {
			inputData[i][j] = 1.0
		}
	}

	result := exec.MustExec(inputData)[0]

	// Should still work, just without DoRA contribution
	assert.Equal(t, []int{batchSize, outputDim}, result.Shape().Dimensions)
}

func TestDoRAVariables(t *testing.T) {
	backend, err := backends.New()
	require.NoError(t, err)
	ctx := context.New()

	config := dora.NewConfig().SetRank(4)

	inputDim := 8
	outputDim := 4

	// Run once to create variables
	exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, input *graph.Node) *graph.Node {
		return dora.Dense(ctx, input, config, outputDim)
	})
	inputData := make([][]float32, 1)
	inputData[0] = make([]float32, inputDim)
	for j := range inputData[0] {
		inputData[0][j] = 1.0
	}
	_ = exec.MustExec(inputData)

	// Check that DoRA variables were created
	var foundDoraA, foundDoraB, foundMagnitude, foundWeights bool
	for v := range ctx.IterVariables() {
		name := v.ParameterName()
		if strings.Contains(name, "dora_A") {
			foundDoraA = true
			// DoRA A should be [inputDim, rank]
			assert.Equal(t, []int{inputDim, 4}, v.Shape().Dimensions)
		}
		if strings.Contains(name, "dora_B") {
			foundDoraB = true
			// DoRA B should be [rank, outputDim]
			assert.Equal(t, []int{4, outputDim}, v.Shape().Dimensions)
		}
		if strings.Contains(name, "dora_magnitude") {
			foundMagnitude = true
			// Magnitude should be [outputDim]
			assert.Equal(t, []int{outputDim}, v.Shape().Dimensions)
		}
		// Base weights have "weights" in name but not "dora_A", "dora_B", or "dora_magnitude"
		if strings.Contains(name, "weights") &&
			!strings.Contains(name, "dora_A") &&
			!strings.Contains(name, "dora_B") &&
			!strings.Contains(name, "dora_magnitude") {
			foundWeights = true
			// Base weights should be [inputDim, outputDim]
			assert.Equal(t, []int{inputDim, outputDim}, v.Shape().Dimensions)
		}
	}

	assert.True(t, foundDoraA, "dora_A variable should be created")
	assert.True(t, foundDoraB, "dora_B variable should be created")
	assert.True(t, foundMagnitude, "dora_magnitude variable should be created")
	assert.True(t, foundWeights, "base weights variable should be created")
}

func TestDoRARegistration(t *testing.T) {
	// The factory should be auto-registered via init()
	factory, ok := adapters.GetFactory(adapters.AdapterTypeDoRA)
	assert.True(t, ok)
	assert.NotNil(t, factory)
	assert.Equal(t, adapters.AdapterTypeDoRA, factory.Type())
}

func TestDoRAMagnitudeTrainability(t *testing.T) {
	// Test that magnitude trainability is configurable
	configTrainable := dora.NewConfig().SetRank(4).SetTrainMagnitude(true)
	configFrozen := dora.NewConfig().SetRank(4).SetTrainMagnitude(false)

	assert.True(t, configTrainable.TrainMagnitude)
	assert.False(t, configFrozen.TrainMagnitude)
}
