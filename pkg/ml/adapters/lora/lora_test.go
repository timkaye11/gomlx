// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package lora_test

import (
	"strings"
	"testing"

	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/adapters"
	"github.com/gomlx/gomlx/pkg/ml/adapters/lora"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLoRAConfig(t *testing.T) {
	config := lora.NewConfig()

	// Test defaults
	assert.Equal(t, 8, config.Rank())
	assert.Equal(t, 16.0, config.Alpha())
	assert.Equal(t, 0.0, config.Dropout())
	assert.True(t, config.Enabled())
	assert.Equal(t, adapters.AdapterTypeLoRA, config.Type())

	// Test setters with chaining
	config.SetRank(16).SetAlpha(32.0).SetDropout(0.1)
	assert.Equal(t, 16, config.Rank())
	assert.Equal(t, 32.0, config.Alpha())
	assert.Equal(t, 0.1, config.Dropout())

	// Test scale calculation
	assert.Equal(t, 2.0, config.Scale()) // 32.0 / 16 = 2.0

	// Test enabled toggle
	config.SetEnabled(false)
	assert.False(t, config.Enabled())
}

func TestLoRAConfigValidation(t *testing.T) {
	config := lora.NewConfig()

	// Valid config should pass
	err := config.Validate()
	require.NoError(t, err)

	// Invalid rank should panic
	assert.Panics(t, func() {
		config.SetRank(0)
	})

	assert.Panics(t, func() {
		config.SetRank(-1)
	})

	// Invalid dropout should panic
	assert.Panics(t, func() {
		config.SetDropout(-0.1)
	})

	assert.Panics(t, func() {
		config.SetDropout(1.0)
	})
}

func TestLoRAConfigClone(t *testing.T) {
	config := lora.NewConfig().
		SetRank(16).
		SetAlpha(32.0).
		SetDropout(0.1)
	config.SetTargetModules("query", "value")

	clone := config.Clone()
	loraClone, ok := clone.(*lora.Config)
	require.True(t, ok)

	// Check values are copied
	assert.Equal(t, 16, loraClone.Rank())
	assert.Equal(t, 32.0, loraClone.Alpha())
	assert.Equal(t, 0.1, loraClone.Dropout())
	assert.Equal(t, []string{"query", "value"}, loraClone.TargetModules())

	// Verify independence
	config.SetRank(8)
	assert.Equal(t, 16, loraClone.Rank())
}

func TestLoRAAdapterCreation(t *testing.T) {
	config := lora.NewConfig().SetRank(8)

	adapter, err := lora.New("test_lora", config)
	require.NoError(t, err)
	require.NotNil(t, adapter)

	assert.Equal(t, "test_lora", adapter.Name())
	assert.Equal(t, adapters.AdapterTypeLoRA, adapter.Type())
	assert.False(t, adapter.IsMerged())

	// Test with nil config (should use defaults)
	adapter2, err := lora.New("default_lora", nil)
	require.NoError(t, err)
	require.NotNil(t, adapter2)
}

func TestLoRAFactory(t *testing.T) {
	factory := lora.NewFactory()

	assert.Equal(t, adapters.AdapterTypeLoRA, factory.Type())

	config := lora.NewConfig()
	adapter, err := factory.Create("test", config)
	require.NoError(t, err)
	assert.Equal(t, "test", adapter.Name())

	// Test with wrong config type
	wrongConfig := &wrongConfigType{}
	_, err = factory.Create("test", wrongConfig)
	require.Error(t, err)
}

type wrongConfigType struct {
	adapters.BaseConfig
}

func (c *wrongConfigType) Type() adapters.AdapterType { return adapters.AdapterType("wrong") }
func (c *wrongConfigType) Validate() error           { return nil }
func (c *wrongConfigType) Clone() adapters.AdapterConfig {
	clone := *c
	return &clone
}
func (c *wrongConfigType) SetEnabled(enabled bool) adapters.AdapterConfig {
	c.BaseConfig.SetEnabled(enabled)
	return c
}
func (c *wrongConfigType) SetTargetModules(modules ...string) adapters.AdapterConfig {
	c.BaseConfig.SetTargetModules(modules...)
	return c
}

func TestLoRADense(t *testing.T) {
	backend, err := backends.New()
	require.NoError(t, err)
	ctx := context.New()

	config := lora.NewConfig().SetRank(4).SetAlpha(8.0)

	batchSize := 2
	inputDim := 8
	outputDim := 4

	// Use context.MustNewExec to handle variables properly
	exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, input *graph.Node) *graph.Node {
		return lora.Dense(ctx, input, config, outputDim)
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

func TestLoRADenseWithBias(t *testing.T) {
	backend, err := backends.New()
	require.NoError(t, err)
	ctx := context.New()

	config := lora.NewConfig().SetRank(4)

	batchSize := 2
	inputDim := 8
	outputDim := 4

	exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, input *graph.Node) *graph.Node {
		return lora.DenseWithBias(ctx, input, config, true, outputDim)
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

func TestLoRADisabled(t *testing.T) {
	backend, err := backends.New()
	require.NoError(t, err)
	ctx := context.New()

	config := lora.NewConfig().SetRank(4)
	config.SetEnabled(false)

	batchSize := 2
	inputDim := 8
	outputDim := 4

	exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, input *graph.Node) *graph.Node {
		return lora.Dense(ctx, input, config, outputDim)
	})

	inputData := make([][]float32, batchSize)
	for i := range inputData {
		inputData[i] = make([]float32, inputDim)
		for j := range inputData[i] {
			inputData[i][j] = 1.0
		}
	}

	result := exec.MustExec(inputData)[0]

	// Should still work, just without LoRA contribution
	assert.Equal(t, []int{batchSize, outputDim}, result.Shape().Dimensions)
}

func TestLoRAVariables(t *testing.T) {
	backend, err := backends.New()
	require.NoError(t, err)
	ctx := context.New()

	config := lora.NewConfig().SetRank(4)

	inputDim := 8
	outputDim := 4

	// Run once to create variables
	exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, input *graph.Node) *graph.Node {
		return lora.Dense(ctx, input, config, outputDim)
	})
	inputData := make([][]float32, 1)
	inputData[0] = make([]float32, inputDim)
	for j := range inputData[0] {
		inputData[0][j] = 1.0
	}
	_ = exec.MustExec(inputData)

	// Check that LoRA variables were created
	var foundLoraA, foundLoraB, foundWeights bool
	for v := range ctx.IterVariables() {
		name := v.ParameterName()
		if strings.Contains(name, "lora_A") {
			foundLoraA = true
			// LoRA A should be [inputDim, rank]
			assert.Equal(t, []int{inputDim, 4}, v.Shape().Dimensions)
		}
		if strings.Contains(name, "lora_B") {
			foundLoraB = true
			// LoRA B should be [rank, outputDim]
			assert.Equal(t, []int{4, outputDim}, v.Shape().Dimensions)
		}
		// Base weights contain "weights" but not "lora_A" or "lora_B"
		if strings.Contains(name, "weights") && !strings.Contains(name, "lora_A") && !strings.Contains(name, "lora_B") {
			foundWeights = true
			// Base weights should be [inputDim, outputDim]
			assert.Equal(t, []int{inputDim, outputDim}, v.Shape().Dimensions)
		}
	}

	assert.True(t, foundLoraA, "lora_A variable should be created")
	assert.True(t, foundLoraB, "lora_B variable should be created")
	assert.True(t, foundWeights, "base weights variable should be created")
}

func TestLoRARegistration(t *testing.T) {
	// The factory should be auto-registered via init()
	factory, ok := adapters.GetFactory(adapters.AdapterTypeLoRA)
	assert.True(t, ok)
	assert.NotNil(t, factory)
	assert.Equal(t, adapters.AdapterTypeLoRA, factory.Type())
}

func TestRSLoRAScaling(t *testing.T) {
	// Standard LoRA: scale = alpha / rank
	config := lora.NewConfig().SetRank(16).SetAlpha(32.0)
	assert.Equal(t, 2.0, config.Scale()) // 32.0 / 16 = 2.0
	assert.False(t, config.UseRSLoRA)

	// rsLoRA: scale = alpha / sqrt(rank)
	config.SetUseRSLoRA(true)
	assert.True(t, config.UseRSLoRA)
	// 32.0 / sqrt(16) = 32.0 / 4 = 8.0
	assert.Equal(t, 8.0, config.Scale())

	// Test with rank=64: standard = 32/64 = 0.5, rsLoRA = 32/8 = 4
	config.SetRank(64)
	assert.Equal(t, 4.0, config.Scale()) // rsLoRA: 32.0 / sqrt(64) = 32.0 / 8 = 4.0

	config.SetUseRSLoRA(false)
	assert.Equal(t, 0.5, config.Scale()) // standard: 32.0 / 64 = 0.5

	// rsLoRA should be cloned correctly
	config.SetUseRSLoRA(true)
	clone := config.Clone().(*lora.Config)
	assert.True(t, clone.UseRSLoRA)
}

func TestLoRAPlusConfig(t *testing.T) {
	config := lora.NewConfig()

	// Check defaults
	assert.False(t, config.UseLoRAPlus)
	assert.Equal(t, 16.0, config.LoRAPlusRatio) // Default ratio

	// Enable LoRA+
	config.SetUseLoRAPlus(true)
	assert.True(t, config.UseLoRAPlus)

	// Change ratio
	config.SetLoRAPlusRatio(8.0)
	assert.Equal(t, 8.0, config.LoRAPlusRatio)

	// Invalid ratio should panic
	assert.Panics(t, func() {
		config.SetLoRAPlusRatio(0)
	})
	assert.Panics(t, func() {
		config.SetLoRAPlusRatio(-1.0)
	})

	// Clone should preserve LoRA+ settings
	config.SetLoRAPlusRatio(32.0)
	clone := config.Clone().(*lora.Config)
	assert.True(t, clone.UseLoRAPlus)
	assert.Equal(t, 32.0, clone.LoRAPlusRatio)
}

func TestLoRAPlusVariableIdentification(t *testing.T) {
	// Test variable name identification
	assert.True(t, lora.IsLoRAAVariable("model/layer0/lora_A"))
	assert.True(t, lora.IsLoRAAVariable("encoder/attention/lora_A_weights"))
	assert.False(t, lora.IsLoRAAVariable("model/layer0/lora_B"))
	assert.False(t, lora.IsLoRAAVariable("model/layer0/weights"))

	assert.True(t, lora.IsLoRABVariable("model/layer0/lora_B"))
	assert.True(t, lora.IsLoRABVariable("encoder/attention/lora_B_weights"))
	assert.False(t, lora.IsLoRABVariable("model/layer0/lora_A"))
	assert.False(t, lora.IsLoRABVariable("model/layer0/weights"))

	assert.True(t, lora.IsLoRAVariable("model/layer0/lora_A"))
	assert.True(t, lora.IsLoRAVariable("model/layer0/lora_B"))
	assert.False(t, lora.IsLoRAVariable("model/layer0/weights"))
}

func TestLoRAPlusLearningRateMultiplier(t *testing.T) {
	config := lora.NewConfig().SetUseLoRAPlus(true).SetLoRAPlusRatio(16.0)

	// A matrices should have multiplier 1.0
	assert.Equal(t, 1.0, lora.GetLearningRateMultiplierForName(config, "model/layer0/lora_A"))

	// B matrices should have multiplier = ratio
	assert.Equal(t, 16.0, lora.GetLearningRateMultiplierForName(config, "model/layer0/lora_B"))

	// Non-LoRA variables should have multiplier 1.0
	assert.Equal(t, 1.0, lora.GetLearningRateMultiplierForName(config, "model/layer0/weights"))

	// With LoRA+ disabled, all multipliers should be 1.0
	config.SetUseLoRAPlus(false)
	assert.Equal(t, 1.0, lora.GetLearningRateMultiplierForName(config, "model/layer0/lora_B"))

	// Nil config should return 1.0
	assert.Equal(t, 1.0, lora.GetLearningRateMultiplierForName(nil, "model/layer0/lora_B"))
}

func TestLoRAPlusPartitionVariables(t *testing.T) {
	backend, err := backends.New()
	require.NoError(t, err)
	ctx := context.New()

	config := lora.NewConfig().SetRank(4)
	inputDim := 8
	outputDim := 4

	// Create some LoRA variables by running the dense layer
	exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, input *graph.Node) *graph.Node {
		return lora.Dense(ctx, input, config, outputDim)
	})
	inputData := make([][]float32, 1)
	inputData[0] = make([]float32, inputDim)
	_ = exec.MustExec(inputData)

	// Partition variables
	loraA, loraB, other := lora.PartitionVariables(ctx)

	assert.Len(t, loraA, 1, "Should have 1 lora_A variable")
	assert.Len(t, loraB, 1, "Should have 1 lora_B variable")
	assert.GreaterOrEqual(t, len(other), 1, "Should have at least 1 other variable (base weights)")

	// Verify the names
	assert.True(t, lora.IsLoRAAVariable(loraA[0].ParameterName()))
	assert.True(t, lora.IsLoRABVariable(loraB[0].ParameterName()))
}

func TestLoRAPlusCountParameters(t *testing.T) {
	backend, err := backends.New()
	require.NoError(t, err)
	ctx := context.New()

	rank := 4
	config := lora.NewConfig().SetRank(rank)
	inputDim := 8
	outputDim := 4

	// Create LoRA variables
	exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, input *graph.Node) *graph.Node {
		return lora.Dense(ctx, input, config, outputDim)
	})
	inputData := make([][]float32, 1)
	inputData[0] = make([]float32, inputDim)
	_ = exec.MustExec(inputData)

	// Count LoRA parameters
	// lora_A: inputDim * rank = 8 * 4 = 32
	// lora_B: rank * outputDim = 4 * 4 = 16
	// Total: 48
	count := lora.CountLoRAParameters(ctx)
	expectedCount := int64(inputDim*rank + rank*outputDim)
	assert.Equal(t, expectedCount, count)
}

func TestSetupLoRAPlus(t *testing.T) {
	ctx := context.New()

	// Setup with LoRA+ enabled
	config := lora.NewConfig().SetUseLoRAPlus(true).SetLoRAPlusRatio(16.0)
	lora.SetupLoRAPlus(ctx, config)

	// Verify config was stored
	configAny, found := ctx.GetParam(lora.ParamLoRAPlusConfig)
	assert.True(t, found)
	storedConfig, ok := configAny.(*lora.Config)
	assert.True(t, ok)
	assert.True(t, storedConfig.UseLoRAPlus)
	assert.Equal(t, 16.0, storedConfig.LoRAPlusRatio)

	// Setup with LoRA+ disabled should not store
	ctx2 := context.New()
	config2 := lora.NewConfig() // UseLoRAPlus defaults to false
	lora.SetupLoRAPlus(ctx2, config2)
	_, found = ctx2.GetParam(lora.ParamLoRAPlusConfig)
	assert.False(t, found)
}
