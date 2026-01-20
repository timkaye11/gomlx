// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package qlora_test

import (
	"strings"
	"testing"

	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/adapters"
	"github.com/gomlx/gomlx/pkg/ml/adapters/qlora"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/quantization"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestQLoRAConfig(t *testing.T) {
	config := qlora.NewConfig()

	// Test defaults from LoRA
	assert.Equal(t, 8, config.Rank())
	assert.Equal(t, 16.0, config.Alpha())
	assert.Equal(t, 0.0, config.Dropout())
	assert.True(t, config.Enabled())
	assert.Equal(t, adapters.AdapterTypeQLoRA, config.Type())

	// Test quantization defaults
	assert.Equal(t, quantization.NF4, config.QuantConfig.QuantType)
	assert.Equal(t, 64, config.QuantConfig.GroupSize)
	assert.Equal(t, dtypes.BFloat16, config.QuantConfig.ComputeDType)
}

func TestQLoRAConfigSetters(t *testing.T) {
	config := qlora.NewConfig().
		SetRank(16).
		SetAlpha(32.0).
		SetDropout(0.1).
		SetQuantType(quantization.Int4).
		SetGroupSize(32)

	assert.Equal(t, 16, config.Rank())
	assert.Equal(t, 32.0, config.Alpha())
	assert.Equal(t, 0.1, config.Dropout())
	assert.Equal(t, quantization.Int4, config.QuantConfig.QuantType)
	assert.Equal(t, 32, config.QuantConfig.GroupSize)

	// Test scale calculation
	assert.Equal(t, 2.0, config.Scale()) // 32.0 / 16 = 2.0
}

func TestQLoRAConfigValidation(t *testing.T) {
	config := qlora.NewConfig()

	// Valid config should pass
	err := config.Validate()
	require.NoError(t, err)

	// Invalid rank should panic
	assert.Panics(t, func() {
		config.SetRank(0)
	})
}

func TestQLoRAConfigClone(t *testing.T) {
	config := qlora.NewConfig().
		SetRank(16).
		SetAlpha(32.0).
		SetQuantType(quantization.Int8).
		SetGroupSize(128)

	clone := config.Clone()
	qloraClone, ok := clone.(*qlora.Config)
	require.True(t, ok)

	// Check values are copied
	assert.Equal(t, 16, qloraClone.Rank())
	assert.Equal(t, 32.0, qloraClone.Alpha())
	assert.Equal(t, quantization.Int8, qloraClone.QuantConfig.QuantType)
	assert.Equal(t, 128, qloraClone.QuantConfig.GroupSize)

	// Verify independence
	config.SetRank(8)
	assert.Equal(t, 16, qloraClone.Rank())
}

func TestQLoRAAdapterCreation(t *testing.T) {
	config := qlora.NewConfig().SetRank(8)

	adapter, err := qlora.New("test_qlora", config)
	require.NoError(t, err)
	require.NotNil(t, adapter)

	assert.Equal(t, "test_qlora", adapter.Name())
	assert.Equal(t, adapters.AdapterTypeQLoRA, adapter.Type())
	assert.False(t, adapter.IsMerged())

	// Test with nil config (should use defaults)
	adapter2, err := qlora.New("default_qlora", nil)
	require.NoError(t, err)
	require.NotNil(t, adapter2)
}

func TestQLoRAFactory(t *testing.T) {
	factory := qlora.NewFactory()

	assert.Equal(t, adapters.AdapterTypeQLoRA, factory.Type())

	config := qlora.NewConfig()
	adapter, err := factory.Create("test", config)
	require.NoError(t, err)
	assert.Equal(t, "test", adapter.Name())
}

func TestQLoRADense(t *testing.T) {
	backend, err := backends.New()
	require.NoError(t, err)
	ctx := context.New()

	config := qlora.NewConfig().SetRank(4).SetAlpha(8.0)

	batchSize := 2
	inputDim := 8
	outputDim := 4

	// Use Float32 for testing since BFloat16 might not be supported everywhere
	config.QuantConfig.ComputeDType = dtypes.Float32

	exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, input *graph.Node) *graph.Node {
		return qlora.Dense(ctx, input, config, outputDim)
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

func TestQLoRADenseWithBias(t *testing.T) {
	backend, err := backends.New()
	require.NoError(t, err)
	ctx := context.New()

	config := qlora.NewConfig().SetRank(4)
	config.QuantConfig.ComputeDType = dtypes.Float32

	batchSize := 2
	inputDim := 8
	outputDim := 4

	exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, input *graph.Node) *graph.Node {
		return qlora.DenseWithBias(ctx, input, config, true, outputDim)
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

func TestQLoRADisabled(t *testing.T) {
	backend, err := backends.New()
	require.NoError(t, err)
	ctx := context.New()

	config := qlora.NewConfig().SetRank(4)
	config.SetEnabled(false)
	config.QuantConfig.ComputeDType = dtypes.Float32

	batchSize := 2
	inputDim := 8
	outputDim := 4

	exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, input *graph.Node) *graph.Node {
		return qlora.Dense(ctx, input, config, outputDim)
	})

	inputData := make([][]float32, batchSize)
	for i := range inputData {
		inputData[i] = make([]float32, inputDim)
		for j := range inputData[i] {
			inputData[i][j] = 1.0
		}
	}

	result := exec.MustExec(inputData)[0]

	// Should still work, just without QLoRA contribution
	assert.Equal(t, []int{batchSize, outputDim}, result.Shape().Dimensions)
}

func TestQLoRAVariables(t *testing.T) {
	backend, err := backends.New()
	require.NoError(t, err)
	ctx := context.New()

	config := qlora.NewConfig().SetRank(4)
	config.QuantConfig.ComputeDType = dtypes.Float32

	inputDim := 8
	outputDim := 4

	// Run once to create variables
	exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, input *graph.Node) *graph.Node {
		return qlora.Dense(ctx, input, config, outputDim)
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
		}
	}

	assert.True(t, foundLoraA, "lora_A variable should be created")
	assert.True(t, foundLoraB, "lora_B variable should be created")
	assert.True(t, foundWeights, "base weights variable should be created")
}

func TestQLoRARegistration(t *testing.T) {
	// The factory should be auto-registered via init()
	factory, ok := adapters.GetFactory(adapters.AdapterTypeQLoRA)
	assert.True(t, ok)
	assert.NotNil(t, factory)
	assert.Equal(t, adapters.AdapterTypeQLoRA, factory.Type())
}

func TestQLoRAMemorySavings(t *testing.T) {
	// Verify the memory savings of QLoRA vs full fine-tuning
	inputDim := 4096  // Example LLM hidden dimension
	outputDim := 4096
	rank := 16

	// Full fine-tuning: all weights are trainable
	fullParams := inputDim * outputDim

	// QLoRA: only A and B are trainable (base weights are quantized and frozen)
	qloraParams := inputDim*rank + rank*outputDim

	// QLoRA should be much smaller
	ratio := float64(fullParams) / float64(qloraParams)
	assert.Greater(t, ratio, 100.0, "QLoRA should have >100x fewer trainable params")
	t.Logf("Full fine-tuning params: %d, QLoRA params: %d, ratio: %.1fx", fullParams, qloraParams, ratio)

	// Memory for base weights
	// Full: 4096*4096*4 bytes (float32) = 64 MB
	// QLoRA: 4096*4096/2 bytes (4-bit) + scales = ~8-9 MB
	fullMemory := fullParams * 4 // float32
	qloraMemory := fullParams/2 + fullParams/64*2 // packed + scales (float16)
	memRatio := float64(fullMemory) / float64(qloraMemory)
	assert.Greater(t, memRatio, 5.0, "QLoRA should use >5x less memory for weights")
	t.Logf("Full memory: %d bytes, QLoRA memory: %d bytes, ratio: %.1fx", fullMemory, qloraMemory, memRatio)
}
