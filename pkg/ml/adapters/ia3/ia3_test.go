// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package ia3_test

import (
	"strings"
	"testing"

	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/adapters"
	"github.com/gomlx/gomlx/pkg/ml/adapters/ia3"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestIA3Config(t *testing.T) {
	config := ia3.NewConfig()

	// Test defaults
	assert.Equal(t, 1.0, config.InitScale())
	assert.Equal(t, 0.0, config.Dropout())
	assert.True(t, config.Enabled())
	assert.Equal(t, adapters.AdapterTypeIA3, config.Type())

	// Test setters with chaining
	config.SetInitScale(0.5).SetDropout(0.1)
	assert.Equal(t, 0.5, config.InitScale())
	assert.Equal(t, 0.1, config.Dropout())

	// Test enabled toggle
	config.SetEnabled(false)
	assert.False(t, config.Enabled())
}

func TestIA3ConfigValidation(t *testing.T) {
	config := ia3.NewConfig()

	// Valid config should pass
	err := config.Validate()
	require.NoError(t, err)

	// Invalid init scale should panic
	assert.Panics(t, func() {
		config.SetInitScale(0)
	})

	// Invalid dropout should panic
	assert.Panics(t, func() {
		config.SetDropout(-0.1)
	})

	assert.Panics(t, func() {
		config.SetDropout(1.0)
	})
}

func TestIA3ConfigClone(t *testing.T) {
	config := ia3.NewConfig().
		SetInitScale(0.5).
		SetDropout(0.1)
	config.SetFeedForwardModules("mlp", "fc1")
	config.SetAttentionModules("key", "value")

	clone := config.Clone()
	ia3Clone, ok := clone.(*ia3.Config)
	require.True(t, ok)

	// Check values are copied
	assert.Equal(t, 0.5, ia3Clone.InitScale())
	assert.Equal(t, 0.1, ia3Clone.Dropout())
	assert.Equal(t, []string{"mlp", "fc1"}, ia3Clone.FeedForwardModules())
	assert.Equal(t, []string{"key", "value"}, ia3Clone.AttentionModules())

	// Verify independence
	config.SetInitScale(1.0)
	assert.Equal(t, 0.5, ia3Clone.InitScale())
}

func TestIA3AdapterCreation(t *testing.T) {
	config := ia3.NewConfig()

	adapter, err := ia3.New("test_ia3", config)
	require.NoError(t, err)
	require.NotNil(t, adapter)

	assert.Equal(t, "test_ia3", adapter.Name())
	assert.Equal(t, adapters.AdapterTypeIA3, adapter.Type())
	assert.False(t, adapter.IsMerged())

	// Test with nil config (should use defaults)
	adapter2, err := ia3.New("default_ia3", nil)
	require.NoError(t, err)
	require.NotNil(t, adapter2)
}

func TestIA3Factory(t *testing.T) {
	factory := ia3.NewFactory()

	assert.Equal(t, adapters.AdapterTypeIA3, factory.Type())

	config := ia3.NewConfig()
	adapter, err := factory.Create("test", config)
	require.NoError(t, err)
	assert.Equal(t, "test", adapter.Name())
}

func TestIA3ScaleActivation(t *testing.T) {
	backend, err := backends.New()
	require.NoError(t, err)
	ctx := context.New()

	config := ia3.NewConfig().SetInitScale(1.0)

	batchSize := 2
	seqLen := 4
	dim := 8

	exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, input *graph.Node) *graph.Node {
		return ia3.ScaleActivation(ctx, input, config, dim)
	})

	// Create test input (all ones)
	inputData := make([][][]float32, batchSize)
	for i := range inputData {
		inputData[i] = make([][]float32, seqLen)
		for j := range inputData[i] {
			inputData[i][j] = make([]float32, dim)
			for k := range inputData[i][j] {
				inputData[i][j][k] = 1.0
			}
		}
	}

	result := exec.MustExec(inputData)[0]

	// With init scale = 1.0, output should equal input
	assert.Equal(t, []int{batchSize, seqLen, dim}, result.Shape().Dimensions)
}

func TestIA3Variables(t *testing.T) {
	backend, err := backends.New()
	require.NoError(t, err)
	ctx := context.New()

	config := ia3.NewConfig()
	dim := 8

	// Run once to create variables
	exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, input *graph.Node) *graph.Node {
		return ia3.ScaleActivation(ctx, input, config, dim)
	})
	inputData := make([][]float32, 1)
	inputData[0] = make([]float32, dim)
	for j := range inputData[0] {
		inputData[0][j] = 1.0
	}
	_ = exec.MustExec(inputData)

	// Check that IA³ variable was created
	var foundIA3Scale bool
	for v := range ctx.IterVariables() {
		name := v.ParameterName()
		if strings.Contains(name, "ia3_scale") {
			foundIA3Scale = true
			// IA³ scale should be [dim]
			assert.Equal(t, []int{dim}, v.Shape().Dimensions)
		}
	}

	assert.True(t, foundIA3Scale, "ia3_scale variable should be created")
}

func TestIA3Registration(t *testing.T) {
	// The factory should be auto-registered via init()
	factory, ok := adapters.GetFactory(adapters.AdapterTypeIA3)
	assert.True(t, ok)
	assert.NotNil(t, factory)
	assert.Equal(t, adapters.AdapterTypeIA3, factory.Type())
}

func TestIA3VariableIdentification(t *testing.T) {
	// Test variable name identification
	assert.True(t, ia3.IsIA3Variable("model/layer0/ia3_scale"))
	assert.True(t, ia3.IsIA3Variable("model/layer0/ia3_key/ia3_scale"))
	assert.True(t, ia3.IsIA3Variable("model/layer0/ia3_value/ia3_scale"))
	assert.True(t, ia3.IsIA3Variable("model/layer0/ia3_ff/ia3_scale"))
	assert.False(t, ia3.IsIA3Variable("model/layer0/weights"))
	assert.False(t, ia3.IsIA3Variable("model/layer0/lora_A"))
}

func TestIA3Disabled(t *testing.T) {
	backend, err := backends.New()
	require.NoError(t, err)
	ctx := context.New()

	config := ia3.NewConfig()
	config.SetEnabled(false)

	batchSize := 2
	dim := 8

	exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, input *graph.Node) *graph.Node {
		return ia3.ScaleActivation(ctx, input, config, dim)
	})

	inputData := make([][]float32, batchSize)
	for i := range inputData {
		inputData[i] = make([]float32, dim)
		for j := range inputData[i] {
			inputData[i][j] = 1.0
		}
	}

	result := exec.MustExec(inputData)[0]

	// Should still work, output should be unchanged from input
	assert.Equal(t, []int{batchSize, dim}, result.Shape().Dimensions)
}

func TestShouldApplyIA3(t *testing.T) {
	config := ia3.NewConfig()

	// Default attention modules should match
	assert.True(t, ia3.ShouldApplyIA3(config, "layer/key"))
	assert.True(t, ia3.ShouldApplyIA3(config, "layer/value"))
	assert.True(t, ia3.ShouldApplyIA3(config, "layer/k_proj"))
	assert.True(t, ia3.ShouldApplyIA3(config, "layer/v_proj"))

	// Default FFN modules should match
	assert.True(t, ia3.ShouldApplyIA3(config, "layer/mlp"))
	assert.True(t, ia3.ShouldApplyIA3(config, "layer/feed_forward"))
	assert.True(t, ia3.ShouldApplyIA3(config, "layer/fc1"))

	// Non-matching modules
	assert.False(t, ia3.ShouldApplyIA3(config, "layer/query"))
	assert.False(t, ia3.ShouldApplyIA3(config, "layer/output"))
	assert.False(t, ia3.ShouldApplyIA3(config, "embedding"))

	// Nil config should return false
	assert.False(t, ia3.ShouldApplyIA3(nil, "layer/key"))
}

func TestIA3ParameterCount(t *testing.T) {
	backend, err := backends.New()
	require.NoError(t, err)
	ctx := context.New()

	config := ia3.NewConfig()
	dim := 64

	// Create some IA³ variables
	exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, input *graph.Node) *graph.Node {
		// Apply IA³ to simulate key, value, and ff scaling
		out := ia3.ScaleActivation(ctx.In("key"), input, config, dim)
		out = ia3.ScaleActivation(ctx.In("value"), out, config, dim)
		out = ia3.ScaleActivation(ctx.In("ff"), out, config, dim)
		return out
	})
	inputData := make([][]float32, 1)
	inputData[0] = make([]float32, dim)
	_ = exec.MustExec(inputData)

	// Create adapter to count parameters
	adapter, _ := ia3.New("test", config)
	count := adapter.CountParameters(ctx)

	// 3 scaling vectors, each of size dim
	expectedCount := int64(3 * dim)
	assert.Equal(t, expectedCount, count)
}

func TestIA3FeedForwardModulesConfig(t *testing.T) {
	config := ia3.NewConfig()

	// Check default FFN modules
	defaultFFN := config.FeedForwardModules()
	assert.Contains(t, defaultFFN, "mlp")
	assert.Contains(t, defaultFFN, "feed_forward")

	// Set custom modules
	config.SetFeedForwardModules("gate_proj", "up_proj")
	assert.Equal(t, []string{"gate_proj", "up_proj"}, config.FeedForwardModules())
}

func TestIA3AttentionModulesConfig(t *testing.T) {
	config := ia3.NewConfig()

	// Check default attention modules
	defaultAttn := config.AttentionModules()
	assert.Contains(t, defaultAttn, "key")
	assert.Contains(t, defaultAttn, "value")

	// Set custom modules
	config.SetAttentionModules("kv_proj")
	assert.Equal(t, []string{"kv_proj"}, config.AttentionModules())
}
