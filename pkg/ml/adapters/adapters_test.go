// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package adapters_test

import (
	"testing"

	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/adapters"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAdapterTypes(t *testing.T) {
	assert.Equal(t, adapters.AdapterType("lora"), adapters.AdapterTypeLoRA)
	assert.Equal(t, adapters.AdapterType("qlora"), adapters.AdapterTypeQLoRA)
	assert.Equal(t, adapters.AdapterType("adalora"), adapters.AdapterTypeAdaLoRA)
	assert.Equal(t, adapters.AdapterType("dora"), adapters.AdapterTypeDoRA)
}

func TestBaseConfig(t *testing.T) {
	config := adapters.NewBaseConfig()

	// Test defaults
	assert.True(t, config.Enabled())
	assert.Nil(t, config.TargetModules())

	// Test setters
	config.SetEnabled(false)
	assert.False(t, config.Enabled())

	config.SetTargetModules("query", "key", "value")
	assert.Equal(t, []string{"query", "key", "value"}, config.TargetModules())

	// Test clone
	clone := config.Clone()
	assert.False(t, clone.Enabled())
	assert.Equal(t, []string{"query", "key", "value"}, clone.TargetModules())

	// Verify clone is independent
	config.SetEnabled(true)
	assert.True(t, config.Enabled())
	assert.False(t, clone.Enabled())
}

func TestBaseAdapter(t *testing.T) {
	adapter := adapters.NewBaseAdapter("test_adapter")

	assert.Equal(t, "test_adapter", adapter.Name())
	assert.False(t, adapter.IsMerged())

	adapter.SetMerged(true)
	assert.True(t, adapter.IsMerged())
}

func TestValidateRank(t *testing.T) {
	// Valid ranks should not panic
	assert.NotPanics(t, func() {
		adapters.ValidateRank(1)
		adapters.ValidateRank(8)
		adapters.ValidateRank(64)
	})

	// Invalid ranks should panic
	assert.Panics(t, func() {
		adapters.ValidateRank(0)
	})
	assert.Panics(t, func() {
		adapters.ValidateRank(-1)
	})
}

func TestValidateAlpha(t *testing.T) {
	// Valid alpha should not panic
	assert.NotPanics(t, func() {
		adapters.ValidateAlpha(1.0)
		adapters.ValidateAlpha(16.0)
		adapters.ValidateAlpha(0.001)
	})

	// Invalid alpha should panic
	assert.Panics(t, func() {
		adapters.ValidateAlpha(0.0)
	})
	assert.Panics(t, func() {
		adapters.ValidateAlpha(-1.0)
	})
}

func TestValidateDropout(t *testing.T) {
	// Valid dropout should not panic
	assert.NotPanics(t, func() {
		adapters.ValidateDropout(0.0)
		adapters.ValidateDropout(0.1)
		adapters.ValidateDropout(0.5)
		adapters.ValidateDropout(0.99)
	})

	// Invalid dropout should panic
	assert.Panics(t, func() {
		adapters.ValidateDropout(-0.1)
	})
	assert.Panics(t, func() {
		adapters.ValidateDropout(1.0)
	})
	assert.Panics(t, func() {
		adapters.ValidateDropout(1.1)
	})
}

func TestRegistry(t *testing.T) {
	registry := adapters.NewRegistry()

	// Initially empty
	assert.Equal(t, 0, registry.Count())

	// Create mock factory
	factory := &mockFactory{adapterType: adapters.AdapterTypeLoRA}

	// Register
	err := registry.Register(factory)
	require.NoError(t, err)
	assert.Equal(t, 1, registry.Count())

	// Get factory
	got, ok := registry.Get(adapters.AdapterTypeLoRA)
	assert.True(t, ok)
	assert.Equal(t, factory, got)

	// Try to register same type again
	err = registry.Register(factory)
	require.Error(t, err)

	// Types
	types := registry.Types()
	assert.Contains(t, types, adapters.AdapterTypeLoRA)

	// Unregister
	registry.Unregister(adapters.AdapterTypeLoRA)
	assert.Equal(t, 0, registry.Count())

	_, ok = registry.Get(adapters.AdapterTypeLoRA)
	assert.False(t, ok)
}

func TestRegistryCreate(t *testing.T) {
	registry := adapters.NewRegistry()
	factory := &mockFactory{adapterType: adapters.AdapterTypeLoRA}
	registry.MustRegister(factory)

	// Create with valid config
	config := &mockConfig{adapterType: adapters.AdapterTypeLoRA}
	adapter, err := registry.Create("test", config)
	require.NoError(t, err)
	assert.Equal(t, "test", adapter.Name())

	// Create with unregistered type
	unknownConfig := &mockConfig{adapterType: adapters.AdapterType("unknown")}
	_, err = registry.Create("test", unknownConfig)
	require.Error(t, err)
}

// mockFactory implements AdapterFactory for testing
type mockFactory struct {
	adapterType adapters.AdapterType
}

func (f *mockFactory) Type() adapters.AdapterType {
	return f.adapterType
}

func (f *mockFactory) Create(name string, config adapters.AdapterConfig) (adapters.Adapter, error) {
	return &mockAdapter{
		BaseAdapter: adapters.NewBaseAdapter(name),
		config:      config,
	}, nil
}

// mockConfig implements AdapterConfig for testing
type mockConfig struct {
	adapters.BaseConfig
	adapterType adapters.AdapterType
}

func (c *mockConfig) Type() adapters.AdapterType {
	return c.adapterType
}

func (c *mockConfig) Validate() error {
	return nil
}

func (c *mockConfig) Clone() adapters.AdapterConfig {
	clone := *c
	return &clone
}

func (c *mockConfig) SetEnabled(enabled bool) adapters.AdapterConfig {
	c.BaseConfig.SetEnabled(enabled)
	return c
}

func (c *mockConfig) SetTargetModules(modules ...string) adapters.AdapterConfig {
	c.BaseConfig.SetTargetModules(modules...)
	return c
}

// mockAdapter implements Adapter for testing
type mockAdapter struct {
	adapters.BaseAdapter
	config adapters.AdapterConfig
}

func (a *mockAdapter) Type() adapters.AdapterType {
	return a.config.Type()
}

func (a *mockAdapter) Config() adapters.AdapterConfig {
	return a.config
}

func (a *mockAdapter) Apply(ctx *context.Context, input, baseOutput *graph.Node, inputDim, outputDim int) *graph.Node {
	return baseOutput
}

func (a *mockAdapter) Merge(ctx *context.Context, baseWeights *graph.Node) *graph.Node {
	return baseWeights
}

func (a *mockAdapter) CountParameters(ctx *context.Context) int64 {
	return 0
}

func (a *mockAdapter) FreezeBaseWeights(ctx *context.Context) {}

func (a *mockAdapter) UnfreezeBaseWeights(ctx *context.Context) {}
