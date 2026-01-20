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

func TestNewManager(t *testing.T) {
	ctx := context.New()
	manager := adapters.NewManager(ctx)

	require.NotNil(t, manager)
	assert.Equal(t, 0, manager.Count())
	assert.Empty(t, manager.ActiveName())
	assert.Nil(t, manager.Active())
}

func TestManagerAddRemove(t *testing.T) {
	ctx := context.New()
	manager := adapters.NewManager(ctx)

	// Create mock adapters
	adapter1 := createTestAdapterForManager("adapter1")
	adapter2 := createTestAdapterForManager("adapter2")

	// Add adapters
	err := manager.Add(adapter1)
	require.NoError(t, err)
	assert.Equal(t, 1, manager.Count())

	err = manager.Add(adapter2)
	require.NoError(t, err)
	assert.Equal(t, 2, manager.Count())

	// Try to add duplicate
	err = manager.Add(adapter1)
	require.Error(t, err)

	// Get adapter
	got, ok := manager.Get("adapter1")
	assert.True(t, ok)
	assert.Equal(t, adapter1, got)

	got, ok = manager.Get("nonexistent")
	assert.False(t, ok)
	assert.Nil(t, got)

	// List adapters
	names := manager.List()
	assert.Len(t, names, 2)
	assert.Contains(t, names, "adapter1")
	assert.Contains(t, names, "adapter2")

	// Remove adapter (without activating first)
	err = manager.Remove("adapter1")
	require.NoError(t, err)
	assert.Equal(t, 1, manager.Count())

	// Try to remove nonexistent
	err = manager.Remove("nonexistent")
	require.Error(t, err)
}

func TestManagerActivation(t *testing.T) {
	ctx := context.New()
	manager := adapters.NewManager(ctx)

	adapter1 := createTestAdapterForManager("adapter1")
	adapter2 := createTestAdapterForManager("adapter2")

	manager.Add(adapter1)
	manager.Add(adapter2)

	// Initially no active adapter
	assert.Empty(t, manager.ActiveName())
	assert.Nil(t, manager.Active())

	// Activate adapter1
	err := manager.SetActive("adapter1")
	require.NoError(t, err)
	assert.Equal(t, "adapter1", manager.ActiveName())
	assert.Equal(t, adapter1, manager.Active())

	// Switch to adapter2
	err = manager.SetActive("adapter2")
	require.NoError(t, err)
	assert.Equal(t, "adapter2", manager.ActiveName())
	assert.Equal(t, adapter2, manager.Active())

	// Deactivate all
	err = manager.SetActive("")
	require.NoError(t, err)
	assert.Empty(t, manager.ActiveName())
	assert.Nil(t, manager.Active())

	// Try to activate nonexistent
	err = manager.SetActive("nonexistent")
	require.Error(t, err)
}

func TestManagerRemoveActive(t *testing.T) {
	ctx := context.New()
	manager := adapters.NewManager(ctx)

	adapter := createTestAdapterForManager("test")
	manager.Add(adapter)
	manager.SetActive("test")

	// Should fail to remove active adapter
	err := manager.Remove("test")
	require.Error(t, err)

	// Deactivate first
	manager.SetActive("")
	err = manager.Remove("test")
	require.NoError(t, err)
}

func TestManagerContext(t *testing.T) {
	ctx := context.New()
	manager := adapters.NewManager(ctx)

	// Test SetInContext and GetFromContext
	adapters.SetInContext(ctx, manager)
	got := adapters.GetFromContext(ctx)
	assert.Equal(t, manager, got)

	// Test with context without manager
	ctx2 := context.New()
	got = adapters.GetFromContext(ctx2)
	assert.Nil(t, got)
}

// testAdapterForManager is a minimal adapter implementation for testing
type testAdapterForManager struct {
	adapters.BaseAdapter
	cfg *testConfigForManager
}

type testConfigForManager struct {
	adapters.BaseConfig
}

func (c *testConfigForManager) Type() adapters.AdapterType {
	return adapters.AdapterTypeLoRA
}

func (c *testConfigForManager) Validate() error {
	return nil
}

func (c *testConfigForManager) Clone() adapters.AdapterConfig {
	clone := *c
	return &clone
}

func (c *testConfigForManager) SetEnabled(enabled bool) adapters.AdapterConfig {
	c.BaseConfig.SetEnabled(enabled)
	return c
}

func (c *testConfigForManager) SetTargetModules(modules ...string) adapters.AdapterConfig {
	c.BaseConfig.SetTargetModules(modules...)
	return c
}

func createTestAdapterForManager(name string) *testAdapterForManager {
	return &testAdapterForManager{
		BaseAdapter: adapters.NewBaseAdapter(name),
		cfg:         &testConfigForManager{BaseConfig: adapters.NewBaseConfig()},
	}
}

func (a *testAdapterForManager) Type() adapters.AdapterType {
	return adapters.AdapterTypeLoRA
}

func (a *testAdapterForManager) Config() adapters.AdapterConfig {
	return a.cfg
}

func (a *testAdapterForManager) Apply(ctx *context.Context, input, baseOutput *graph.Node, inputDim, outputDim int) *graph.Node {
	return baseOutput
}

func (a *testAdapterForManager) Merge(ctx *context.Context, baseWeights *graph.Node) *graph.Node {
	return baseWeights
}

func (a *testAdapterForManager) CountParameters(ctx *context.Context) int64 {
	return 0
}

func (a *testAdapterForManager) FreezeBaseWeights(ctx *context.Context) {}

func (a *testAdapterForManager) UnfreezeBaseWeights(ctx *context.Context) {}
