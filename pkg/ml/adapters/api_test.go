// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package adapters_test

import (
	"testing"

	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/pkg/ml/adapters"
	_ "github.com/gomlx/gomlx/pkg/ml/adapters/adalora"
	_ "github.com/gomlx/gomlx/pkg/ml/adapters/dora"
	"github.com/gomlx/gomlx/pkg/ml/adapters/lora"
	_ "github.com/gomlx/gomlx/pkg/ml/adapters/qlora"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGetAdapterModel(t *testing.T) {
	ctx := context.New()
	config := lora.NewConfig().SetRank(4)

	manager, err := adapters.GetAdapterModel(ctx, config, "test_lora")
	require.NoError(t, err)
	require.NotNil(t, manager)

	// Check adapter was added
	assert.Equal(t, 1, manager.Count())
	assert.Equal(t, "test_lora", manager.ActiveName())

	// Check manager is stored in context
	fromCtx := adapters.GetFromContext(ctx)
	assert.Same(t, manager, fromCtx)
}

func TestGetAdapterModelValidation(t *testing.T) {
	ctx := context.New()

	// Nil config
	_, err := adapters.GetAdapterModel(ctx, nil, "test")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "cannot be nil")

	// Empty name
	config := lora.NewConfig()
	_, err = adapters.GetAdapterModel(ctx, config, "")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "cannot be empty")
}

func TestAddAdapter(t *testing.T) {
	ctx := context.New()
	config1 := lora.NewConfig().SetRank(4)
	config2 := lora.NewConfig().SetRank(8)

	// Add first adapter
	manager, err := adapters.AddAdapter(ctx, config1, "lora_1")
	require.NoError(t, err)
	assert.Equal(t, 1, manager.Count())
	assert.Equal(t, "", manager.ActiveName()) // Not activated

	// Add second adapter
	manager, err = adapters.AddAdapter(ctx, config2, "lora_2")
	require.NoError(t, err)
	assert.Equal(t, 2, manager.Count())

	// Can activate manually
	err = manager.SetActive("lora_2")
	require.NoError(t, err)
	assert.Equal(t, "lora_2", manager.ActiveName())
}

func TestSwitchAdapter(t *testing.T) {
	ctx := context.New()
	config := lora.NewConfig().SetRank(4)

	// Add two adapters
	manager, _ := adapters.GetAdapterModel(ctx, config, "lora_a")
	_, _ = adapters.AddAdapter(ctx, config, "lora_b")

	assert.Equal(t, "lora_a", manager.ActiveName())

	// Switch to lora_b
	err := adapters.SwitchAdapter(ctx, "lora_b")
	require.NoError(t, err)
	assert.Equal(t, "lora_b", manager.ActiveName())

	// Switch back
	err = adapters.SwitchAdapter(ctx, "lora_a")
	require.NoError(t, err)
	assert.Equal(t, "lora_a", manager.ActiveName())

	// Switch to non-existent
	err = adapters.SwitchAdapter(ctx, "nonexistent")
	assert.Error(t, err)
}

func TestDisableEnableAdapter(t *testing.T) {
	ctx := context.New()
	config := lora.NewConfig().SetRank(4)

	_, err := adapters.GetAdapterModel(ctx, config, "test")
	require.NoError(t, err)

	// Disable
	err = adapters.DisableAdapter(ctx)
	require.NoError(t, err)

	manager := adapters.GetFromContext(ctx)
	adapter := manager.Active()
	assert.False(t, adapter.Config().Enabled())

	// Enable
	err = adapters.EnableAdapter(ctx)
	require.NoError(t, err)
	assert.True(t, adapter.Config().Enabled())
}

func TestDeactivateAdapter(t *testing.T) {
	ctx := context.New()
	config := lora.NewConfig().SetRank(4)

	manager, _ := adapters.GetAdapterModel(ctx, config, "test")
	assert.Equal(t, "test", manager.ActiveName())

	// Deactivate
	err := adapters.DeactivateAdapter(ctx)
	require.NoError(t, err)
	assert.Equal(t, "", manager.ActiveName())

	// Adapter still exists
	assert.Equal(t, 1, manager.Count())
}

func TestListAdapters(t *testing.T) {
	ctx := context.New()
	config := lora.NewConfig()

	// No adapters initially
	list := adapters.ListAdapters(ctx)
	assert.Nil(t, list)

	// Add adapters
	adapters.AddAdapter(ctx, config, "adapter_a")
	adapters.AddAdapter(ctx, config, "adapter_b")
	adapters.AddAdapter(ctx, config, "adapter_c")

	list = adapters.ListAdapters(ctx)
	assert.Len(t, list, 3)
	assert.Contains(t, list, "adapter_a")
	assert.Contains(t, list, "adapter_b")
	assert.Contains(t, list, "adapter_c")
}

func TestGetActiveAdapterName(t *testing.T) {
	ctx := context.New()

	// No manager
	name := adapters.GetActiveAdapterName(ctx)
	assert.Equal(t, "", name)

	// With manager but no active
	config := lora.NewConfig()
	adapters.AddAdapter(ctx, config, "test")
	name = adapters.GetActiveAdapterName(ctx)
	assert.Equal(t, "", name)

	// With active adapter
	adapters.SwitchAdapter(ctx, "test")
	name = adapters.GetActiveAdapterName(ctx)
	assert.Equal(t, "test", name)
}

func TestAllAdapterTypesRegistered(t *testing.T) {
	// Verify all adapter types have factories registered
	adapterTypes := []adapters.AdapterType{
		adapters.AdapterTypeLoRA,
		adapters.AdapterTypeQLoRA,
		adapters.AdapterTypeDoRA,
		adapters.AdapterTypeAdaLoRA,
	}

	for _, adapterType := range adapterTypes {
		factory, ok := adapters.GetFactory(adapterType)
		assert.True(t, ok, "Factory for %s should be registered", adapterType)
		assert.NotNil(t, factory)
		assert.Equal(t, adapterType, factory.Type())
	}
}
