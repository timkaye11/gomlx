// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package lora

import (
	"testing"

	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewConfig(t *testing.T) {
	config := NewConfig()

	assert.Equal(t, 8, config.Rank)
	assert.Equal(t, 16.0, config.Alpha)
	assert.Equal(t, 0.0, config.Dropout)
	assert.True(t, config.Enabled)
	assert.Nil(t, config.TargetModules)
}

func TestConfigSetters(t *testing.T) {
	config := NewConfig().
		SetRank(16).
		SetAlpha(32.0).
		SetDropout(0.1).
		SetEnabled(false).
		SetTargetModules("query", "key", "value")

	assert.Equal(t, 16, config.Rank)
	assert.Equal(t, 32.0, config.Alpha)
	assert.Equal(t, 0.1, config.Dropout)
	assert.False(t, config.Enabled)
	assert.Equal(t, []string{"query", "key", "value"}, config.TargetModules)
}

func TestConfigScale(t *testing.T) {
	config := NewConfig().SetRank(8).SetAlpha(16.0)
	assert.Equal(t, 2.0, config.Scale())

	config.SetRank(4).SetAlpha(16.0)
	assert.Equal(t, 4.0, config.Scale())

	config.SetRank(16).SetAlpha(16.0)
	assert.Equal(t, 1.0, config.Scale())

	// Test zero rank returns zero scale
	config.Rank = 0
	assert.Equal(t, 0.0, config.Scale())
}

func TestConfigSetRankPanicsOnInvalid(t *testing.T) {
	assert.Panics(t, func() {
		NewConfig().SetRank(0)
	})
	assert.Panics(t, func() {
		NewConfig().SetRank(-1)
	})
}

func TestConfigSetDropoutPanicsOnInvalid(t *testing.T) {
	assert.Panics(t, func() {
		NewConfig().SetDropout(-0.1)
	})
	assert.Panics(t, func() {
		NewConfig().SetDropout(1.0)
	})
	assert.Panics(t, func() {
		NewConfig().SetDropout(1.5)
	})
}

func TestFromContext(t *testing.T) {
	ctx := context.New()
	ctx.SetParam(ParamRank, 4)
	ctx.SetParam(ParamAlpha, 8.0)
	ctx.SetParam(ParamDropout, 0.05)
	ctx.SetParam(ParamEnabled, false)

	config := FromContext(ctx)

	assert.Equal(t, 4, config.Rank)
	assert.Equal(t, 8.0, config.Alpha)
	assert.Equal(t, 0.05, config.Dropout)
	assert.False(t, config.Enabled)
}

func TestFromContextDefaults(t *testing.T) {
	ctx := context.New()

	config := FromContext(ctx)

	assert.Equal(t, 8, config.Rank)
	assert.Equal(t, 16.0, config.Alpha)
	assert.Equal(t, 0.0, config.Dropout)
	assert.True(t, config.Enabled)
}

func TestIsLoRAParameter(t *testing.T) {
	tests := []struct {
		name     string
		expected bool
	}{
		{"/model/layer_0/lora_A", true},
		{"/model/layer_0/lora_B", true},
		{"/model/lora_dense/lora_A", true},
		{"/model/lora_dense/lora_B", true},
		{"/model/layer_0/weights", false},
		{"/model/layer_0/bias", false},
		{"/model/dense/weights", false},
		{"lora_A", true},
		{"lora_B", true},
		{"lora", false},
		{"", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.expected, isLoRAParameter(tt.name))
		})
	}
}

func TestFreezeBaseWeights(t *testing.T) {
	ctx := context.New()

	// Create some mock variables by setting up the context
	// We'll use the variable iteration to verify the behavior
	// Note: This test verifies the function doesn't panic
	// Full integration testing would require a backend

	FreezeBaseWeights(ctx)
	// Should not panic with empty context
}

func TestUnfreezeBaseWeights(t *testing.T) {
	ctx := context.New()

	UnfreezeBaseWeights(ctx)
	// Should not panic with empty context
}

func TestCountLoRAParameters(t *testing.T) {
	ctx := context.New()

	count := CountLoRAParameters(ctx)
	assert.Equal(t, int64(0), count) // Empty context has no variables
}

func TestCountBaseParameters(t *testing.T) {
	ctx := context.New()

	count := CountBaseParameters(ctx)
	assert.Equal(t, int64(0), count) // Empty context has no variables
}

func TestMergeLoRAWeights(t *testing.T) {
	ctx := context.New()
	config := NewConfig()

	err := MergeLoRAWeights(ctx, config)
	require.NoError(t, err)
}

func TestConfigChaining(t *testing.T) {
	// Test that all setters return *Config for chaining
	config := NewConfig()

	result := config.
		SetRank(4).
		SetAlpha(8.0).
		SetDropout(0.1).
		SetEnabled(true).
		SetTargetModules("dense")

	assert.Same(t, config, result)
}

func TestContextParamNames(t *testing.T) {
	// Verify param name constants are set correctly
	assert.Equal(t, "lora_rank", ParamRank)
	assert.Equal(t, "lora_alpha", ParamAlpha)
	assert.Equal(t, "lora_dropout", ParamDropout)
	assert.Equal(t, "lora_enabled", ParamEnabled)
}
