// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package finetune

import (
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	_ "github.com/gomlx/gomlx/backends/default"
)

// createTestContext creates a context with variables at various scopes for testing
func createTestContext(t *testing.T) *context.Context {
	ctx := context.New()
	backend := graphtest.BuildTestBackend()

	// Create variables at different scopes
	// /encoder/layer_0/weights
	// /encoder/layer_0/bias
	// /encoder/layer_1/weights
	// /encoder/layer_1/bias
	// /decoder/layer_0/weights
	// /head/dense/weights
	// /head/dense/bias
	// /embeddings/token_embed

	ctx.In("encoder").In("layer_0").VariableWithShape("weights", shapes.Make(dtypes.Float32, 768, 768))
	ctx.In("encoder").In("layer_0").VariableWithShape("bias", shapes.Make(dtypes.Float32, 768))
	ctx.In("encoder").In("layer_1").VariableWithShape("weights", shapes.Make(dtypes.Float32, 768, 768))
	ctx.In("encoder").In("layer_1").VariableWithShape("bias", shapes.Make(dtypes.Float32, 768))
	ctx.In("decoder").In("layer_0").VariableWithShape("weights", shapes.Make(dtypes.Float32, 768, 768))
	ctx.In("head").In("dense").VariableWithShape("weights", shapes.Make(dtypes.Float32, 768, 10))
	ctx.In("head").In("dense").VariableWithShape("bias", shapes.Make(dtypes.Float32, 10))
	ctx.In("embeddings").VariableWithShape("token_embed", shapes.Make(dtypes.Float32, 30000, 768))

	ctx.InitializeVariables(backend, nil)
	return ctx
}

func TestFreezeAll(t *testing.T) {
	ctx := createTestContext(t)

	// All variables should be trainable by default
	require.Equal(t, 8, CountTrainable(ctx))
	require.Equal(t, 0, CountFrozen(ctx))

	// Freeze all
	n := FreezeAll(ctx)
	assert.Equal(t, 8, n)
	assert.Equal(t, 0, CountTrainable(ctx))
	assert.Equal(t, 8, CountFrozen(ctx))
}

func TestUnfreezeAll(t *testing.T) {
	ctx := createTestContext(t)

	// Freeze all first
	FreezeAll(ctx)
	require.Equal(t, 0, CountTrainable(ctx))

	// Unfreeze all
	n := UnfreezeAll(ctx)
	assert.Equal(t, 8, n)
	assert.Equal(t, 8, CountTrainable(ctx))
	assert.Equal(t, 0, CountFrozen(ctx))
}

func TestFreezeByPattern_DoubleGlob(t *testing.T) {
	ctx := createTestContext(t)

	// Freeze encoder using "**" pattern
	n := FreezeByPattern(ctx, "**/encoder/**")
	assert.Equal(t, 4, n) // 4 encoder variables
	assert.Equal(t, 4, CountFrozen(ctx))
	assert.Equal(t, 4, CountTrainable(ctx))

	// Verify only encoder variables are frozen
	frozen := ListFrozenVariables(ctx)
	for _, path := range frozen {
		assert.Contains(t, path, "encoder")
	}
}

func TestFreezeByPattern_ScopeGlob(t *testing.T) {
	ctx := createTestContext(t)

	// Freeze head scope using double glob (head is first segment in /head/dense/...)
	n := FreezeByPattern(ctx, "**/head/**")
	assert.Equal(t, 2, n) // 2 head variables
	assert.Equal(t, 2, CountFrozen(ctx))
}

func TestFreezeScope(t *testing.T) {
	ctx := createTestContext(t)

	// Freeze encoder scope
	n := FreezeScope(ctx, "/encoder")
	assert.Equal(t, 4, n)

	frozen := ListFrozenVariables(ctx)
	assert.Equal(t, 4, len(frozen))
	for _, path := range frozen {
		assert.Contains(t, path, "encoder")
	}
}

func TestUnfreezeScope(t *testing.T) {
	ctx := createTestContext(t)

	// Freeze all first
	FreezeAll(ctx)
	require.Equal(t, 0, CountTrainable(ctx))

	// Unfreeze only head
	n := UnfreezeScope(ctx, "/head")
	assert.Equal(t, 2, n)
	assert.Equal(t, 2, CountTrainable(ctx))

	trainable := ListTrainableVariables(ctx)
	for _, path := range trainable {
		assert.Contains(t, path, "head")
	}
}

func TestFreezeByPattern_LayerRange(t *testing.T) {
	ctx := createTestContext(t)

	// Freeze only layer_0 (using double glob to match /encoder/layer_0/* and /decoder/layer_0/*)
	n := FreezeByPattern(ctx, "**/layer_0/**")
	// Should match /encoder/layer_0/* and /decoder/layer_0/*
	assert.Equal(t, 3, n) // 2 encoder + 1 decoder
}

func TestListVariables(t *testing.T) {
	ctx := createTestContext(t)

	paths := ListVariables(ctx)
	assert.Equal(t, 8, len(paths))

	// Check that expected paths exist
	pathSet := make(map[string]bool)
	for _, p := range paths {
		pathSet[p] = true
	}

	assert.True(t, pathSet["/encoder/layer_0/weights"])
	assert.True(t, pathSet["/head/dense/bias"])
	assert.True(t, pathSet["/embeddings/token_embed"])
}

func TestSetDiscriminativeLR(t *testing.T) {
	ctx := createTestContext(t)

	SetDiscriminativeLR(ctx, []LayerGroup{
		{Pattern: "**/encoder/**", LRMultiplier: 0.1},
		{Pattern: "**/head/**", LRMultiplier: 1.0},
		{Pattern: "**/embeddings/**", LRMultiplier: 0.01},
	})

	// Test getting multipliers for different variable paths
	assert.Equal(t, 0.1, GetLRMultiplier(ctx, "/encoder/layer_0/weights"))
	assert.Equal(t, 0.1, GetLRMultiplier(ctx, "/encoder/layer_1/bias"))
	assert.Equal(t, 1.0, GetLRMultiplier(ctx, "/head/dense/weights"))
	assert.Equal(t, 0.01, GetLRMultiplier(ctx, "/embeddings/token_embed"))
	assert.Equal(t, 1.0, GetLRMultiplier(ctx, "/decoder/layer_0/weights")) // No match, default to 1.0
}

func TestGetLRMultiplier_NoConfig(t *testing.T) {
	ctx := createTestContext(t)

	// Without SetDiscriminativeLR, should always return 1.0
	assert.Equal(t, 1.0, GetLRMultiplier(ctx, "/encoder/layer_0/weights"))
	assert.Equal(t, 1.0, GetLRMultiplier(ctx, "/head/dense/weights"))
}

func TestMatchPattern(t *testing.T) {
	tests := []struct {
		pattern  string
		path     string
		expected bool
	}{
		// Double glob tests - "**" matches zero or more segments
		{"**", "/encoder/layer_0/weights", true},
		{"**/encoder/**", "/encoder/layer_0/weights", true},
		{"**/encoder/**", "/decoder/layer_0/weights", false},
		{"**/layer_0/**", "/encoder/layer_0/weights", true},
		{"**/layer_0/**", "/decoder/layer_0/weights", true},
		{"**/weights", "/encoder/layer_0/weights", true},
		{"**/weights", "/encoder/layer_0/bias", false},
		{"**/head/**", "/head/dense/weights", true},
		{"**/dense/**", "/head/dense/weights", true},

		// Single glob tests - "*" matches exactly one segment
		// "*/encoder/*" means something/encoder/something, which is 3 segments
		// "/encoder/layer_0/weights" has encoder as first segment, so this needs encoder/*
		{"encoder/*", "/encoder/layer_0/weights", true},  // 2 segs in pattern, 3 in path (sliding match)
		{"encoder/layer_0/*", "/encoder/layer_0/weights", true},
		{"*/layer_0/*", "/encoder/layer_0/weights", true}, // encoder/layer_0/weights - matches
		{"head/dense/*", "/head/dense/weights", true},
		{"head/*", "/head/dense/weights", true},
		{"*/dense/*", "/head/dense/weights", true},  // head/dense/weights - matches

		// Exact match (full path with leading /)
		{"/encoder/layer_0/weights", "/encoder/layer_0/weights", true},
		{"/encoder/layer_0/weights", "/encoder/layer_0/bias", false},
	}

	for _, tt := range tests {
		t.Run(tt.pattern+"_"+tt.path, func(t *testing.T) {
			result := matchPattern(tt.pattern, tt.path)
			assert.Equal(t, tt.expected, result, "pattern=%q path=%q", tt.pattern, tt.path)
		})
	}
}

func TestFreezeUnfreezeWorkflow(t *testing.T) {
	// Simulate a typical fine-tuning workflow:
	// 1. Freeze all layers
	// 2. Unfreeze only the head
	// 3. Train for a while
	// 4. Unfreeze more layers

	ctx := createTestContext(t)

	// Step 1: Freeze everything
	FreezeAll(ctx)
	assert.Equal(t, 0, CountTrainable(ctx))

	// Step 2: Unfreeze head for initial training
	UnfreezeByPattern(ctx, "**/head/**")
	assert.Equal(t, 2, CountTrainable(ctx))
	trainable := ListTrainableVariables(ctx)
	for _, p := range trainable {
		assert.Contains(t, p, "head")
	}

	// Step 3: Later unfreeze encoder layer 1
	UnfreezeByPattern(ctx, "**/layer_1/**")
	assert.Equal(t, 4, CountTrainable(ctx)) // head(2) + encoder layer_1(2)

	// Step 4: Unfreeze all encoder layers
	UnfreezeByPattern(ctx, "**/encoder/**")
	assert.Equal(t, 6, CountTrainable(ctx)) // head(2) + encoder(4)

	// Step 5: Finally unfreeze everything
	UnfreezeAll(ctx)
	assert.Equal(t, 8, CountTrainable(ctx))
}
