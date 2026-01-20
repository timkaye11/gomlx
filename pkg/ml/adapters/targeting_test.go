// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package adapters_test

import (
	"testing"

	"github.com/gomlx/gomlx/pkg/ml/adapters"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestModuleMatcherLiteral(t *testing.T) {
	// Test literal (substring) matching
	matcher, err := adapters.NewModuleMatcher([]string{"query", "value"}, false)
	require.NoError(t, err)

	// Should match modules containing "query" or "value"
	assert.True(t, matcher.Matches("model/layer0/query"))
	assert.True(t, matcher.Matches("model/layer0/value"))
	assert.True(t, matcher.Matches("encoder.attention.query_proj"))
	assert.True(t, matcher.Matches("decoder.self_attn.v_proj_value"))

	// Should not match
	assert.False(t, matcher.Matches("model/layer0/key"))
	assert.False(t, matcher.Matches("model/layer0/output"))
	assert.False(t, matcher.Matches("embedding"))
}

func TestModuleMatcherRegex(t *testing.T) {
	// Test regex matching
	matcher, err := adapters.NewModuleMatcher([]string{
		`.*attention.*`,
		`model\.layers\.\d+\.mlp`,
	}, true)
	require.NoError(t, err)

	// Should match attention-related modules
	assert.True(t, matcher.Matches("model/layer0/attention"))
	assert.True(t, matcher.Matches("encoder.self_attention.query"))
	assert.True(t, matcher.Matches("attention_output"))

	// Should match MLP layers
	assert.True(t, matcher.Matches("model.layers.0.mlp"))
	assert.True(t, matcher.Matches("model.layers.12.mlp"))

	// Should not match
	assert.False(t, matcher.Matches("model/layer0/embedding"))
	assert.False(t, matcher.Matches("model.layers.mlp")) // missing layer number
}

func TestModuleMatcherInvalidRegex(t *testing.T) {
	// Invalid regex should return error
	_, err := adapters.NewModuleMatcher([]string{"[invalid"}, true)
	require.Error(t, err)
}

func TestModuleMatcherEmptyPatterns(t *testing.T) {
	// Empty patterns should match everything
	matcher, err := adapters.NewModuleMatcher([]string{}, false)
	require.NoError(t, err)

	assert.True(t, matcher.Matches("any_module"))
	assert.True(t, matcher.Matches("model/layer0/query"))
	assert.True(t, matcher.Matches(""))
}

func TestMustNewModuleMatcher(t *testing.T) {
	// Valid patterns should not panic
	assert.NotPanics(t, func() {
		adapters.MustNewModuleMatcher([]string{"query"}, false)
	})

	// Invalid regex should panic
	assert.Panics(t, func() {
		adapters.MustNewModuleMatcher([]string{"[invalid"}, true)
	})
}

func TestMatchesAny(t *testing.T) {
	patterns := []string{"query", "value", "key"}

	assert.True(t, adapters.MatchesAny("layer/query_proj", patterns))
	assert.True(t, adapters.MatchesAny("layer/value_proj", patterns))
	assert.True(t, adapters.MatchesAny("layer/key_proj", patterns))
	assert.False(t, adapters.MatchesAny("layer/output_proj", patterns))

	// Empty patterns matches everything
	assert.True(t, adapters.MatchesAny("anything", []string{}))
}

func TestMatchesAnyRegex(t *testing.T) {
	patterns := []string{`^encoder\.layer\.\d+\.query$`, `^encoder\.layer\.\d+\.value$`}

	match, err := adapters.MatchesAnyRegex("encoder.layer.0.query", patterns)
	require.NoError(t, err)
	assert.True(t, match)

	match, err = adapters.MatchesAnyRegex("encoder.layer.5.value", patterns)
	require.NoError(t, err)
	assert.True(t, match)

	match, err = adapters.MatchesAnyRegex("encoder.layer.0.key", patterns)
	require.NoError(t, err)
	assert.False(t, match)

	match, err = adapters.MatchesAnyRegex("decoder.layer.0.query", patterns)
	require.NoError(t, err)
	assert.False(t, match)

	// Empty patterns matches everything
	match, err = adapters.MatchesAnyRegex("anything", []string{})
	require.NoError(t, err)
	assert.True(t, match)

	// Invalid regex should return error
	_, err = adapters.MatchesAnyRegex("test", []string{"[invalid"})
	require.Error(t, err)
}

func TestCommonTargetPatterns(t *testing.T) {
	// Test that common patterns exist and are reasonable
	assert.NotEmpty(t, adapters.CommonTargetPatterns.Attention)
	assert.NotEmpty(t, adapters.CommonTargetPatterns.QKV)
	assert.NotEmpty(t, adapters.CommonTargetPatterns.QV)
	assert.NotEmpty(t, adapters.CommonTargetPatterns.MLP)
	assert.Empty(t, adapters.CommonTargetPatterns.All) // All is empty (matches everything)

	// Test Attention patterns
	for _, pattern := range adapters.CommonTargetPatterns.Attention {
		assert.True(t, adapters.MatchesAny("layer/"+pattern+"_proj", adapters.CommonTargetPatterns.Attention))
	}

	// Test QKV patterns
	qkvModules := []string{"query", "key", "value", "q_proj", "k_proj", "v_proj"}
	for _, mod := range qkvModules {
		assert.True(t, adapters.MatchesAny("layer/"+mod, adapters.CommonTargetPatterns.QKV),
			"QKV should match %s", mod)
	}

	// Test QV patterns (common LoRA targets)
	qvModules := []string{"query", "value", "q_proj", "v_proj"}
	for _, mod := range qvModules {
		assert.True(t, adapters.MatchesAny("layer/"+mod, adapters.CommonTargetPatterns.QV),
			"QV should match %s", mod)
	}
	// QV should not match key
	assert.False(t, adapters.MatchesAny("layer/key", adapters.CommonTargetPatterns.QV))
	assert.False(t, adapters.MatchesAny("layer/k_proj", adapters.CommonTargetPatterns.QV))
}

func TestFilterModules(t *testing.T) {
	moduleNames := []string{
		"model/layer0/query",
		"model/layer0/key",
		"model/layer0/value",
		"model/layer0/output",
		"model/layer1/query",
		"model/layer1/key",
		"model/layer1/value",
		"model/embeddings",
	}

	// Filter with literal patterns
	filtered, err := adapters.FilterModules(moduleNames, []string{"query", "value"}, false)
	require.NoError(t, err)
	assert.Equal(t, []string{
		"model/layer0/query",
		"model/layer0/value",
		"model/layer1/query",
		"model/layer1/value",
	}, filtered)

	// Filter with regex patterns
	filtered, err = adapters.FilterModules(moduleNames, []string{`layer\d+/query`}, true)
	require.NoError(t, err)
	assert.Equal(t, []string{
		"model/layer0/query",
		"model/layer1/query",
	}, filtered)

	// Empty patterns returns all
	filtered, err = adapters.FilterModules(moduleNames, []string{}, false)
	require.NoError(t, err)
	assert.Equal(t, moduleNames, filtered)

	// Invalid regex returns error
	_, err = adapters.FilterModules(moduleNames, []string{"[invalid"}, true)
	require.Error(t, err)
}

func TestGetPreset(t *testing.T) {
	// Test known architectures
	llama := adapters.GetPreset("llama")
	require.NotNil(t, llama)
	assert.Equal(t, "llama", llama.Name)
	assert.Contains(t, llama.DefaultTargets, "q_proj")
	assert.Contains(t, llama.DefaultTargets, "v_proj")
	assert.Contains(t, llama.AllLinear, "gate_proj")

	gpt2 := adapters.GetPreset("gpt2")
	require.NotNil(t, gpt2)
	assert.Equal(t, "gpt2", gpt2.Name)
	assert.Contains(t, gpt2.DefaultTargets, "c_attn")

	bert := adapters.GetPreset("bert")
	require.NotNil(t, bert)
	assert.Contains(t, bert.DefaultTargets, "query")
	assert.Contains(t, bert.DefaultTargets, "value")

	t5 := adapters.GetPreset("t5")
	require.NotNil(t, t5)
	assert.Contains(t, t5.DefaultTargets, "q")
	assert.Contains(t, t5.DefaultTargets, "v")

	mistral := adapters.GetPreset("mistral")
	require.NotNil(t, mistral)
	assert.Contains(t, mistral.DefaultTargets, "q_proj")

	// Test case insensitivity
	llamaUpper := adapters.GetPreset("LLAMA")
	require.NotNil(t, llamaUpper)
	assert.Equal(t, llama.Name, llamaUpper.Name)

	// Test unknown architecture
	unknown := adapters.GetPreset("unknown_model")
	assert.Nil(t, unknown)
}

func TestShouldApplyAdapter(t *testing.T) {
	// Create a mock config that implements AdapterConfig
	// Note: We test with nil config (should return true for all)
	assert.True(t, adapters.ShouldApplyAdapter(nil, "any_module"))
}
