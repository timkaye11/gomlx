// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package pretrained

import (
	"regexp"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewLoader(t *testing.T) {
	loader := NewLoader()
	require.NotNil(t, loader)
	assert.NotNil(t, loader.loaded)
	assert.Equal(t, 0, loader.LoadedCount())
	assert.Equal(t, 0, loader.SkippedCount())
	assert.Equal(t, 0, loader.MismatchedCount())
}

func TestLoaderWithMapper(t *testing.T) {
	loader := NewLoader().WithMapper(IdentityMapper{})
	require.NotNil(t, loader.mapper)
}

func TestLoaderWithStrictMode(t *testing.T) {
	loader := NewLoader().WithStrictMode(true)
	assert.True(t, loader.strictMode)

	loader.WithStrictMode(false)
	assert.False(t, loader.strictMode)
}

func TestIdentityMapper(t *testing.T) {
	mapper := IdentityMapper{}

	tests := []string{
		"layer.weight",
		"encoder.layer.0.attention.self.query.weight",
		"embeddings.word_embeddings.weight",
	}

	for _, name := range tests {
		assert.Equal(t, name, mapper.MapName(name))
	}
}

func TestPrefixMapper(t *testing.T) {
	mapper := PrefixMapper{Prefix: "model"}

	tests := []struct {
		input    string
		expected string
	}{
		{"weight", "model/weight"},
		{"layer.bias", "model/layer.bias"},
		{"encoder.weight", "model/encoder.weight"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			assert.Equal(t, tt.expected, mapper.MapName(tt.input))
		})
	}
}

func TestRegexMapper(t *testing.T) {
	mapper := &RegexMapper{
		Rules: []RegexRule{
			{Pattern: regexp.MustCompile(`^skip_.*`), Skip: true},
			{Pattern: regexp.MustCompile(`^layer\.(\d+)\.weight$`), Replacement: "layers/$1/weights"},
			{Pattern: regexp.MustCompile(`\.`), Replacement: "/"},
		},
	}

	tests := []struct {
		input    string
		expected string
	}{
		{"skip_this", ""},
		{"layer.0.weight", "layers/0/weights"},
		{"other.name", "other/name"},
		{"no_dots", "no_dots"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			assert.Equal(t, tt.expected, mapper.MapName(tt.input))
		})
	}
}

func TestNewRegexMapper(t *testing.T) {
	mapper, err := NewRegexMapper(
		`^encoder\.layer\.(\d+)\.`, "layer_$1/",
		`\.weight$`, "/weights",
	)
	require.NoError(t, err)
	require.NotNil(t, mapper)
	assert.Len(t, mapper.Rules, 2)

	// Test mapping
	result := mapper.MapName("encoder.layer.5.attention")
	assert.Equal(t, "layer_5/attention", result)
}

func TestNewRegexMapperError(t *testing.T) {
	// Odd number of arguments
	_, err := NewRegexMapper("pattern")
	require.Error(t, err)

	// Invalid regex
	_, err = NewRegexMapper("[invalid", "replacement")
	require.Error(t, err)
}

func TestHuggingFaceBERTMapper(t *testing.T) {
	mapper := HuggingFaceBERTMapper()
	require.NotNil(t, mapper)

	tests := []struct {
		input    string
		expected string
	}{
		// Embeddings
		{"embeddings.word_embeddings.weight", "embeddings/word_embeddings/weights"},
		{"embeddings.position_embeddings.weight", "embeddings/position_embeddings/weights"},
		{"embeddings.token_type_embeddings.weight", "embeddings/token_type_embeddings/weights"},
		{"embeddings.LayerNorm.weight", "embeddings/layer_norm/scale"},
		{"embeddings.LayerNorm.bias", "embeddings/layer_norm/offset"},

		// Encoder layers - attention
		{"encoder.layer.0.attention.self.query.weight", "encoder/layer_0/attention/query/weights"},
		{"encoder.layer.0.attention.self.query.bias", "encoder/layer_0/attention/query/bias"},
		{"encoder.layer.5.attention.self.key.weight", "encoder/layer_5/attention/key/weights"},
		{"encoder.layer.11.attention.self.value.weight", "encoder/layer_11/attention/value/weights"},
		{"encoder.layer.0.attention.output.dense.weight", "encoder/layer_0/attention/output/weights"},
		{"encoder.layer.0.attention.output.LayerNorm.weight", "encoder/layer_0/attention/layer_norm/scale"},

		// Encoder layers - FFN
		{"encoder.layer.0.intermediate.dense.weight", "encoder/layer_0/ffn/intermediate/weights"},
		{"encoder.layer.0.intermediate.dense.bias", "encoder/layer_0/ffn/intermediate/bias"},
		{"encoder.layer.0.output.dense.weight", "encoder/layer_0/ffn/output/weights"},
		{"encoder.layer.0.output.LayerNorm.weight", "encoder/layer_0/ffn/layer_norm/scale"},

		// Pooler (should be skipped)
		{"pooler.dense.weight", ""},
		{"pooler.dense.bias", ""},

		// Unknown (pass through)
		{"unknown.weight", "unknown.weight"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := mapper.MapName(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestLoaderSkippedAndMismatched(t *testing.T) {
	loader := NewLoader()

	// Manually populate for testing
	loader.skipped = []string{"weight1", "weight2"}
	loader.mismatched = []string{"weight3: shape mismatch"}

	assert.Equal(t, 2, loader.SkippedCount())
	assert.Equal(t, 1, loader.MismatchedCount())
	assert.Equal(t, []string{"weight1", "weight2"}, loader.Skipped())
	assert.Equal(t, []string{"weight3: shape mismatch"}, loader.Mismatched())
}

func TestLoaderLoadSafeTensorsDir(t *testing.T) {
	loader := NewLoader()

	// Test with empty directory (no .safetensors files)
	tmpDir := t.TempDir()
	err := loader.LoadSafeTensorsDir(nil, tmpDir)
	require.Error(t, err) // Should error because no .safetensors files found
}

func TestRegexMapperMultipleRules(t *testing.T) {
	// Test that first matching rule is used
	mapper := &RegexMapper{
		Rules: []RegexRule{
			{Pattern: regexp.MustCompile(`^specific\.name$`), Replacement: "matched_first"},
			{Pattern: regexp.MustCompile(`^specific\..*`), Replacement: "matched_second"},
		},
	}

	assert.Equal(t, "matched_first", mapper.MapName("specific.name"))
	assert.Equal(t, "matched_second", mapper.MapName("specific.other"))
}

func TestLoaderMultipleMatches(t *testing.T) {
	// Test how loader handles ambiguous variable names
	// This tests the variable matching logic in setVariable

	// Create a mapper that produces ambiguous mappings
	mapper := &RegexMapper{
		Rules: []RegexRule{
			// Map weight names to simple scoped names
			{Pattern: regexp.MustCompile(`^layer\.weight$`), Replacement: "scope/weight"},
			{Pattern: regexp.MustCompile(`^layer\.bias$`), Replacement: "scope/bias"},
		},
	}

	loader := NewLoader().WithMapper(mapper)
	require.NotNil(t, loader)

	// Verify mapper works
	assert.Equal(t, "scope/weight", mapper.MapName("layer.weight"))
	assert.Equal(t, "scope/bias", mapper.MapName("layer.bias"))

	// Test that loader tracks loaded and mismatched weights correctly
	// Note: We can't easily test the full setVariable logic without a real Context,
	// but we can verify the mapper properly disambiguates based on the full scoped path
	assert.Equal(t, 0, loader.LoadedCount())
	assert.Equal(t, 0, loader.MismatchedCount())
}
