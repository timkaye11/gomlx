// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package adapters

import (
	"regexp"
	"strings"
)

// ModuleMatcher determines which modules/layers should have adapters applied.
// It supports both literal string matching and regex patterns.
type ModuleMatcher struct {
	// Patterns to match (can be literal strings or regex patterns)
	patterns []string

	// Compiled regex patterns (nil if pattern is literal)
	regexes []*regexp.Regexp

	// If true, patterns are treated as regex; if false, as literal substrings
	useRegex bool
}

// NewModuleMatcher creates a new module matcher from the given patterns.
// If useRegex is true, patterns are compiled as regular expressions.
// If useRegex is false, patterns are treated as literal substrings to match.
//
// Example patterns (regex mode):
//   - ".*attention.*" - matches any module containing "attention"
//   - "model\\.layers\\.\\d+\\.mlp" - matches MLP layers in numbered model layers
//   - "^encoder\\.layer\\.[0-9]+\\.(query|key|value)$" - matches Q/K/V in encoder
//
// Example patterns (literal mode):
//   - "attention" - matches modules containing "attention"
//   - "query" - matches modules containing "query"
func NewModuleMatcher(patterns []string, useRegex bool) (*ModuleMatcher, error) {
	m := &ModuleMatcher{
		patterns: patterns,
		useRegex: useRegex,
	}

	if useRegex {
		m.regexes = make([]*regexp.Regexp, len(patterns))
		for i, pattern := range patterns {
			re, err := regexp.Compile(pattern)
			if err != nil {
				return nil, err
			}
			m.regexes[i] = re
		}
	}

	return m, nil
}

// MustNewModuleMatcher creates a new module matcher, panicking on error.
func MustNewModuleMatcher(patterns []string, useRegex bool) *ModuleMatcher {
	m, err := NewModuleMatcher(patterns, useRegex)
	if err != nil {
		panic(err)
	}
	return m
}

// Matches returns true if the given module name matches any of the patterns.
// If no patterns are set (empty list), returns true for all modules.
func (m *ModuleMatcher) Matches(moduleName string) bool {
	// No patterns = match all
	if len(m.patterns) == 0 {
		return true
	}

	if m.useRegex {
		for _, re := range m.regexes {
			if re.MatchString(moduleName) {
				return true
			}
		}
	} else {
		for _, pattern := range m.patterns {
			if strings.Contains(moduleName, pattern) {
				return true
			}
		}
	}

	return false
}

// MatchesAny returns true if the module name matches any pattern in the list.
// This is a convenience function that doesn't require creating a ModuleMatcher.
// Patterns are treated as literal substrings (not regex).
func MatchesAny(moduleName string, patterns []string) bool {
	if len(patterns) == 0 {
		return true
	}
	for _, pattern := range patterns {
		if strings.Contains(moduleName, pattern) {
			return true
		}
	}
	return false
}

// MatchesAnyRegex returns true if the module name matches any regex pattern.
// Returns an error if any pattern fails to compile.
func MatchesAnyRegex(moduleName string, patterns []string) (bool, error) {
	if len(patterns) == 0 {
		return true, nil
	}
	for _, pattern := range patterns {
		re, err := regexp.Compile(pattern)
		if err != nil {
			return false, err
		}
		if re.MatchString(moduleName) {
			return true, nil
		}
	}
	return false, nil
}

// CommonTargetPatterns provides commonly used target module patterns.
var CommonTargetPatterns = struct {
	// Attention targets all attention-related modules
	Attention []string

	// QKV targets query, key, value projections
	QKV []string

	// QV targets query and value projections (common LoRA target)
	QV []string

	// MLP targets feed-forward/MLP layers
	MLP []string

	// All targets all linear/dense layers
	All []string
}{
	Attention: []string{"attention", "attn", "self_attn"},
	QKV:       []string{"query", "key", "value", "q_proj", "k_proj", "v_proj"},
	QV:        []string{"query", "value", "q_proj", "v_proj"},
	MLP:       []string{"mlp", "ffn", "feed_forward", "fc1", "fc2", "dense"},
	All:       []string{}, // Empty means match all
}

// ShouldApplyAdapter checks if an adapter should be applied to the given module.
// Uses the target modules from the config, treating them as literal patterns.
func ShouldApplyAdapter(config AdapterConfig, moduleName string) bool {
	if config == nil {
		return true
	}
	return MatchesAny(moduleName, config.TargetModules())
}

// ShouldApplyAdapterRegex checks if an adapter should be applied using regex patterns.
// Returns an error if any pattern fails to compile.
func ShouldApplyAdapterRegex(config AdapterConfig, moduleName string) (bool, error) {
	if config == nil {
		return true, nil
	}
	return MatchesAnyRegex(moduleName, config.TargetModules())
}

// FilterModules returns only the module names that match the given patterns.
func FilterModules(moduleNames []string, patterns []string, useRegex bool) ([]string, error) {
	matcher, err := NewModuleMatcher(patterns, useRegex)
	if err != nil {
		return nil, err
	}

	var result []string
	for _, name := range moduleNames {
		if matcher.Matches(name) {
			result = append(result, name)
		}
	}
	return result, nil
}

// TargetModulePresets provides preset configurations for common model architectures.
type TargetModulePresets struct {
	// Name of the preset (e.g., "llama", "gpt2", "bert")
	Name string

	// DefaultTargets are the recommended modules to target
	DefaultTargets []string

	// AllLinear targets all linear layers in the model
	AllLinear []string
}

// GetPreset returns a preset for common model architectures.
// Returns nil if the architecture is not recognized.
func GetPreset(architecture string) *TargetModulePresets {
	presets := map[string]*TargetModulePresets{
		"llama": {
			Name:           "llama",
			DefaultTargets: []string{"q_proj", "v_proj"},
			AllLinear:      []string{"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"},
		},
		"gpt2": {
			Name:           "gpt2",
			DefaultTargets: []string{"c_attn"},
			AllLinear:      []string{"c_attn", "c_proj", "c_fc"},
		},
		"bert": {
			Name:           "bert",
			DefaultTargets: []string{"query", "value"},
			AllLinear:      []string{"query", "key", "value", "dense"},
		},
		"t5": {
			Name:           "t5",
			DefaultTargets: []string{"q", "v"},
			AllLinear:      []string{"q", "k", "v", "o", "wi", "wo"},
		},
		"mistral": {
			Name:           "mistral",
			DefaultTargets: []string{"q_proj", "v_proj"},
			AllLinear:      []string{"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"},
		},
	}

	return presets[strings.ToLower(architecture)]
}
