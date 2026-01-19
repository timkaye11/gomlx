// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package finetune provides utilities for fine-tuning pre-trained models.
//
// It includes functions for:
//   - Freezing and unfreezing layers by pattern matching
//   - Discriminative learning rates (different LR for different layer groups)
//   - Gradual unfreezing during training
package finetune

import (
	"path"
	"strings"
	"sync"

	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/train"
)

// FreezeByPattern freezes (sets trainable=false) all variables whose full path
// matches the given glob pattern.
//
// The pattern uses Go's path.Match syntax:
//   - "*" matches any sequence of non-separator characters
//   - "**" is treated as matching any path (including separators)
//   - "?" matches any single non-separator character
//
// Examples:
//
//	FreezeByPattern(ctx, "*/encoder/*")     // Freeze all encoder layers
//	FreezeByPattern(ctx, "*/layer_[0-5]/*") // Freeze layers 0-5
//	FreezeByPattern(ctx, "**/embeddings/*") // Freeze embedding layers
//
// Returns the number of variables that were frozen.
func FreezeByPattern(ctx *context.Context, pattern string) int {
	return setTrainableByPattern(ctx, pattern, false)
}

// UnfreezeByPattern unfreezes (sets trainable=true) all variables whose full path
// matches the given glob pattern.
//
// See FreezeByPattern for pattern syntax details.
//
// Returns the number of variables that were unfrozen.
func UnfreezeByPattern(ctx *context.Context, pattern string) int {
	return setTrainableByPattern(ctx, pattern, true)
}

// FreezeAll freezes all variables in the context.
// Returns the number of variables that were frozen.
func FreezeAll(ctx *context.Context) int {
	return FreezeByPattern(ctx, "**")
}

// UnfreezeAll unfreezes all variables in the context.
// Returns the number of variables that were unfrozen.
func UnfreezeAll(ctx *context.Context) int {
	return UnfreezeByPattern(ctx, "**")
}

// FreezeScope freezes all variables within the given scope.
// The scope should be a path like "/encoder" or "/model/layer_0".
// Returns the number of variables that were frozen.
func FreezeScope(ctx *context.Context, scope string) int {
	// Normalize scope to have leading slash
	if !strings.HasPrefix(scope, "/") {
		scope = "/" + scope
	}
	// Match anything within this scope, including all descendants recursively
	pattern := scope + "/**"
	return FreezeByPattern(ctx, pattern)
}

// UnfreezeScope unfreezes all variables within the given scope.
// Returns the number of variables that were unfrozen.
func UnfreezeScope(ctx *context.Context, scope string) int {
	if !strings.HasPrefix(scope, "/") {
		scope = "/" + scope
	}
	// Match anything within this scope, including all descendants recursively
	pattern := scope + "/**"
	return UnfreezeByPattern(ctx, pattern)
}

// isInternalVariable returns true if the variable is an internal variable (name starts with #).
func isInternalVariable(v *context.Variable) bool {
	return strings.HasPrefix(v.Name(), "#")
}

// setTrainableByPattern sets the trainable flag for all variables matching the pattern.
func setTrainableByPattern(ctx *context.Context, pattern string, trainable bool) int {
	count := 0

	// Handle "**" pattern specially - it matches everything
	matchAll := pattern == "**" || pattern == "**/*"

	for v := range ctx.IterVariables() {
		// Skip internal variables like #rngState
		if isInternalVariable(v) {
			continue
		}

		fullPath := v.Scope() + "/" + v.Name()

		matched := false
		if matchAll {
			matched = true
		} else {
			// Try to match with path.Match
			// Handle "**" in patterns by trying multiple match strategies
			matched = matchPattern(pattern, fullPath)
		}

		if matched {
			v.SetTrainable(trainable)
			count++
		}
	}

	return count
}

// matchPattern matches a path against a glob pattern.
// Supports "**" for matching any path segment (zero or more segments).
// Supports "*" for matching a single path segment.
func matchPattern(pattern, fullPath string) bool {
	// Handle patterns with "**"
	if strings.Contains(pattern, "**") {
		return matchDoubleGlob(pattern, fullPath)
	}

	// Try standard glob matching on the full path
	matched, _ := path.Match(pattern, fullPath)
	if matched {
		return true
	}

	// For patterns like "*/encoder/*", try matching against path segments
	// Split paths into segments (skip empty strings from leading /)
	pathSegs := splitPathSegments(fullPath)
	patternSegs := splitPathSegments(pattern)

	return matchSegments(patternSegs, pathSegs)
}

// splitPathSegments splits a path into non-empty segments.
func splitPathSegments(p string) []string {
	var segs []string
	for _, s := range strings.Split(p, "/") {
		if s != "" {
			segs = append(segs, s)
		}
	}
	return segs
}

// matchDoubleGlob handles patterns containing "**".
func matchDoubleGlob(pattern, fullPath string) bool {
	// Split pattern by "**" to get the fixed parts
	parts := strings.Split(pattern, "**")

	pathSegs := splitPathSegments(fullPath)

	// For patterns like "**/foo/**", we need to check if "foo" appears anywhere in the path
	// For patterns like "**/foo", check if path ends with "foo"
	// For patterns like "foo/**", check if path starts with "foo"

	// Build a list of required segment patterns between the "**"
	var required [][]string
	for _, part := range parts {
		segs := splitPathSegments(part)
		if len(segs) > 0 {
			required = append(required, segs)
		}
	}

	// If no required segments, "**" matches everything
	if len(required) == 0 {
		return true
	}

	// For a single required segment group (e.g., "**/encoder/**" has ["encoder"])
	// Check if it appears anywhere in the path
	if len(required) == 1 {
		return containsSegments(pathSegs, required[0])
	}

	// For multiple required groups, they must appear in order
	return matchOrderedSegments(pathSegs, required)
}

// containsSegments checks if patternSegs appear contiguously anywhere in pathSegs.
func containsSegments(pathSegs, patternSegs []string) bool {
	if len(patternSegs) > len(pathSegs) {
		return false
	}

	for i := 0; i <= len(pathSegs)-len(patternSegs); i++ {
		match := true
		for j, ps := range patternSegs {
			matched, _ := path.Match(ps, pathSegs[i+j])
			if !matched {
				match = false
				break
			}
		}
		if match {
			return true
		}
	}
	return false
}

// matchOrderedSegments checks if multiple segment groups appear in order.
func matchOrderedSegments(pathSegs []string, required [][]string) bool {
	pathIdx := 0
	for _, group := range required {
		found := false
		for ; pathIdx <= len(pathSegs)-len(group); pathIdx++ {
			match := true
			for j, ps := range group {
				matched, _ := path.Match(ps, pathSegs[pathIdx+j])
				if !matched {
					match = false
					break
				}
			}
			if match {
				pathIdx += len(group)
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

// matchSegments attempts to match pattern segments against path segments.
// Handles "*" wildcards within segments.
func matchSegments(patternSegs, pathSegs []string) bool {
	if len(patternSegs) > len(pathSegs) {
		return false
	}

	// Try matching at each possible starting position
	for i := 0; i <= len(pathSegs)-len(patternSegs); i++ {
		match := true
		for j, ps := range patternSegs {
			matched, _ := path.Match(ps, pathSegs[i+j])
			if !matched {
				match = false
				break
			}
		}
		if match {
			return true
		}
	}
	return false
}

// LayerGroup defines a group of layers with a specific learning rate multiplier.
type LayerGroup struct {
	// Pattern is a glob pattern to match variable paths.
	Pattern string

	// LRMultiplier is the multiplier applied to the base learning rate for this group.
	// A value of 0.1 means the learning rate will be 10x smaller than the base.
	LRMultiplier float64
}

// LRMultiplierKey is the context parameter key for storing per-scope LR multipliers.
const LRMultiplierKey = "lr_multiplier"

// SetDiscriminativeLR configures per-layer-group learning rate multipliers.
//
// This allows different parts of the model to train at different rates,
// which is useful for fine-tuning where you want the pre-trained layers
// to update more slowly than the new task-specific layers.
//
// The multipliers are stored in the context and should be read by the optimizer.
//
// Example:
//
//	finetune.SetDiscriminativeLR(ctx, []finetune.LayerGroup{
//	    {Pattern: "*/encoder/*", LRMultiplier: 0.1},  // Encoder trains 10x slower
//	    {Pattern: "*/head/*", LRMultiplier: 1.0},     // Head trains at full LR
//	})
//
// IMPORTANT: This feature requires manual integration with your optimizer.
// Currently, no built-in GoMLX optimizers automatically read these multipliers.
// To use discriminative learning rates, you must:
//  1. Call GetLRMultiplier(ctx, varPath) for each variable during optimization
//  2. Multiply the base learning rate by the returned multiplier
//  3. Apply the scaled learning rate to that variable's gradient update
//
// This is planned for future implementation in the standard optimizers.
func SetDiscriminativeLR(ctx *context.Context, groups []LayerGroup) {
	// Store the groups in context for the optimizer to read
	ctx.SetParam(LRMultiplierKey, groups)
}

// GetLRMultiplier returns the learning rate multiplier for a given variable path.
// If no matching group is found, returns 1.0 (no modification).
//
// IMPORTANT: This function only retrieves the stored multiplier values.
// You must manually integrate this into your optimizer's gradient update logic.
// See SetDiscriminativeLR for details on how to use this feature.
func GetLRMultiplier(ctx *context.Context, varPath string) float64 {
	groups := context.GetParamOr(ctx, LRMultiplierKey, ([]LayerGroup)(nil))
	if groups == nil {
		return 1.0
	}

	// Find the first matching group
	for _, group := range groups {
		if matchPattern(group.Pattern, varPath) {
			return group.LRMultiplier
		}
	}

	return 1.0
}

// GradualUnfreezeConfig configures gradual unfreezing during training.
type GradualUnfreezeConfig struct {
	// Schedule maps step numbers to patterns to unfreeze.
	// At each step threshold, the matching patterns are unfrozen.
	Schedule map[int][]string
}

// GradualUnfreezeHook creates a training hook that gradually unfreezes layers.
//
// This implements the gradual unfreezing strategy from "Universal Language Model
// Fine-tuning for Text Classification" (Howard & Ruder, 2018), where layers
// are progressively unfrozen from the top (task-specific) to the bottom (general).
//
// Example:
//
//	hook := finetune.GradualUnfreezeHook(ctx, finetune.GradualUnfreezeConfig{
//	    Schedule: map[int][]string{
//	        0:    {"*/head/*"},           // Start with only head unfrozen
//	        1000: {"*/layer_11/*"},       // After 1000 steps, unfreeze layer 11
//	        2000: {"*/layer_10/*", "*/layer_9/*"},  // After 2000 steps, unfreeze layers 10, 9
//	    },
//	})
//	train.EveryNSteps(loop, 1, "gradual_unfreeze", 90, hook)
func GradualUnfreezeHook(ctx *context.Context, config GradualUnfreezeConfig) train.OnStepFn {
	// Track which steps we've already processed
	processedSteps := make(map[int]bool)
	var mu sync.Mutex

	return func(loop *train.Loop, _ []*tensors.Tensor) error {
		currentStep := loop.LoopStep

		mu.Lock()
		defer mu.Unlock()

		// Check each threshold in the schedule
		for step, patterns := range config.Schedule {
			if currentStep >= step && !processedSteps[step] {
				// Unfreeze all matching patterns
				for _, pattern := range patterns {
					UnfreezeByPattern(ctx, pattern)
				}
				processedSteps[step] = true
			}
		}

		return nil
	}
}

// CountTrainable returns the number of trainable variables in the context.
// Internal variables (those starting with #) are excluded.
func CountTrainable(ctx *context.Context) int {
	count := 0
	for v := range ctx.IterVariables() {
		if isInternalVariable(v) {
			continue
		}
		if v.Trainable {
			count++
		}
	}
	return count
}

// CountFrozen returns the number of frozen (non-trainable) variables in the context.
// Internal variables (those starting with #) are excluded.
func CountFrozen(ctx *context.Context) int {
	count := 0
	for v := range ctx.IterVariables() {
		if isInternalVariable(v) {
			continue
		}
		if !v.Trainable {
			count++
		}
	}
	return count
}

// ListVariables returns a list of all variable paths in the context.
// Internal variables (those starting with #) are excluded.
// Useful for debugging pattern matching.
func ListVariables(ctx *context.Context) []string {
	var paths []string
	for v := range ctx.IterVariables() {
		if isInternalVariable(v) {
			continue
		}
		paths = append(paths, v.Scope()+"/"+v.Name())
	}
	return paths
}

// ListTrainableVariables returns a list of all trainable variable paths.
// Internal variables (those starting with #) are excluded.
func ListTrainableVariables(ctx *context.Context) []string {
	var paths []string
	for v := range ctx.IterVariables() {
		if isInternalVariable(v) {
			continue
		}
		if v.Trainable {
			paths = append(paths, v.Scope()+"/"+v.Name())
		}
	}
	return paths
}

// ListFrozenVariables returns a list of all frozen variable paths.
// Internal variables (those starting with #) are excluded.
func ListFrozenVariables(ctx *context.Context) []string {
	var paths []string
	for v := range ctx.IterVariables() {
		if isInternalVariable(v) {
			continue
		}
		if !v.Trainable {
			paths = append(paths, v.Scope()+"/"+v.Name())
		}
	}
	return paths
}
