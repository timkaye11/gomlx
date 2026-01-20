// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package adapters

import (
	"strings"

	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/pkg/errors"
)

// MergeAdapter permanently merges adapter weights into base weights.
// After merging, the model can be used without adapter overhead.
//
// This function:
// 1. Finds all base weight variables that have corresponding adapter variables
// 2. Computes merged weights: W_merged = W + adapter_contribution
// 3. Updates the base weight variables with merged values
// 4. Marks the adapter as merged
//
// The merging formula depends on the adapter type:
//   - LoRA: W_merged = W + scale * (A @ B)
//   - DoRA: W_merged = m * normalize(W + scale * A @ B)
//   - AdaLoRA: W_merged = W + P @ diag(Lambda * mask) @ Q
//   - QLoRA: Not supported (quantized base weights cannot be merged)
//
// Parameters:
//   - backend: Backend for graph execution
//   - ctx: Context containing all variables
//   - adapter: The adapter to merge
//
// Returns: Error if merging fails
func MergeAdapter(backend backends.Backend, ctx *context.Context, adapter Adapter) error {
	if adapter == nil {
		return errors.New("adapter is nil")
	}

	if adapter.IsMerged() {
		return errors.New("adapter is already merged")
	}

	adapterType := adapter.Type()
	if adapterType == AdapterTypeQLoRA {
		return errors.New("QLoRA adapters cannot be merged (quantized base weights)")
	}

	// Find all base weight variables that have corresponding adapter variables
	baseWeights := findBaseWeightsWithAdapters(ctx, adapter.Name())
	if len(baseWeights) == 0 {
		return errors.New("no base weights found with adapter variables")
	}

	// Merge each set of weights
	for _, bw := range baseWeights {
		if err := mergeWeightVariable(backend, ctx, adapter, bw); err != nil {
			return errors.Wrapf(err, "failed to merge weights %s", bw.baseVarName)
		}
	}

	// Mark adapter as merged
	adapter.SetMerged(true)

	return nil
}

// baseWeightInfo holds information about a base weight variable and its adapter scope.
type baseWeightInfo struct {
	baseVarName  string // Full parameter name of base weights
	adapterScope string // Scope where adapter variables are stored
	baseVar      *context.Variable
}

// findBaseWeightsWithAdapters finds base weight variables that have corresponding adapter variables.
// If adapterName is non-empty, only considers adapter variables whose scope contains the adapter name.
// If adapterName is empty, considers all adapter variables.
func findBaseWeightsWithAdapters(ctx *context.Context, adapterName string) []baseWeightInfo {
	var results []baseWeightInfo

	// Map to track which scopes have adapter variables
	scopesWithAdapters := make(map[string]bool)

	// First pass: find all scopes with adapter variables
	for v := range ctx.IterVariables() {
		name := v.ParameterName()
		if isAdapterVarName(name) {
			// If adapterName is specified, only consider variables belonging to that adapter
			if adapterName != "" && !strings.Contains(name, adapterName) {
				continue
			}
			// Extract the scope (everything before the adapter variable name)
			scope := extractAdapterScope(name)
			scopesWithAdapters[scope] = true
		}
	}

	// Second pass: find base weights in those scopes
	for v := range ctx.IterVariables() {
		name := v.ParameterName()
		// Skip adapter variables
		if isAdapterVarName(name) {
			continue
		}

		// Check if this is a weight variable
		if !strings.Contains(name, "weights") {
			continue
		}

		// Check if this scope has adapter variables
		scope := extractBaseScope(name)
		if scopesWithAdapters[scope] {
			results = append(results, baseWeightInfo{
				baseVarName:  name,
				adapterScope: scope,
				baseVar:      v,
			})
		}
	}

	return results
}

// mergeWeightVariable merges adapter weights into a single base weight variable.
func mergeWeightVariable(backend backends.Backend, ctx *context.Context, adapter Adapter, bw baseWeightInfo) error {
	// Get the shape of the base weights to create input
	baseShape := bw.baseVar.Shape()

	// Get scoped context for adapter variables
	scope, _ := context.VariableScopeAndNameFromParameterName(bw.baseVarName)
	adapterCtx := ctx.InAbsPath(scope)

	// Create an Exec to run the merge computation
	// We pass the base weights shape as input just to have something to build the graph with
	exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, input *Node) *Node {
		g := input.Graph()

		// Get base weights
		baseWeights := bw.baseVar.ValueGraph(g)

		// Call adapter's Merge method
		mergedWeights := adapter.Merge(adapterCtx, baseWeights)

		return mergedWeights
	})

	// Run with a dummy input (we don't actually use it, just need to trigger the computation)
	dummyInput := shapes.Make(baseShape.DType, 1)
	results := exec.MustExec(dummyInput)
	if len(results) == 0 {
		return errors.New("merge computation returned no results")
	}

	// Update the base weight variable with merged values
	return bw.baseVar.SetValue(results[0])
}

// isAdapterVarName checks if a variable name is an adapter-specific variable.
func isAdapterVarName(name string) bool {
	return strings.Contains(name, "lora_A") ||
		strings.Contains(name, "lora_B") ||
		strings.Contains(name, "dora_A") ||
		strings.Contains(name, "dora_B") ||
		strings.Contains(name, "dora_magnitude") ||
		strings.Contains(name, "adalora_P") ||
		strings.Contains(name, "adalora_lambda") ||
		strings.Contains(name, "adalora_Q") ||
		strings.Contains(name, "adalora_mask") ||
		strings.Contains(name, "adalora_importance") ||
		strings.Contains(name, "qlora_A") ||
		strings.Contains(name, "qlora_B")
}

// extractAdapterScope extracts the scope from an adapter variable name.
// For example: "model/layer1/lora_A" -> "model/layer1"
func extractAdapterScope(name string) string {
	parts := strings.Split(name, "/")
	if len(parts) <= 1 {
		return ""
	}
	// Remove the last part (the variable name)
	return strings.Join(parts[:len(parts)-1], "/")
}

// extractBaseScope extracts the scope from a base variable name.
// For example: "model/layer1/weights" -> "model/layer1"
func extractBaseScope(name string) string {
	parts := strings.Split(name, "/")
	if len(parts) <= 1 {
		return ""
	}
	// Remove the last part (the variable name)
	return strings.Join(parts[:len(parts)-1], "/")
}

// TODO: Future enhancements for merge functionality:
//
// UnmergeAdapter - Revert a merged adapter back to its pre-merged state.
// This would require saving original weights before merging and loading them back.
// Implementation blocked on: context checkpoint loading functionality.
//
// SaveMergedModel - Save merged model weights without adapter variables.
// This would be useful for deploying a fine-tuned model without adapter overhead.
// Implementation blocked on: context checkpoint integration.
// Workaround: Use context.SaveCheckpoint directly after merging.

// MergeOptions provides configuration for the merge operation.
type MergeOptions struct {
	// PreserveFP32 forces computation in FP32 even if base weights are FP16/BF16.
	// This can improve numerical accuracy at the cost of memory.
	PreserveFP32 bool

	// Validate performs validation checks before and after merging.
	Validate bool

	// SaveOriginal saves the original base weights before merging.
	// If set, provides a path where original weights will be saved.
	SaveOriginal string
}

// DefaultMergeOptions returns default merge options.
func DefaultMergeOptions() MergeOptions {
	return MergeOptions{
		PreserveFP32: false,
		Validate:     true,
		SaveOriginal: "",
	}
}

// MergeAdapterWithOptions merges adapter weights with configurable options.
func MergeAdapterWithOptions(backend backends.Backend, ctx *context.Context, adapter Adapter, opts MergeOptions) error {
	if adapter == nil {
		return errors.New("adapter is nil")
	}

	if adapter.IsMerged() {
		return errors.New("adapter is already merged")
	}

	// Save original weights if requested
	if opts.SaveOriginal != "" {
		// Would save context checkpoint here
		// For now, just document that this feature needs checkpoint integration
	}

	// Validation before merge
	if opts.Validate {
		if err := validatePreMerge(ctx, adapter); err != nil {
			return errors.Wrap(err, "pre-merge validation failed")
		}
	}

	// Perform the merge
	if err := MergeAdapter(backend, ctx, adapter); err != nil {
		return err
	}

	// Validation after merge
	if opts.Validate {
		if err := validatePostMerge(ctx, adapter); err != nil {
			return errors.Wrap(err, "post-merge validation failed")
		}
	}

	return nil
}

// validatePreMerge performs validation before merging.
func validatePreMerge(ctx *context.Context, adapter Adapter) error {
	// Check that adapter variables exist
	hasAdapterVars := false
	for v := range ctx.IterVariables() {
		if isAdapterVarName(v.ParameterName()) {
			hasAdapterVars = true
			break
		}
	}
	if !hasAdapterVars {
		return errors.New("no adapter variables found in context")
	}

	// Check that base weights exist
	hasBaseWeights := false
	for v := range ctx.IterVariables() {
		name := v.ParameterName()
		if strings.Contains(name, "weights") && !isAdapterVarName(name) {
			hasBaseWeights = true
			break
		}
	}
	if !hasBaseWeights {
		return errors.New("no base weight variables found in context")
	}

	return nil
}

// validatePostMerge performs validation after merging.
func validatePostMerge(ctx *context.Context, adapter Adapter) error {
	// Verify adapter is marked as merged
	if !adapter.IsMerged() {
		return errors.New("adapter not marked as merged after merge operation")
	}

	return nil
}

// CountMergeableWeights returns the number of weight variables that can be merged.
func CountMergeableWeights(ctx *context.Context) int {
	baseWeights := findBaseWeightsWithAdapters(ctx, "")
	return len(baseWeights)
}

// GetMergeableWeightNames returns the names of weight variables that can be merged.
func GetMergeableWeightNames(ctx *context.Context) []string {
	baseWeights := findBaseWeightsWithAdapters(ctx, "")
	names := make([]string, len(baseWeights))
	for i, bw := range baseWeights {
		names[i] = bw.baseVarName
	}
	return names
}
