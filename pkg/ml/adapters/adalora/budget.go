// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package adalora

import (
	"sort"
	"strings"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
)

// LayerImportance holds importance information for a single AdaLoRA layer.
type LayerImportance struct {
	// Scope is the context scope for this layer's variables.
	Scope string

	// Importance contains the importance scores [rank].
	Importance []float32

	// Mask contains the current pruning mask [rank].
	Mask []float32

	// ActiveRank is the number of active (non-pruned) singular values.
	ActiveRank int
}

// BudgetState tracks the global budget allocation state across all layers.
type BudgetState struct {
	// Layers contains importance info for each AdaLoRA layer.
	Layers []LayerImportance

	// TotalActiveSingularValues is the sum of active ranks across all layers.
	TotalActiveSingularValues int

	// Step is the current training step.
	Step int
}

// NewBudgetState creates a new budget state by scanning the context for AdaLoRA layers.
func NewBudgetState(ctx *context.Context) *BudgetState {
	state := &BudgetState{
		Layers: []LayerImportance{},
	}

	// Find all AdaLoRA layers by scanning for adalora_lambda variables
	seenScopes := make(map[string]bool)
	for v := range ctx.IterVariables() {
		if strings.Contains(v.Name(), "adalora_lambda") {
			scope := v.Scope()
			if seenScopes[scope] {
				continue
			}
			seenScopes[scope] = true

			rank := v.Shape().Dimensions[0]
			layer := LayerImportance{
				Scope:      scope,
				Importance: make([]float32, rank),
				Mask:       make([]float32, rank),
				ActiveRank: rank,
			}
			// Initialize mask to all ones
			for i := range layer.Mask {
				layer.Mask[i] = 1.0
			}
			state.Layers = append(state.Layers, layer)
			state.TotalActiveSingularValues += rank
		}
	}

	return state
}

// UpdateImportance updates the importance scores using EMA with lambda gradients.
//
// The importance of each singular value is estimated from its gradient magnitude:
//
//	importance_new = beta * importance_old + (1 - beta) * |grad_lambda|
//
// This should be called after each training step with gradient information.
//
// Parameters:
//   - ctx: Context containing the variables
//   - lambdaGradients: Map from scope to lambda gradient tensor
//   - beta: EMA smoothing factor
func (s *BudgetState) UpdateImportance(ctx *context.Context, lambdaGradients map[string]*tensors.Tensor, beta float64) {
	for i := range s.Layers {
		layer := &s.Layers[i]

		// Get gradient for this layer
		grad, ok := lambdaGradients[layer.Scope]
		if !ok {
			continue
		}

		// Extract gradient values
		var gradValues []float32
		err := grad.ConstFlatData(func(flat any) {
			gradValues = flat.([]float32)
		})
		if err != nil {
			continue
		}

		// Update importance with EMA
		for j := 0; j < len(layer.Importance) && j < len(gradValues); j++ {
			absGrad := gradValues[j]
			if absGrad < 0 {
				absGrad = -absGrad
			}
			layer.Importance[j] = float32(beta)*layer.Importance[j] + float32(1-beta)*absGrad
		}
	}
}

// UpdateImportanceFromLambda updates importance directly from lambda values.
// This is a simpler approach that uses |lambda| as importance.
//
// Parameters:
//   - ctx: Context containing the variables
//   - beta: EMA smoothing factor
func (s *BudgetState) UpdateImportanceFromLambda(ctx *context.Context, beta float64) {
	for i := range s.Layers {
		layer := &s.Layers[i]

		// Find lambda variable for this layer
		for v := range ctx.IterVariables() {
			if v.Scope() == layer.Scope && strings.Contains(v.Name(), "adalora_lambda") {
				tensor, err := v.Value()
				if err != nil {
					continue
				}
				var lambdaValues []float32
				err = tensor.ConstFlatData(func(flat any) {
					lambdaValues = flat.([]float32)
				})
				if err != nil {
					continue
				}

				// Update importance with EMA of |lambda|
				for j := 0; j < len(layer.Importance) && j < len(lambdaValues); j++ {
					absLambda := lambdaValues[j]
					if absLambda < 0 {
						absLambda = -absLambda
					}
					layer.Importance[j] = float32(beta)*layer.Importance[j] + float32(1-beta)*absLambda
				}
				break
			}
		}
	}
}

// singularValueScore combines importance and mask for sorting
type singularValueScore struct {
	layerIdx int
	svIdx    int
	score    float32
}

// ReallocateBudget redistributes rank budget across layers based on importance.
//
// This implements the core AdaLoRA budget allocation:
// 1. Collect all importance scores across all layers
// 2. Sort by importance (lowest first)
// 3. Prune the lowest-importance singular values until target budget is reached
//
// Parameters:
//   - ctx: Context to update mask variables
//   - targetBudget: Target total number of active singular values
//   - minRankPerLayer: Minimum rank to keep per layer
func (s *BudgetState) ReallocateBudget(ctx *context.Context, targetBudget, minRankPerLayer int) {
	// Collect all singular value scores
	var scores []singularValueScore
	for layerIdx, layer := range s.Layers {
		for svIdx, importance := range layer.Importance {
			if layer.Mask[svIdx] > 0.5 { // Only consider currently active values
				scores = append(scores, singularValueScore{
					layerIdx: layerIdx,
					svIdx:    svIdx,
					score:    importance,
				})
			}
		}
	}

	// Sort by score (ascending - lowest importance first)
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score < scores[j].score
	})

	// Calculate how many to prune
	currentBudget := len(scores)
	toPrune := currentBudget - targetBudget
	if toPrune <= 0 {
		return // Already at or below budget
	}

	// Track active ranks per layer to enforce minimum
	activePerLayer := make([]int, len(s.Layers))
	for i, layer := range s.Layers {
		activePerLayer[i] = layer.ActiveRank
	}

	// Prune lowest-importance values while respecting min rank
	pruned := 0
	for _, sv := range scores {
		if pruned >= toPrune {
			break
		}

		// Check if we can prune from this layer
		if activePerLayer[sv.layerIdx] <= minRankPerLayer {
			continue // Can't prune, would go below minimum
		}

		// Prune this singular value
		s.Layers[sv.layerIdx].Mask[sv.svIdx] = 0.0
		activePerLayer[sv.layerIdx]--
		pruned++
	}

	// Update active ranks
	for i := range s.Layers {
		s.Layers[i].ActiveRank = activePerLayer[i]
	}

	// Calculate new total
	s.TotalActiveSingularValues = 0
	for _, layer := range s.Layers {
		s.TotalActiveSingularValues += layer.ActiveRank
	}

	// Write masks back to context
	s.writeMasksToContext(ctx)
}

// writeMasksToContext writes the current masks back to context variables.
func (s *BudgetState) writeMasksToContext(ctx *context.Context) {
	for _, layer := range s.Layers {
		for v := range ctx.IterVariables() {
			if v.Scope() == layer.Scope && strings.Contains(v.Name(), "adalora_mask") {
				// Create tensor from mask values
				tensor := tensors.FromFlatDataAndDimensions(layer.Mask, v.Shape().Dimensions...)
				_ = v.SetValuePreservingOld(tensor)
				break
			}
		}
	}
}

// BudgetStep performs one step of budget allocation.
// Should be called at each training step.
//
// Parameters:
//   - ctx: Context containing variables
//   - config: AdaLoRA configuration
//   - globalStep: Current training step
//
// Returns: Whether pruning was performed
func (s *BudgetState) BudgetStep(ctx *context.Context, config *Config, globalStep int) bool {
	s.Step = globalStep

	// During warmup, just update importance scores
	if globalStep < config.WarmupSteps {
		s.UpdateImportanceFromLambda(ctx, config.Beta)
		return false
	}

	// After final budgeting step, do nothing
	if globalStep > config.FinalBudgetingStep {
		return false
	}

	// Update importance
	s.UpdateImportanceFromLambda(ctx, config.Beta)

	// Check if this is a budgeting step
	if (globalStep-config.WarmupSteps)%config.BudgetingInterval != 0 {
		return false
	}

	// Calculate target budget for this step
	// Linearly decrease from initial to target
	totalLayers := len(s.Layers)
	if totalLayers == 0 {
		return false
	}

	initialBudget := totalLayers * config.InitialRank()
	targetBudget := totalLayers * config.TargetRank()

	// Calculate progress through budgeting phase
	budgetingSteps := config.FinalBudgetingStep - config.WarmupSteps
	if budgetingSteps <= 0 {
		budgetingSteps = 1
	}
	progress := float64(globalStep-config.WarmupSteps) / float64(budgetingSteps)
	if progress > 1.0 {
		progress = 1.0
	}

	// Interpolate budget
	currentTargetBudget := int(float64(initialBudget) + progress*(float64(targetBudget)-float64(initialBudget)))

	// Reallocate budget
	s.ReallocateBudget(ctx, currentTargetBudget, config.MinRank)

	return true
}

// GetLayerRanks returns the current active rank for each layer.
func (s *BudgetState) GetLayerRanks() map[string]int {
	ranks := make(map[string]int)
	for _, layer := range s.Layers {
		ranks[layer.Scope] = layer.ActiveRank
	}
	return ranks
}

// GetTotalParameters returns the total number of trainable parameters.
func (s *BudgetState) GetTotalParameters(inputDims, outputDims map[string]int) int64 {
	var total int64
	for _, layer := range s.Layers {
		inputDim := inputDims[layer.Scope]
		outputDim := outputDims[layer.Scope]
		rank := layer.ActiveRank

		// P: [inputDim, rank], Lambda: [rank], Q: [rank, outputDim]
		total += int64(inputDim*rank + rank + rank*outputDim)
	}
	return total
}

// ImportanceStats returns statistics about importance scores.
type ImportanceStats struct {
	Min    float32
	Max    float32
	Mean   float32
	Median float32
}

// GetImportanceStats returns statistics about the importance scores.
func (s *BudgetState) GetImportanceStats() ImportanceStats {
	var allImportance []float32
	for _, layer := range s.Layers {
		for i, importance := range layer.Importance {
			if layer.Mask[i] > 0.5 { // Only active values
				allImportance = append(allImportance, importance)
			}
		}
	}

	if len(allImportance) == 0 {
		return ImportanceStats{}
	}

	// Sort for median
	sorted := make([]float32, len(allImportance))
	copy(sorted, allImportance)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })

	var sum float32
	min := sorted[0]
	max := sorted[len(sorted)-1]
	for _, v := range sorted {
		sum += v
	}

	return ImportanceStats{
		Min:    min,
		Max:    max,
		Mean:   sum / float32(len(sorted)),
		Median: sorted[len(sorted)/2],
	}
}

// CreateBudgetStateVariable creates a variable to store the budget state.
// This is useful for checkpointing.
func CreateBudgetStateVariable(ctx *context.Context, state *BudgetState) {
	// Store total layers and ranks as a tensor for checkpoint compatibility
	numLayers := len(state.Layers)
	if numLayers == 0 {
		return
	}

	// Pack state into a tensor: [numLayers, maxRank] for masks + [numLayers, maxRank] for importance
	maxRank := 0
	for _, layer := range state.Layers {
		if len(layer.Mask) > maxRank {
			maxRank = len(layer.Mask)
		}
	}

	// Create packed data
	data := make([]float32, numLayers*maxRank*2) // masks + importance
	for i, layer := range state.Layers {
		for j := range layer.Mask {
			data[i*maxRank+j] = layer.Mask[j]                        // masks
			data[numLayers*maxRank+i*maxRank+j] = layer.Importance[j] // importance
		}
	}

	tensor := tensors.FromFlatDataAndDimensions(data, 2, numLayers, maxRank)
	ctx.In("adalora_budget").VariableWithValue("state", tensor).SetTrainable(false)
}

// LoadBudgetStateFromVariable loads the budget state from a checkpoint variable.
func LoadBudgetStateFromVariable(ctx *context.Context, state *BudgetState) error {
	budgetVar := ctx.GetVariableByScopeAndName("adalora_budget", "state")
	if budgetVar == nil {
		return nil // No saved state
	}

	tensor, err := budgetVar.Value()
	if err != nil {
		return err
	}
	if tensor.Shape().Rank() != 3 || tensor.Shape().Dimensions[0] != 2 {
		return nil // Invalid shape
	}

	numLayers := tensor.Shape().Dimensions[1]
	maxRank := tensor.Shape().Dimensions[2]

	var data []float32
	err = tensor.ConstFlatData(func(flat any) {
		data = flat.([]float32)
	})
	if err != nil {
		return err
	}

	// Unpack data
	for i := range state.Layers {
		if i >= numLayers {
			break
		}
		for j := range state.Layers[i].Mask {
			if j >= maxRank {
				break
			}
			state.Layers[i].Mask[j] = data[i*maxRank+j]
			state.Layers[i].Importance[j] = data[numLayers*maxRank+i*maxRank+j]
		}
		// Recalculate active rank
		state.Layers[i].ActiveRank = 0
		for _, m := range state.Layers[i].Mask {
			if m > 0.5 {
				state.Layers[i].ActiveRank++
			}
		}
	}

	// Update total
	state.TotalActiveSingularValues = 0
	for _, layer := range state.Layers {
		state.TotalActiveSingularValues += layer.ActiveRank
	}

	// Write masks back to context
	state.writeMasksToContext(ctx)

	return nil
}

// MaskShape returns the shape needed for the mask variable.
func MaskShape(rank int) shapes.Shape {
	return shapes.Make(dtypes.Float32, rank)
}
