// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package adapters

import (
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
)

// Adapter is the main interface for all adapter implementations.
// Adapters modify layer computations without changing base weights,
// enabling parameter-efficient fine-tuning.
type Adapter interface {
	// Name returns the unique name of this adapter instance.
	Name() string

	// Type returns the adapter type.
	Type() AdapterType

	// Config returns the adapter's configuration.
	Config() AdapterConfig

	// Apply adds the adapter's transformation to the base output.
	//
	// Parameters:
	//   - ctx: Context scoped to the current layer
	//   - input: The input to the layer [batch, ..., inputDim]
	//   - baseOutput: The output from the base layer (before adapter)
	//   - inputDim, outputDim: Dimensions for parameter creation
	//
	// Returns: baseOutput + adapter contribution
	Apply(ctx *context.Context, input, baseOutput *Node, inputDim, outputDim int) *Node

	// Merge merges adapter weights into base weights permanently.
	// Returns the merged weights as a graph Node.
	//
	// For LoRA: W_merged = W + scale * (A @ B)
	// For DoRA: W_merged = m * normalize(W + scale * A @ B)
	// For AdaLoRA: W_merged = W + P @ diag(Lambda) @ Q
	//
	// After merging, the adapter should be disabled for inference
	// without adapter overhead.
	Merge(ctx *context.Context, baseWeights *Node) *Node

	// IsMerged returns whether this adapter has been merged into base weights.
	IsMerged() bool

	// SetMerged sets the merged state. Called after weights are merged.
	SetMerged(merged bool)

	// CountParameters returns the number of trainable parameters in this adapter.
	CountParameters(ctx *context.Context) int64

	// FreezeBaseWeights marks all non-adapter weights as non-trainable.
	// This is typically called after loading pre-trained weights.
	FreezeBaseWeights(ctx *context.Context)

	// UnfreezeBaseWeights marks all weights as trainable.
	UnfreezeBaseWeights(ctx *context.Context)
}

// AdapterFactory creates adapter instances from configurations.
type AdapterFactory interface {
	// Create returns a new adapter instance with the given name and config.
	// Returns an error if the config type doesn't match this factory.
	Create(name string, config AdapterConfig) (Adapter, error)

	// Type returns the adapter type this factory creates.
	Type() AdapterType
}

// BaseAdapter provides common functionality for adapter implementations.
// Embed this in specific adapters to reduce boilerplate.
type BaseAdapter struct {
	name   string
	merged bool
}

// NewBaseAdapter creates a BaseAdapter with the given name.
func NewBaseAdapter(name string) BaseAdapter {
	return BaseAdapter{
		name:   name,
		merged: false,
	}
}

// Name returns the adapter name.
func (a *BaseAdapter) Name() string {
	return a.name
}

// IsMerged returns whether the adapter is merged.
func (a *BaseAdapter) IsMerged() bool {
	return a.merged
}

// SetMerged sets the merged state.
func (a *BaseAdapter) SetMerged(merged bool) {
	a.merged = merged
}
