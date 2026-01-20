// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package ia3 implements IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations).
//
// IA³ is a highly parameter-efficient fine-tuning method that uses learned rescaling vectors
// to modify key, value, and feedforward activations. Unlike LoRA which uses low-rank matrices,
// IA³ uses simple element-wise multiplication with learned vectors.
//
// The method is described in "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper
// than In-Context Learning" (Liu et al., 2022): https://arxiv.org/abs/2205.05638
//
// Key advantages:
//   - Extremely few parameters (only scaling vectors)
//   - Simple forward pass modification
//   - Works well for few-shot learning scenarios
//
// Example usage:
//
//	config := ia3.NewConfig().SetInitScale(1.0)
//	// Use with Dense layers that will be scaled
//	output := ia3.ScaleActivation(ctx, hidden, config, hiddenDim)
package ia3

import (
	"strings"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/adapters"
	"github.com/gomlx/gomlx/pkg/ml/context"
)

// Adapter implements the IA³ adapter.
type Adapter struct {
	adapters.BaseAdapter
	config *Config
}

// New creates a new IA³ adapter with the given name and configuration.
// If config is nil, default configuration is used.
func New(name string, config *Config) (*Adapter, error) {
	if config == nil {
		config = NewConfig()
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	return &Adapter{
		BaseAdapter: adapters.NewBaseAdapter(name),
		config:      config,
	}, nil
}

// Type returns the adapter type.
func (a *Adapter) Type() adapters.AdapterType {
	return adapters.AdapterTypeIA3
}

// Config returns the adapter configuration.
func (a *Adapter) Config() adapters.AdapterConfig {
	return a.config
}

// Apply scales the activation with the learned IA³ vector.
// Unlike LoRA which takes input and base output, IA³ directly rescales the activation.
func (a *Adapter) Apply(ctx *context.Context, input, baseOutput *Node, inputDim, outputDim int) *Node {
	if !a.config.Enabled() || a.IsMerged() {
		return baseOutput
	}
	return ScaleActivation(ctx, baseOutput, a.config, outputDim)
}

// Merge merges the IA³ scaling into base weights.
// For IA³, merging means multiplying the base weights by the scaling vector.
// Note: This is only applicable when the scaling is applied after the linear layer,
// so we can absorb it into the weight matrix: W' = diag(l) @ W
func (a *Adapter) Merge(ctx *context.Context, baseWeights *Node) *Node {
	g := baseWeights.Graph()
	outputDim := baseWeights.Shape().Dim(-1)

	// Get the scaling vector
	scaleVar := ctx.GetVariableByScopeAndName(ctx.Scope(), "ia3_scale")
	if scaleVar == nil {
		// No IA³ variables found, return base weights unchanged
		return baseWeights
	}

	scale := scaleVar.ValueGraph(g)

	// For a weight matrix W [inputDim, outputDim] and scale vector l [outputDim],
	// the merged weights are: W' = W * l (broadcasting)
	// This is equivalent to: output = x @ W * l = x @ (W * l[None, :])
	if scale.Shape().Rank() == 1 && scale.Shape().Dim(0) == outputDim {
		// Reshape scale for broadcasting: [outputDim] -> [1, outputDim]
		scale = Reshape(scale, 1, outputDim)
	}

	return Mul(baseWeights, scale)
}

// CountParameters returns the number of trainable IA³ parameters.
func (a *Adapter) CountParameters(ctx *context.Context) int64 {
	var count int64
	for v := range ctx.IterVariables() {
		if IsIA3Variable(v.ParameterName()) && v.Trainable {
			count += int64(v.Shape().Size())
		}
	}
	return count
}

// FreezeBaseWeights marks all non-IA³ weights as non-trainable.
func (a *Adapter) FreezeBaseWeights(ctx *context.Context) {
	for v := range ctx.IterVariables() {
		if !IsIA3Variable(v.ParameterName()) {
			v.Trainable = false
		}
	}
}

// UnfreezeBaseWeights marks all weights as trainable.
func (a *Adapter) UnfreezeBaseWeights(ctx *context.Context) {
	for v := range ctx.IterVariables() {
		v.Trainable = true
	}
}

// ScaleActivation applies IA³ scaling to an activation tensor.
// The scaling vector has shape [dim] and is multiplied element-wise.
//
// Parameters:
//   - ctx: Context for variable creation
//   - activation: Input tensor [batch, ..., dim]
//   - config: IA³ configuration
//   - dim: Dimension of the scaling vector (last dim of activation)
//
// Returns: Scaled activation
func ScaleActivation(ctx *context.Context, activation *Node, config *Config, dim int) *Node {
	if config != nil && !config.Enabled() {
		return activation
	}

	g := activation.Graph()
	dtype := activation.DType()

	// Initialize scale vector to config.InitScale (default 1.0 for identity start)
	initScale := 1.0
	if config != nil {
		initScale = config.InitScale()
	}

	// Create scaling vector variable with constant initializer
	scaleInitializer := func(g *Graph, shape shapes.Shape) *Node {
		return AddScalar(Zeros(g, shape), initScale)
	}
	scaleVar := ctx.WithInitializer(scaleInitializer).
		VariableWithShape("ia3_scale", shapes.Make(dtype, dim)).
		SetTrainable(true)

	scale := scaleVar.ValueGraph(g)

	// Apply dropout during training if configured
	if config != nil && config.Dropout() > 0 && ctx.IsTraining(g) {
		keepProb := Scalar(g, dtype, 1.0-config.Dropout())
		mask := ctx.RandomBernoulli(keepProb, scale.Shape())
		mask = ConvertDType(mask, dtype)
		// Scale to maintain expected value
		scaleVal := Scalar(g, dtype, 1.0/(1.0-config.Dropout()))
		scale = Mul(Mul(scale, mask), scaleVal)
	}

	// Element-wise multiply: output = activation * scale
	// Scale has shape [dim], activation has shape [batch, ..., dim]
	// Need to reshape scale for broadcasting: [dim] -> [1, ..., 1, dim]
	if activation.Rank() > 1 {
		// Build shape with 1s for all leading dimensions
		expandedShape := make([]int, activation.Rank())
		for i := 0; i < activation.Rank()-1; i++ {
			expandedShape[i] = 1
		}
		expandedShape[activation.Rank()-1] = dim
		scale = Reshape(scale, expandedShape...)
	}

	return Mul(activation, scale)
}

// ScaleKeyValue applies IA³ scaling to key and value tensors in attention.
// This is a convenience function for attention layer adaptation.
//
// Parameters:
//   - ctx: Context for variable creation
//   - key: Key tensor [batch, numHeads, seqLen, headDim]
//   - value: Value tensor [batch, numHeads, seqLen, headDim]
//   - config: IA³ configuration
//   - headDim: Dimension per attention head
//
// Returns: Scaled key and value tensors
func ScaleKeyValue(ctx *context.Context, key, value *Node, config *Config, headDim int) (*Node, *Node) {
	if config != nil && !config.Enabled() {
		return key, value
	}

	// Scale key
	scaledKey := ScaleActivation(ctx.In("ia3_key"), key, config, headDim)

	// Scale value
	scaledValue := ScaleActivation(ctx.In("ia3_value"), value, config, headDim)

	return scaledKey, scaledValue
}

// ScaleFeedForward applies IA³ scaling to feedforward intermediate activations.
// This is typically applied after the first linear layer (up projection) in the FFN.
//
// Parameters:
//   - ctx: Context for variable creation
//   - hidden: Intermediate activation [batch, seqLen, hiddenDim]
//   - config: IA³ configuration
//   - hiddenDim: Dimension of the intermediate representation
//
// Returns: Scaled intermediate activation
func ScaleFeedForward(ctx *context.Context, hidden *Node, config *Config, hiddenDim int) *Node {
	return ScaleActivation(ctx.In("ia3_ff"), hidden, config, hiddenDim)
}

// IsIA3Variable returns true if the variable name indicates an IA³ variable.
func IsIA3Variable(name string) bool {
	return strings.Contains(name, "ia3_scale") ||
		strings.Contains(name, "ia3_key") ||
		strings.Contains(name, "ia3_value") ||
		strings.Contains(name, "ia3_ff")
}

// GetIA3Variables returns all IA³ variables from the context.
func GetIA3Variables(ctx *context.Context) []*context.Variable {
	var vars []*context.Variable
	for v := range ctx.IterVariables() {
		if IsIA3Variable(v.ParameterName()) {
			vars = append(vars, v)
		}
	}
	return vars
}

// ShouldApplyIA3 checks if IA³ should be applied to a module based on its name.
// Uses the config's attention and feedforward module patterns.
func ShouldApplyIA3(config *Config, moduleName string) bool {
	if config == nil {
		return false
	}

	// Check attention modules
	for _, pattern := range config.AttentionModules() {
		if strings.Contains(moduleName, pattern) {
			return true
		}
	}

	// Check feedforward modules
	for _, pattern := range config.FeedForwardModules() {
		if strings.Contains(moduleName, pattern) {
			return true
		}
	}

	return false
}
