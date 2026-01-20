// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package lora implements Low-Rank Adaptation (LoRA) for efficient fine-tuning.
//
// LoRA freezes the pre-trained model weights and injects trainable low-rank
// decomposition matrices into each layer, greatly reducing the number of
// trainable parameters for downstream tasks.
//
// Reference: "LoRA: Low-Rank Adaptation of Large Language Models"
// https://arxiv.org/abs/2106.09685
//
// Basic usage:
//
//	// Create LoRA adapter
//	config := lora.NewConfig().SetRank(8).SetAlpha(16.0)
//	adapter, _ := lora.New("my_lora", config)
//
//	// Apply to a computation
//	output := adapter.Apply(ctx, input, baseOutput, inputDim, outputDim)
package lora

import (
	"math"
	"strings"

	"github.com/gomlx/gomlx/pkg/ml/adapters"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/initializers"
)

// Adapter implements the adapters.Adapter interface for LoRA.
type Adapter struct {
	adapters.BaseAdapter
	config *Config
}

// New creates a new LoRA adapter with the given name and configuration.
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
	return adapters.AdapterTypeLoRA
}

// Config returns the adapter configuration.
func (a *Adapter) Config() adapters.AdapterConfig {
	return a.config
}

// Apply adds the LoRA transformation to the base output.
//
// The LoRA update is: scale * (input @ A @ B)
// where A is [inputDim, rank] and B is [rank, outputDim].
//
// Parameters:
//   - ctx: Context scoped to the current layer
//   - input: The input to the layer [batch, ..., inputDim]
//   - baseOutput: The output from the base layer (before adapter)
//   - inputDim, outputDim: Dimensions for parameter creation
//
// Returns: baseOutput + LoRA adaptation
func (a *Adapter) Apply(ctx *context.Context, input, baseOutput *Node, inputDim, outputDim int) *Node {
	if !a.config.Enabled() || a.IsMerged() {
		return baseOutput
	}

	loraOutput := applyLoRA(ctx, input, a.config, inputDim, outputDim)
	return Add(baseOutput, loraOutput)
}

// Merge merges LoRA weights into base weights permanently.
//
// For LoRA: W_merged = W + scale * (A @ B)
//
// Parameters:
//   - ctx: Context for accessing LoRA variables
//   - baseWeights: The base weight matrix [inputDim, outputDim]
//
// Returns: The merged weight matrix
func (a *Adapter) Merge(ctx *context.Context, baseWeights *Node) *Node {
	if a.IsMerged() {
		return baseWeights
	}

	g := baseWeights.Graph()
	dtype := baseWeights.DType()
	shape := baseWeights.Shape()
	inputDim := shape.Dimensions[0]
	outputDim := shape.Dimensions[1]
	rank := a.config.Rank()

	// Get LoRA matrices
	loraAVar := ctx.GetVariableByScopeAndName(ctx.Scope(), "lora_A")
	loraBVar := ctx.GetVariableByScopeAndName(ctx.Scope(), "lora_B")

	if loraAVar == nil || loraBVar == nil {
		// LoRA weights don't exist yet, nothing to merge
		return baseWeights
	}

	loraA := loraAVar.ValueGraph(g)
	loraB := loraBVar.ValueGraph(g)

	// Ensure correct shapes
	if loraA.Shape().Dimensions[0] != inputDim || loraA.Shape().Dimensions[1] != rank {
		loraA = Reshape(loraA, inputDim, rank)
	}
	if loraB.Shape().Dimensions[0] != rank || loraB.Shape().Dimensions[1] != outputDim {
		loraB = Reshape(loraB, rank, outputDim)
	}

	// Compute A @ B
	deltaW := Dot(loraA, loraB)

	// Scale by alpha/rank
	scale := a.config.Scale()
	if scale != 1.0 {
		deltaW = MulScalar(deltaW, scale)
	}

	// Ensure matching dtype
	if deltaW.DType() != dtype {
		deltaW = ConvertDType(deltaW, dtype)
	}

	// Merge: W + scale * (A @ B)
	merged := Add(baseWeights, deltaW)

	a.SetMerged(true)
	return merged
}

// CountParameters returns the number of trainable parameters in this adapter.
func (a *Adapter) CountParameters(ctx *context.Context) int64 {
	var count int64
	for v := range ctx.IterVariables() {
		if isLoRAParameter(v.Name()) && v.Trainable {
			count += int64(v.Shape().Size())
		}
	}
	return count
}

// FreezeBaseWeights freezes all non-LoRA weights in the context.
func (a *Adapter) FreezeBaseWeights(ctx *context.Context) {
	for v := range ctx.IterVariables() {
		name := v.Name()
		if isLoRAParameter(name) {
			v.SetTrainable(true)
		} else {
			v.SetTrainable(false)
		}
	}
}

// UnfreezeBaseWeights unfreezes all weights in the context.
func (a *Adapter) UnfreezeBaseWeights(ctx *context.Context) {
	for v := range ctx.IterVariables() {
		v.SetTrainable(true)
	}
}

// applyLoRA computes the LoRA adaptation: scale * (x @ A @ B)
func applyLoRA(ctx *context.Context, input *Node, config *Config, inputDim, outputDim int) *Node {
	g := input.Graph()
	dtype := input.DType()
	rank := config.Rank()

	// LoRA A matrix: [inputDim, rank]
	// Initialized with Kaiming uniform (like the original paper)
	loraACtx := ctx.WithInitializer(kaimingUniformInitializer(ctx, inputDim))
	loraAVar := loraACtx.VariableWithShape("lora_A", shapes.Make(dtype, inputDim, rank))
	loraA := loraAVar.ValueGraph(g)

	// LoRA B matrix: [rank, outputDim]
	// Initialized with zeros (so LoRA starts as identity)
	loraBCtx := ctx.WithInitializer(initializers.Zero)
	loraBVar := loraBCtx.VariableWithShape("lora_B", shapes.Make(dtype, rank, outputDim))
	loraB := loraBVar.ValueGraph(g)

	// Compute LoRA output: x @ A @ B
	hidden := Dot(input, loraA) // [..., rank]

	// Apply dropout to hidden if configured
	if config.Dropout() > 0 && ctx.IsTraining(g) {
		keepProb := Scalar(g, dtype, 1.0-config.Dropout())
		mask := ctx.RandomBernoulli(keepProb, hidden.Shape())
		mask = ConvertDType(mask, dtype)
		// Scale to maintain expected value
		scaleVal := Scalar(g, dtype, 1.0/(1.0-config.Dropout()))
		hidden = Mul(Mul(hidden, mask), scaleVal)
	}

	output := Dot(hidden, loraB) // [..., outputDim]

	// Scale by alpha/rank
	scale := config.Scale()
	if scale != 1.0 {
		output = MulScalar(output, scale)
	}

	return output
}

// kaimingUniformInitializer creates a Kaiming uniform initializer for LoRA A matrix.
func kaimingUniformInitializer(ctx *context.Context, fanIn int) initializers.VariableInitializer {
	return func(g *Graph, shape shapes.Shape) *Node {
		if !shape.DType.IsFloat() {
			return Zeros(g, shape)
		}
		// Kaiming uniform: U(-bound, bound) where bound = sqrt(3/fan_in)
		bound := math.Sqrt(3.0 / float64(fanIn))
		values := ctx.RandomUniform(g, shape)
		values = MulScalar(values, 2*bound)
		values = AddScalar(values, -bound)
		return values
	}
}

// isLoRAParameter checks if a variable name belongs to LoRA.
func isLoRAParameter(name string) bool {
	return strings.Contains(name, "lora_A") || strings.Contains(name, "lora_B")
}

// Dense applies a dense layer with LoRA adaptation.
//
// This creates:
//   - A frozen base weight matrix W of shape [inputDim, outputDim]
//   - A trainable "lora_A" matrix of shape [inputDim, rank]
//   - A trainable "lora_B" matrix of shape [rank, outputDim]
//
// The output is: y = Wx + (scale * dropout(x @ A) @ B)
//
// Parameters:
//   - ctx: Context for variable management
//   - input: Input tensor of shape [..., inputDim]
//   - config: LoRA configuration
//   - outputDim: Output dimension
//
// Returns tensor of shape [..., outputDim]
func Dense(ctx *context.Context, input *Node, config *Config, outputDim int) *Node {
	return DenseWithBias(ctx, input, config, false, outputDim)
}

// DenseWithBias applies a dense layer with LoRA adaptation and optional bias.
func DenseWithBias(ctx *context.Context, input *Node, config *Config, useBias bool, outputDim int) *Node {
	ctx = ctx.In("lora_dense")
	g := input.Graph()
	dtype := input.DType()
	inputDim := input.Shape().Dimensions[input.Rank()-1]

	// Base weights (frozen during LoRA training)
	baseWeightsVar := ctx.VariableWithShape("weights", shapes.Make(dtype, inputDim, outputDim))
	baseWeights := baseWeightsVar.ValueGraph(g)

	// Compute base output: x @ W
	output := Dot(input, baseWeights)

	// Add LoRA adaptation if enabled
	if config != nil && config.Enabled() && config.Rank() > 0 {
		loraOutput := applyLoRA(ctx, input, config, inputDim, outputDim)
		output = Add(output, loraOutput)
	}

	// Add bias if requested
	if useBias {
		biasVar := ctx.VariableWithShape("bias", shapes.Make(dtype, outputDim))
		bias := biasVar.ValueGraph(g)
		// Expand bias to match output shape for broadcasting
		expandedBiasShape := output.Shape().Clone()
		for ii := range expandedBiasShape.Dimensions[:output.Rank()-1] {
			expandedBiasShape.Dimensions[ii] = 1
		}
		expandedBias := ReshapeWithShape(bias, expandedBiasShape)
		output = Add(output, expandedBias)
	}

	return output
}
