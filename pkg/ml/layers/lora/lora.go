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
//	// Create LoRA config
//	config := lora.NewConfig().Rank(8).Alpha(16.0)
//
//	// Apply LoRA to a dense layer
//	output := lora.Dense(ctx, input, config, outputDim)
//
//	// Or wrap an existing weight matrix
//	output := lora.Apply(ctx, input, baseWeights, config)
package lora

import (
	"math"
	"strings"

	"github.com/gomlx/gomlx/internal/exceptions"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/initializers"
)

// Context parameter names for LoRA configuration.
const (
	// ParamRank is the rank of the low-rank decomposition.
	ParamRank = "lora_rank"

	// ParamAlpha is the scaling factor alpha. The actual scale is alpha/rank.
	ParamAlpha = "lora_alpha"

	// ParamDropout is the dropout rate applied to the LoRA path.
	ParamDropout = "lora_dropout"

	// ParamEnabled controls whether LoRA is applied. When false, uses base weights only.
	ParamEnabled = "lora_enabled"
)

// Config holds the configuration for LoRA layers.
type Config struct {
	// Rank is the rank of the low-rank decomposition (r in the paper).
	// Higher rank = more parameters = more capacity.
	// Typical values: 4, 8, 16, 32, 64.
	Rank int

	// Alpha is the scaling factor. The LoRA output is scaled by alpha/rank.
	// This allows tuning the magnitude of LoRA updates relative to pretrained weights.
	// Typical values: 16, 32, or same as rank.
	Alpha float64

	// Dropout is the dropout rate applied to the LoRA path (before scaling).
	// Set to 0 to disable.
	Dropout float64

	// Enabled controls whether LoRA adaptation is active.
	// When false, only base weights are used.
	Enabled bool

	// TargetModules specifies which module types to apply LoRA to.
	// Common targets: "query", "key", "value", "output", "dense".
	// Empty means apply to all compatible layers.
	//
	// Note: This field is currently reserved for future use and has no effect.
	// LoRA is applied wherever Dense() or Apply() is called explicitly.
	TargetModules []string
}

// NewConfig creates a new LoRA configuration with sensible defaults.
func NewConfig() *Config {
	return &Config{
		Rank:    8,
		Alpha:   16.0,
		Dropout: 0.0,
		Enabled: true,
	}
}

// FromContext creates a Config by reading values from context parameters.
func FromContext(ctx *context.Context) *Config {
	return &Config{
		Rank:    context.GetParamOr(ctx, ParamRank, 8),
		Alpha:   context.GetParamOr(ctx, ParamAlpha, 16.0),
		Dropout: context.GetParamOr(ctx, ParamDropout, 0.0),
		Enabled: context.GetParamOr(ctx, ParamEnabled, true),
	}
}

// SetRank sets the rank of the decomposition.
func (c *Config) SetRank(rank int) *Config {
	if rank <= 0 {
		exceptions.Panicf("LoRA rank must be > 0, got %d", rank)
	}
	c.Rank = rank
	return c
}

// SetAlpha sets the scaling factor alpha.
func (c *Config) SetAlpha(alpha float64) *Config {
	c.Alpha = alpha
	return c
}

// SetDropout sets the dropout rate.
func (c *Config) SetDropout(dropout float64) *Config {
	if dropout < 0 || dropout >= 1 {
		exceptions.Panicf("LoRA dropout must be in [0, 1), got %f", dropout)
	}
	c.Dropout = dropout
	return c
}

// SetEnabled sets whether LoRA is active.
func (c *Config) SetEnabled(enabled bool) *Config {
	c.Enabled = enabled
	return c
}

// SetTargetModules sets the target modules for LoRA.
func (c *Config) SetTargetModules(modules ...string) *Config {
	c.TargetModules = modules
	return c
}

// Scale returns the scaling factor (alpha / rank).
func (c *Config) Scale() float64 {
	if c.Rank == 0 {
		return 0
	}
	return c.Alpha / float64(c.Rank)
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
	if config != nil && config.Enabled && config.Rank > 0 {
		loraOutput := applyLoRA(ctx, input, config, inputDim, outputDim)
		output = Add(output, loraOutput)
	}

	// Add bias if requested
	if useBias {
		biasVar := ctx.VariableWithShape("bias", shapes.Make(dtype, outputDim))
		bias := biasVar.ValueGraph(g)
		output = Add(output, bias)
	}

	return output
}

// Apply adds LoRA adaptation to an existing linear transformation.
//
// This is useful when you have pre-trained weights and want to add LoRA
// without replacing the entire layer.
//
// Parameters:
//   - ctx: Context for variable management
//   - input: Input tensor of shape [..., inputDim]
//   - baseOutput: Output from the base linear transformation (x @ W)
//   - config: LoRA configuration
//   - inputDim: Input dimension
//   - outputDim: Output dimension
//
// Returns: baseOutput + LoRA adaptation
func Apply(ctx *context.Context, input, baseOutput *Node, config *Config, inputDim, outputDim int) *Node {
	if config == nil || !config.Enabled || config.Rank <= 0 {
		return baseOutput
	}

	ctx = ctx.In("lora")
	loraOutput := applyLoRA(ctx, input, config, inputDim, outputDim)
	return Add(baseOutput, loraOutput)
}

// applyLoRA computes the LoRA adaptation: scale * (x @ A @ B)
func applyLoRA(ctx *context.Context, input *Node, config *Config, inputDim, outputDim int) *Node {
	g := input.Graph()
	dtype := input.DType()
	rank := config.Rank

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
	if config.Dropout > 0 && ctx.IsTraining(g) {
		// RandomBernoulli takes (prob *Node, shape) where prob is the probability of 1
		keepProb := Scalar(g, dtype, 1.0-config.Dropout)
		mask := ctx.RandomBernoulli(keepProb, hidden.Shape())
		mask = ConvertDType(mask, dtype)
		// Scale to maintain expected value
		scaleVal := Scalar(g, dtype, 1.0/(1.0-config.Dropout))
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
// Uses the standard deviation of 1/sqrt(fan_in).
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

// FreezeBaseWeights freezes all non-LoRA weights in the context.
// This is typically called after loading pre-trained weights.
func FreezeBaseWeights(ctx *context.Context) {
	for v := range ctx.IterVariables() {
		name := v.Name()
		// Keep LoRA parameters trainable
		if isLoRAParameter(name) {
			v.SetTrainable(true)
		} else {
			v.SetTrainable(false)
		}
	}
}

// UnfreezeBaseWeights unfreezes all weights in the context.
func UnfreezeBaseWeights(ctx *context.Context) {
	for v := range ctx.IterVariables() {
		v.SetTrainable(true)
	}
}

// isLoRAParameter checks if a variable name belongs to LoRA.
func isLoRAParameter(name string) bool {
	// LoRA parameters contain "lora_A" or "lora_B" in their path
	return strings.Contains(name, "lora_A") || strings.Contains(name, "lora_B")
}

// CountLoRAParameters counts the number of trainable LoRA parameters.
func CountLoRAParameters(ctx *context.Context) int64 {
	var count int64
	for v := range ctx.IterVariables() {
		if isLoRAParameter(v.Name()) && v.Trainable {
			count += int64(v.Shape().Size())
		}
	}
	return count
}

// CountBaseParameters counts the number of frozen base parameters.
func CountBaseParameters(ctx *context.Context) int64 {
	var count int64
	for v := range ctx.IterVariables() {
		if !isLoRAParameter(v.Name()) && !v.Trainable {
			count += int64(v.Shape().Size())
		}
	}
	return count
}

// MergeLoRAWeights merges LoRA weights into base weights permanently.
// After merging, the model can run without LoRA overhead.
//
// This modifies the base weight tensors in place:
//
//	W_new = W + scale * (A @ B)
//
// Note: This is a destructive operation. The LoRA weights should be
// removed or zeroed after merging.
//
// TODO: This function is currently a stub and does nothing. To merge LoRA weights
// into base weights, you must manually compute the merged weights using graph operations:
//
//	scale := config.Scale()
//	loraA := ctx.Variable("lora_A").Value()
//	loraB := ctx.Variable("lora_B").Value()
//	baseWeights := ctx.Variable("weights").Value()
//	mergedWeights := Add(baseWeights, MulScalar(Dot(loraA, loraB), scale))
//
// Then save the merged weights and load them into a model without LoRA layers.
func MergeLoRAWeights(ctx *context.Context, config *Config) error {
	// This would require tensor manipulation outside the graph
	// which is more complex. For now, we provide the formula
	// and leave implementation to the user.
	//
	// In practice, users would:
	// 1. Execute a graph that computes W + scale * (A @ B)
	// 2. Save the merged weights
	// 3. Load into a model without LoRA
	return nil
}
