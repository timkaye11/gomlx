// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package qlora implements QLoRA (Quantized Low-Rank Adaptation) for memory-efficient fine-tuning.
//
// QLoRA combines 4-bit NormalFloat (NF4) quantization of base model weights with
// full-precision LoRA adapters, enabling fine-tuning of large language models on
// consumer hardware with limited GPU memory.
//
// Key features:
//   - Base weights quantized to 4-bit NF4 format (~4x memory reduction)
//   - LoRA adapters (A, B matrices) in full precision (BFloat16/Float16)
//   - Per-group scaling for better quantization accuracy
//   - Dequantize-on-the-fly during forward pass
//
// Reference: "QLoRA: Efficient Finetuning of Quantized LLMs"
// https://arxiv.org/abs/2305.14314
//
// Basic usage:
//
//	config := qlora.NewConfig().SetRank(16).SetAlpha(32)
//	adapter, _ := qlora.New("qlora_finetune", config)
//	output := adapter.Apply(ctx, input, baseOutput, inputDim, outputDim)
package qlora

import (
	"math"
	"strings"

	"github.com/gomlx/gomlx/pkg/ml/adapters"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/initializers"
	"github.com/gomlx/gomlx/pkg/ml/quantization"
)

// Adapter implements the adapters.Adapter interface for QLoRA.
type Adapter struct {
	adapters.BaseAdapter
	config *Config
}

// New creates a new QLoRA adapter with the given name and configuration.
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
	return adapters.AdapterTypeQLoRA
}

// Config returns the adapter configuration.
func (a *Adapter) Config() adapters.AdapterConfig {
	return a.config
}

// Apply adds the QLoRA transformation to the base output.
// Note: For QLoRA, the base output should come from dequantized quantized weights.
// This method adds the LoRA adaptation on top.
//
// The QLoRA update is: dequant(W_q) @ x + scale * (x @ A @ B)
// where W_q is the quantized base weights.
//
// Parameters:
//   - ctx: Context scoped to the current layer
//   - input: The input to the layer [batch, ..., inputDim]
//   - baseOutput: The output from the dequantized base layer
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

// Merge merges QLoRA weights into base weights.
// Note: This produces a merged float weight that is no longer quantized.
// To get a quantized merged weight, re-quantize the result.
//
// For QLoRA: W_merged = dequant(W_q) + scale * (A @ B)
//
// Parameters:
//   - ctx: Context for accessing variables
//   - baseWeights: The dequantized base weight matrix [inputDim, outputDim]
//
// Returns: The merged weight matrix (in compute dtype, not quantized)
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
		if isQLoRAParameter(v.Name()) && v.Trainable {
			count += int64(v.Shape().Size())
		}
	}
	return count
}

// FreezeBaseWeights freezes all non-QLoRA weights in the context.
func (a *Adapter) FreezeBaseWeights(ctx *context.Context) {
	for v := range ctx.IterVariables() {
		name := v.Name()
		if isQLoRAParameter(name) {
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
	dtype := config.QuantConfig.ComputeDType
	rank := config.Rank()

	// Ensure input is in compute dtype
	if input.DType() != dtype {
		input = ConvertDType(input, dtype)
	}

	// LoRA A matrix: [inputDim, rank]
	// Initialized with Kaiming uniform
	loraACtx := ctx.WithInitializer(kaimingUniformInitializer(ctx, inputDim))
	loraAVar := loraACtx.VariableWithShape("lora_A", shapes.Make(dtype, inputDim, rank))
	loraA := loraAVar.ValueGraph(g)

	// LoRA B matrix: [rank, outputDim]
	// Initialized with zeros (so QLoRA starts as quantized base)
	loraBCtx := ctx.WithInitializer(initializers.Zero)
	loraBVar := loraBCtx.VariableWithShape("lora_B", shapes.Make(dtype, rank, outputDim))
	loraB := loraBVar.ValueGraph(g)

	// Compute LoRA output: x @ A @ B
	hidden := Dot(input, loraA) // [..., rank]

	// Apply dropout if configured
	if config.Dropout() > 0 && ctx.IsTraining(g) {
		keepProb := Scalar(g, dtype, 1.0-config.Dropout())
		mask := ctx.RandomBernoulli(keepProb, hidden.Shape())
		mask = ConvertDType(mask, dtype)
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
		bound := math.Sqrt(3.0 / float64(fanIn))
		values := ctx.RandomUniform(g, shape)
		values = MulScalar(values, 2*bound)
		values = AddScalar(values, -bound)
		return values
	}
}

// isQLoRAParameter checks if a variable name belongs to QLoRA.
func isQLoRAParameter(name string) bool {
	return strings.Contains(name, "lora_A") || strings.Contains(name, "lora_B")
}

// Dense applies a dense layer with QLoRA adaptation.
//
// This creates:
//   - Quantized base weights W_q (packed uint8 + scales)
//   - Trainable LoRA matrices A [inputDim, rank] and B [rank, outputDim]
//
// The output is: dequant(W_q) @ x + scale * (x @ A @ B)
//
// Parameters:
//   - ctx: Context for variable management
//   - input: Input tensor of shape [..., inputDim]
//   - config: QLoRA configuration
//   - outputDim: Output dimension
//
// Returns tensor of shape [..., outputDim]
func Dense(ctx *context.Context, input *Node, config *Config, outputDim int) *Node {
	return DenseWithBias(ctx, input, config, false, outputDim)
}

// DenseWithBias applies a dense layer with QLoRA adaptation and optional bias.
//
// Note: For QLoRA, the base weights must be pre-quantized and loaded.
// This function expects the quantized weights to already exist as variables.
// If they don't exist, it falls back to creating float weights (for initialization).
func DenseWithBias(ctx *context.Context, input *Node, config *Config, useBias bool, outputDim int) *Node {
	ctx = ctx.In("qlora_dense")
	g := input.Graph()
	computeDType := config.QuantConfig.ComputeDType
	inputDim := input.Shape().Dimensions[input.Rank()-1]

	// Ensure input is in compute dtype
	if input.DType() != computeDType {
		input = ConvertDType(input, computeDType)
	}

	// Try to get pre-quantized weights
	packedVar := ctx.GetVariableByScopeAndName(ctx.Scope(), "weights_packed")
	scalesVar := ctx.GetVariableByScopeAndName(ctx.Scope(), "weights_scales")

	var output *Node

	if packedVar != nil && scalesVar != nil {
		// Use quantized weights
		packedWeights := packedVar.ValueGraph(g)
		scales := scalesVar.ValueGraph(g)

		// Dequantize and compute base output
		weightShape := shapes.Make(computeDType, inputDim, outputDim)
		dequantizedWeights := quantization.Dequantize(packedWeights, scales, weightShape, config.QuantConfig)
		output = Dot(input, dequantizedWeights)
	} else {
		// Fallback to float weights (for testing or initialization)
		baseWeightsVar := ctx.VariableWithShape("weights", shapes.Make(computeDType, inputDim, outputDim))
		baseWeights := baseWeightsVar.ValueGraph(g)
		output = Dot(input, baseWeights)
	}

	// Add LoRA adaptation if enabled
	if config != nil && config.Enabled() && config.Rank() > 0 {
		loraOutput := applyLoRA(ctx, input, config, inputDim, outputDim)
		output = Add(output, loraOutput)
	}

	// Add bias if requested
	if useBias {
		biasVar := ctx.VariableWithShape("bias", shapes.Make(computeDType, outputDim))
		bias := biasVar.ValueGraph(g)
		expandedBiasShape := output.Shape().Clone()
		for ii := range expandedBiasShape.Dimensions[:output.Rank()-1] {
			expandedBiasShape.Dimensions[ii] = 1
		}
		expandedBias := ReshapeWithShape(bias, expandedBiasShape)
		output = Add(output, expandedBias)
	}

	return output
}

// QuantizeBaseWeights quantizes float weights and stores them as QLoRA-compatible variables.
//
// This function takes a float tensor and creates quantized weight variables
// that can be used with QLoRA layers.
//
// Parameters:
//   - ctx: Context for variable management
//   - weights: Float tensor to quantize [inputDim, outputDim]
//   - config: QLoRA configuration
//
// Returns: The QuantizedWeight structure
func QuantizeBaseWeights(ctx *context.Context, weights *tensors.Tensor, config *Config) (*quantization.QuantizedWeight, error) {
	return quantization.Quantize(weights, config.QuantConfig)
}

// LoadQuantizedWeights loads pre-quantized weights into context variables.
//
// This function stores packed weights and scales as context variables
// that will be used by QLoRA dense layers.
//
// Parameters:
//   - ctx: Context for variable management
//   - qw: Quantized weight structure
//   - name: Variable name prefix (e.g., "layer1")
func LoadQuantizedWeights(ctx *context.Context, qw *quantization.QuantizedWeight, name string) {
	ctx = ctx.In(name)

	// Store packed data
	ctx.VariableWithValue("weights_packed", qw.PackedData)

	// Store scales
	ctx.VariableWithValue("weights_scales", qw.Scales)

	// Mark as non-trainable (base weights are frozen)
	if packedVar := ctx.GetVariableByScopeAndName(ctx.Scope(), "weights_packed"); packedVar != nil {
		packedVar.SetTrainable(false)
	}
	if scalesVar := ctx.GetVariableByScopeAndName(ctx.Scope(), "weights_scales"); scalesVar != nil {
		scalesVar.SetTrainable(false)
	}
}
