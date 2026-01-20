// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package adalora implements AdaLoRA (Adaptive Low-Rank Adaptation).
//
// AdaLoRA uses SVD-like parameterization with dynamic rank allocation:
//
//	DeltaW = P @ diag(Lambda) @ Q
//
// Where:
//   - P [inputDim, rank]: Left singular vectors
//   - Lambda [rank]: Singular values (used as importance scores)
//   - Q [rank, outputDim]: Right singular vectors
//
// Key features:
//   - Importance-based rank allocation: singular values indicate importance
//   - Dynamic pruning: low-importance components are masked out
//   - Orthogonal regularization: encourages P and Q to stay orthogonal
//
// Reference: "AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning"
// https://arxiv.org/abs/2303.10512
//
// Basic usage:
//
//	config := adalora.NewConfig().SetInitialRank(12).SetTargetRank(8)
//	adapter, _ := adalora.New("adalora_finetune", config)
//	output := adalora.Dense(ctx, input, config, outputDim)
package adalora

import (
	"math"
	"strings"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/adapters"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/initializers"
)

// Adapter implements the adapters.Adapter interface for AdaLoRA.
type Adapter struct {
	adapters.BaseAdapter
	config *Config
}

// New creates a new AdaLoRA adapter with the given name and configuration.
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
	return adapters.AdapterTypeAdaLoRA
}

// Config returns the adapter configuration.
func (a *Adapter) Config() adapters.AdapterConfig {
	return a.config
}

// Apply adds the AdaLoRA transformation to the base output.
//
// AdaLoRA formula: output = base_output + scale * input @ P @ diag(Lambda * mask) @ Q
//
// Parameters:
//   - ctx: Context scoped to the current layer
//   - input: The input to the layer [batch, ..., inputDim]
//   - baseOutput: The output from the base layer
//   - inputDim, outputDim: Dimensions for parameter creation
//
// Returns: AdaLoRA adapted output
func (a *Adapter) Apply(ctx *context.Context, input, baseOutput *Node, inputDim, outputDim int) *Node {
	if !a.config.Enabled() || a.IsMerged() {
		return baseOutput
	}

	return applyAdaLoRA(ctx, input, baseOutput, a.config, inputDim, outputDim)
}

// Merge merges AdaLoRA weights into base weights permanently.
//
// For AdaLoRA: W_merged = W + scale * P @ diag(Lambda * mask) @ Q
//
// Parameters:
//   - ctx: Context for accessing AdaLoRA variables
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
	rank := a.config.InitialRank()

	// Get AdaLoRA variables
	pVar := ctx.GetVariableByScopeAndName(ctx.Scope(), "adalora_P")
	lambdaVar := ctx.GetVariableByScopeAndName(ctx.Scope(), "adalora_lambda")
	qVar := ctx.GetVariableByScopeAndName(ctx.Scope(), "adalora_Q")
	maskVar := ctx.GetVariableByScopeAndName(ctx.Scope(), "adalora_mask")

	if pVar == nil || lambdaVar == nil || qVar == nil {
		return baseWeights
	}

	P := pVar.ValueGraph(g)
	lambda := lambdaVar.ValueGraph(g)
	Q := qVar.ValueGraph(g)

	// Apply mask if present
	if maskVar != nil {
		mask := maskVar.ValueGraph(g)
		if mask.DType() != dtype {
			mask = ConvertDType(mask, dtype)
		}
		lambda = Mul(lambda, mask)
	}

	// Ensure correct shapes
	if P.Shape().Dimensions[0] != inputDim || P.Shape().Dimensions[1] != rank {
		P = Reshape(P, inputDim, rank)
	}
	if Q.Shape().Dimensions[0] != rank || Q.Shape().Dimensions[1] != outputDim {
		Q = Reshape(Q, rank, outputDim)
	}

	// Compute P @ diag(Lambda) @ Q
	// First: P @ diag(Lambda) = P * Lambda (broadcast)
	lambdaExpanded := Reshape(lambda, 1, rank)
	pScaled := Mul(P, lambdaExpanded) // [inputDim, rank]

	// Then: pScaled @ Q
	deltaW := Dot(pScaled, Q) // [inputDim, outputDim]

	// Scale by alpha/rank
	scale := a.config.Scale()
	if scale != 1.0 {
		deltaW = MulScalar(deltaW, scale)
	}

	// Ensure matching dtype
	if deltaW.DType() != dtype {
		deltaW = ConvertDType(deltaW, dtype)
	}

	merged := Add(baseWeights, deltaW)

	a.SetMerged(true)
	return merged
}

// CountParameters returns the number of trainable parameters in this adapter.
func (a *Adapter) CountParameters(ctx *context.Context) int64 {
	var count int64
	for v := range ctx.IterVariables() {
		if isAdaLoRAParameter(v.Name()) && v.Trainable {
			count += int64(v.Shape().Size())
		}
	}
	return count
}

// FreezeBaseWeights freezes all non-AdaLoRA weights in the context.
func (a *Adapter) FreezeBaseWeights(ctx *context.Context) {
	for v := range ctx.IterVariables() {
		name := v.Name()
		if isAdaLoRAParameter(name) {
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

// applyAdaLoRA computes the AdaLoRA adaptation.
func applyAdaLoRA(ctx *context.Context, input, baseOutput *Node, config *Config, inputDim, outputDim int) *Node {
	g := input.Graph()
	dtype := input.DType()
	rank := config.InitialRank()

	// P matrix: [inputDim, rank] - left singular vectors
	pCtx := ctx.WithInitializer(orthogonalInitializer(ctx))
	pVar := pCtx.VariableWithShape("adalora_P", shapes.Make(dtype, inputDim, rank))
	P := pVar.ValueGraph(g)

	// Lambda vector: [rank] - singular values (importance scores)
	lambdaCtx := ctx.WithInitializer(initializers.One)
	lambdaVar := lambdaCtx.VariableWithShape("adalora_lambda", shapes.Make(dtype, rank))
	lambda := lambdaVar.ValueGraph(g)

	// Q matrix: [rank, outputDim] - right singular vectors
	qCtx := ctx.WithInitializer(initializers.Zero)
	qVar := qCtx.VariableWithShape("adalora_Q", shapes.Make(dtype, rank, outputDim))
	Q := qVar.ValueGraph(g)

	// Mask: [rank] - binary mask for pruning (1 = active, 0 = pruned)
	// Initialized to all ones (no pruning initially)
	maskCtx := ctx.WithInitializer(initializers.One)
	maskVar := maskCtx.VariableWithShape("adalora_mask", shapes.Make(dtype, rank))
	mask := maskVar.ValueGraph(g)
	maskVar.SetTrainable(false) // Mask is not trained via gradients

	// Importance scores (EMA): [rank] - accumulated importance
	importanceCtx := ctx.WithInitializer(initializers.Zero)
	importanceVar := importanceCtx.VariableWithShape("adalora_importance", shapes.Make(dtype, rank))
	_ = importanceVar // Used during budget allocation
	importanceVar.SetTrainable(false)

	// Apply mask to lambda
	maskedLambda := Mul(lambda, mask)

	// Compute: input @ P @ diag(maskedLambda) @ Q
	hidden := Dot(input, P) // [..., rank]

	// Apply dropout if configured
	if config.Dropout() > 0 && ctx.IsTraining(g) {
		keepProb := Scalar(g, dtype, 1.0-config.Dropout())
		dropMask := ctx.RandomBernoulli(keepProb, hidden.Shape())
		dropMask = ConvertDType(dropMask, dtype)
		scaleVal := Scalar(g, dtype, 1.0/(1.0-config.Dropout()))
		hidden = Mul(Mul(hidden, dropMask), scaleVal)
	}

	// Scale by lambda: hidden * maskedLambda (broadcast)
	// Expand maskedLambda to match hidden's rank for broadcasting
	maskedLambdaExpanded := ExpandLeftToRank(maskedLambda, hidden.Rank())
	hidden = Mul(hidden, maskedLambdaExpanded) // [..., rank]

	// Final projection: hidden @ Q
	adaLoRAOutput := Dot(hidden, Q) // [..., outputDim]

	// Scale by alpha/rank
	scale := config.Scale()
	if scale != 1.0 {
		adaLoRAOutput = MulScalar(adaLoRAOutput, scale)
	}

	// Combined output
	output := Add(baseOutput, adaLoRAOutput)

	return output
}

// orthogonalInitializer creates an orthogonal initializer for P matrix.
func orthogonalInitializer(ctx *context.Context) initializers.VariableInitializer {
	return func(g *Graph, shape shapes.Shape) *Node {
		if !shape.DType.IsFloat() || shape.Rank() != 2 {
			return Zeros(g, shape)
		}

		// Generate random values and orthogonalize via Gram-Schmidt approximation
		// For simplicity, we use scaled random initialization
		// Full orthogonalization would require SVD which is complex in graph form
		rows := shape.Dimensions[0]
		cols := shape.Dimensions[1]

		// Xavier/Glorot initialization as approximation
		scale := math.Sqrt(2.0 / float64(rows+cols))
		values := ctx.RandomNormal(g, shape)
		return MulScalar(values, scale)
	}
}

// isAdaLoRAParameter checks if a variable name belongs to AdaLoRA.
func isAdaLoRAParameter(name string) bool {
	return strings.Contains(name, "adalora_P") ||
		strings.Contains(name, "adalora_lambda") ||
		strings.Contains(name, "adalora_Q")
	// Note: mask and importance are not trainable
}

// Dense applies a dense layer with AdaLoRA adaptation.
//
// This creates:
//   - A frozen base weight matrix W of shape [inputDim, outputDim]
//   - A trainable "adalora_P" matrix of shape [inputDim, rank]
//   - A trainable "adalora_lambda" vector of shape [rank]
//   - A trainable "adalora_Q" matrix of shape [rank, outputDim]
//   - A non-trainable "adalora_mask" vector of shape [rank]
//
// The output is: y = x @ W + scale * x @ P @ diag(Lambda * mask) @ Q
//
// Parameters:
//   - ctx: Context for variable management
//   - input: Input tensor of shape [..., inputDim]
//   - config: AdaLoRA configuration
//   - outputDim: Output dimension
//
// Returns tensor of shape [..., outputDim]
func Dense(ctx *context.Context, input *Node, config *Config, outputDim int) *Node {
	return DenseWithBias(ctx, input, config, false, outputDim)
}

// DenseWithBias applies a dense layer with AdaLoRA adaptation and optional bias.
func DenseWithBias(ctx *context.Context, input *Node, config *Config, useBias bool, outputDim int) *Node {
	ctx = ctx.In("adalora_dense")
	g := input.Graph()
	dtype := input.DType()
	inputDim := input.Shape().Dimensions[input.Rank()-1]

	// Base weights (frozen during AdaLoRA training)
	baseWeightsVar := ctx.VariableWithShape("weights", shapes.Make(dtype, inputDim, outputDim))
	baseWeights := baseWeightsVar.ValueGraph(g)

	// Base output
	baseOutput := Dot(input, baseWeights)

	var output *Node

	if config != nil && config.Enabled() && config.InitialRank() > 0 {
		// Apply AdaLoRA
		output = applyAdaLoRA(ctx, input, baseOutput, config, inputDim, outputDim)
	} else {
		output = baseOutput
	}

	// Add bias if requested
	if useBias {
		biasVar := ctx.VariableWithShape("bias", shapes.Make(dtype, outputDim))
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

// OrthogonalRegularization computes the orthogonal regularization loss for AdaLoRA.
//
// Loss = ||P^T @ P - I||_F^2 + ||Q @ Q^T - I||_F^2
//
// This encourages P and Q to maintain orthogonality.
//
// Parameters:
//   - ctx: Context containing AdaLoRA variables
//   - g: The computation graph
//
// Returns: Regularization loss scalar
func OrthogonalRegularization(ctx *context.Context, g *Graph) *Node {
	dtype := dtypes.Float32
	totalLoss := Scalar(g, dtype, 0.0)

	for v := range ctx.IterVariables() {
		name := v.Name()
		if strings.Contains(name, "adalora_P") {
			P := v.ValueGraph(g)
			if P.DType() != dtype {
				P = ConvertDType(P, dtype)
			}
			// ||P^T @ P - I||_F^2
			rank := P.Shape().Dimensions[1]
			PtP := Dot(TransposeAllDims(P), P)
			I := identityMatrix(g, dtype, rank)
			diff := Sub(PtP, I)
			loss := ReduceSum(Mul(diff, diff))
			totalLoss = Add(totalLoss, loss)
		}
		if strings.Contains(name, "adalora_Q") {
			Q := v.ValueGraph(g)
			if Q.DType() != dtype {
				Q = ConvertDType(Q, dtype)
			}
			// ||Q @ Q^T - I||_F^2
			rank := Q.Shape().Dimensions[0]
			QQt := Dot(Q, TransposeAllDims(Q))
			I := identityMatrix(g, dtype, rank)
			diff := Sub(QQt, I)
			loss := ReduceSum(Mul(diff, diff))
			totalLoss = Add(totalLoss, loss)
		}
	}

	return totalLoss
}

// identityMatrix creates an identity matrix of given size.
func identityMatrix(g *Graph, dtype dtypes.DType, size int) *Node {
	// Use DiagonalWithValue to create an identity matrix
	one := Scalar(g, dtype, 1.0)
	return DiagonalWithValue(one, size)
}
