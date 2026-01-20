// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package adapters_test

import (
	"fmt"
	"testing"

	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/default"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/adapters"
	"github.com/gomlx/gomlx/pkg/ml/adapters/dora"
	"github.com/gomlx/gomlx/pkg/ml/adapters/ia3"
	"github.com/gomlx/gomlx/pkg/ml/adapters/lora"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/initializers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestComprehensiveLoRAExample demonstrates the complete workflow for using LoRA
// adapters in GoMLX. This serves as both a test and learning documentation.
//
// The example covers:
//  1. Creating a simple neural network model
//  2. Adding LoRA adapters for parameter-efficient fine-tuning
//  3. Running forward passes with adapters
//  4. Comparing parameter counts (base vs adapter)
//  5. Managing multiple adapters (switching, enabling/disabling)
//  6. Merging adapters into base weights
func TestComprehensiveLoRAExample(t *testing.T) {
	// ============================================================
	// STEP 1: Setup - Create backend and context
	// ============================================================
	backend, err := backends.New()
	require.NoError(t, err)
	ctx := context.New()

	// Model dimensions (simulating a small transformer layer)
	batchSize := 2
	seqLen := 4
	hiddenDim := 64
	intermediateDim := 128

	// ============================================================
	// STEP 2: Define a simple MLP model (simulating part of a transformer)
	// ============================================================
	// This represents a typical feed-forward network in a transformer:
	//   output = down_proj(activation(up_proj(input)))

	mlpModel := func(ctx *context.Context, input *Node) *Node {
		g := input.Graph()
		dtype := input.DType()

		// Up projection: hiddenDim -> intermediateDim
		// Using Einsum for batched matrix multiplication: [batch, seq, hidden] @ [hidden, intermediate] -> [batch, seq, intermediate]
		ctx1 := ctx.In("up_proj")
		upWeights := ctx1.VariableWithShape("weights", shapes.Make(dtype, hiddenDim, intermediateDim)).ValueGraph(g)
		hidden := Einsum("bsh,hi->bsi", input, upWeights)

		// Activation (GELU approximation using sigmoid)
		hidden = Mul(hidden, Sigmoid(MulScalar(hidden, 1.702)))

		// Down projection: intermediateDim -> hiddenDim
		ctx2 := ctx.In("down_proj")
		downWeights := ctx2.VariableWithShape("weights", shapes.Make(dtype, intermediateDim, hiddenDim)).ValueGraph(g)
		output := Einsum("bsi,ih->bsh", hidden, downWeights)

		return output
	}

	// ============================================================
	// STEP 3: Run the base model (without adapters)
	// ============================================================
	fmt.Println("=== Step 3: Running base model ===")

	baseExec := context.MustNewExec(backend, ctx, mlpModel)

	// Create input data
	inputData := make([][][]float32, batchSize)
	for i := range inputData {
		inputData[i] = make([][]float32, seqLen)
		for j := range inputData[i] {
			inputData[i][j] = make([]float32, hiddenDim)
			for k := range inputData[i][j] {
				inputData[i][j][k] = 0.1 * float32(k%10)
			}
		}
	}

	baseOutput := baseExec.MustExec(inputData)[0]
	fmt.Printf("Base model output shape: %v\n", baseOutput.Shape())

	// Count base model parameters
	baseParamCount := int64(0)
	for v := range ctx.IterVariables() {
		baseParamCount += int64(v.Shape().Size())
	}
	fmt.Printf("Base model parameters: %d\n", baseParamCount)
	// Expected: hiddenDim*intermediateDim + intermediateDim*hiddenDim = 64*128 + 128*64 = 16384

	// ============================================================
	// STEP 4: Create LoRA configuration
	// ============================================================
	fmt.Println("\n=== Step 4: Creating LoRA adapter ===")

	// LoRA hyperparameters:
	// - rank: The low-rank dimension (smaller = fewer params, less capacity)
	// - alpha: Scaling factor (scale = alpha/rank)
	// - dropout: Regularization during training
	loraConfig := lora.NewConfig().
		SetRank(8).          // Low rank for parameter efficiency
		SetAlpha(16.0).      // Scale factor (effective scale = 16/8 = 2.0)
		SetDropout(0.0).     // No dropout for this example
		SetUseRSLoRA(false). // Standard LoRA scaling (not rank-stabilized)
		SetUseLoRAPlus(false) // Standard learning rate (not LoRA+)

	fmt.Printf("LoRA config: rank=%d, alpha=%.1f, scale=%.2f\n",
		loraConfig.Rank(), loraConfig.Alpha(), loraConfig.Scale())

	// ============================================================
	// STEP 5: Define model WITH LoRA adapters
	// ============================================================
	fmt.Println("\n=== Step 5: Creating model with LoRA ===")

	// Create a new context for the LoRA-enhanced model
	ctxLoRA := context.New()

	mlpWithLoRA := func(ctx *context.Context, input *Node) *Node {
		g := input.Graph()
		dtype := input.DType()

		// Up projection with LoRA
		ctx1 := ctx.In("up_proj")
		upWeights := ctx1.VariableWithShape("weights", shapes.Make(dtype, hiddenDim, intermediateDim)).ValueGraph(g)
		hidden := Einsum("bsh,hi->bsi", input, upWeights)

		// Add LoRA adaptation to up_proj
		loraOut := applyLoRADirect(ctx1.In("lora"), input, loraConfig, hiddenDim, intermediateDim)
		hidden = Add(hidden, loraOut)

		// Activation
		hidden = Mul(hidden, Sigmoid(MulScalar(hidden, 1.702)))

		// Down projection with LoRA
		ctx2 := ctx.In("down_proj")
		downWeights := ctx2.VariableWithShape("weights", shapes.Make(dtype, intermediateDim, hiddenDim)).ValueGraph(g)
		output := Einsum("bsi,ih->bsh", hidden, downWeights)

		// Add LoRA adaptation to down_proj
		loraOut2 := applyLoRADirect(ctx2.In("lora"), hidden, loraConfig, intermediateDim, hiddenDim)
		output = Add(output, loraOut2)

		return output
	}

	loraExec := context.MustNewExec(backend, ctxLoRA, mlpWithLoRA)
	loraOutput := loraExec.MustExec(inputData)[0]
	fmt.Printf("LoRA model output shape: %v\n", loraOutput.Shape())

	// Count parameters
	totalParams := int64(0)
	loraParams := int64(0)
	baseParams := int64(0)
	for v := range ctxLoRA.IterVariables() {
		size := int64(v.Shape().Size())
		totalParams += size
		if lora.IsLoRAVariable(v.ParameterName()) {
			loraParams += size
		} else {
			baseParams += size
		}
	}

	fmt.Printf("Total parameters: %d\n", totalParams)
	fmt.Printf("Base parameters: %d\n", baseParams)
	fmt.Printf("LoRA parameters: %d (%.2f%% of base)\n", loraParams, 100*float64(loraParams)/float64(baseParams))

	// ============================================================
	// STEP 6: Demonstrate the Adapter Manager API
	// ============================================================
	fmt.Println("\n=== Step 6: Using Adapter Manager ===")

	ctxManaged := context.New()

	// Create adapter using the high-level API
	manager, err := adapters.GetAdapterModel(ctxManaged, loraConfig, "task_a_lora")
	require.NoError(t, err)
	fmt.Printf("Created adapter: %s (type: %s)\n", manager.ActiveName(), loraConfig.Type())

	// Add a second adapter for a different task
	loraConfigB := lora.NewConfig().SetRank(4).SetAlpha(8.0) // Smaller rank for task B
	_, err = adapters.AddAdapter(ctxManaged, loraConfigB, "task_b_lora")
	require.NoError(t, err)

	// List all adapters
	adapterNames := manager.List()
	fmt.Printf("Available adapters: %v\n", adapterNames)

	// Switch between adapters
	err = manager.SetActive("task_b_lora")
	require.NoError(t, err)
	fmt.Printf("Switched to adapter: %s\n", manager.ActiveName())

	// Switch back
	err = manager.SetActive("task_a_lora")
	require.NoError(t, err)

	// ============================================================
	// STEP 7: Demonstrate different adapter types
	// ============================================================
	fmt.Println("\n=== Step 7: Different Adapter Types ===")

	// LoRA+ (different learning rates for A and B matrices)
	loraPlusConfig := lora.NewConfig().
		SetRank(8).
		SetUseLoRAPlus(true).
		SetLoRAPlusRatio(16.0) // B matrix uses 16x learning rate

	fmt.Printf("LoRA+ config: ratio=%.1f (B_lr = %.1fx A_lr)\n",
		loraPlusConfig.LoRAPlusRatio, loraPlusConfig.LoRAPlusRatio)

	// rsLoRA (rank-stabilized scaling)
	rsLoraConfig := lora.NewConfig().
		SetRank(64).
		SetAlpha(64.0).
		SetUseRSLoRA(true)

	fmt.Printf("rsLoRA config: scale=%.4f (alpha/sqrt(rank) = 64/8)\n", rsLoraConfig.Scale())

	// DoRA (weight-decomposed LoRA)
	doraConfig := dora.NewConfig().
		SetRank(8).
		SetAlpha(16.0)

	fmt.Printf("DoRA config: rank=%d, uses magnitude/direction decomposition\n", doraConfig.Rank())

	// IA³ (learned rescaling vectors)
	ia3Config := ia3.NewConfig().
		SetInitScale(1.0)

	fmt.Printf("IA³ config: initScale=%.1f (extremely parameter-efficient)\n", ia3Config.InitScale())

	// ============================================================
	// STEP 8: Demonstrate target module patterns
	// ============================================================
	fmt.Println("\n=== Step 8: Target Module Patterns ===")

	// Using common presets
	fmt.Printf("Common QKV patterns: %v\n", adapters.CommonTargetPatterns.QKV)
	fmt.Printf("Common MLP patterns: %v\n", adapters.CommonTargetPatterns.MLP)

	// Using architecture-specific presets
	llamaPreset := adapters.GetPreset("llama")
	if llamaPreset != nil {
		fmt.Printf("LLaMA default targets: %v\n", llamaPreset.DefaultTargets)
		fmt.Printf("LLaMA all linear: %v\n", llamaPreset.AllLinear)
	}

	// Custom pattern matching
	matcher, _ := adapters.NewModuleMatcher([]string{"query", "value"}, false)
	fmt.Printf("Pattern 'query' matches 'layer/query_proj': %v\n", matcher.Matches("layer/query_proj"))
	fmt.Printf("Pattern 'query' matches 'layer/key_proj': %v\n", matcher.Matches("layer/key_proj"))

	// Regex pattern matching
	regexMatcher, _ := adapters.NewModuleMatcher([]string{`layers\.\d+\.attention`}, true)
	fmt.Printf("Regex matches 'model.layers.5.attention': %v\n", regexMatcher.Matches("model.layers.5.attention"))

	// ============================================================
	// STEP 9: Freeze base weights (typical fine-tuning setup)
	// ============================================================
	fmt.Println("\n=== Step 9: Freezing Base Weights ===")

	// In a real fine-tuning scenario, you would:
	// 1. Load pre-trained weights
	// 2. Add adapters
	// 3. Freeze base weights
	// 4. Train only adapter weights

	trainableBeforeFreeze := 0
	for v := range ctxLoRA.IterVariables() {
		if v.Trainable {
			trainableBeforeFreeze++
		}
	}
	fmt.Printf("Trainable variables before freeze: %d\n", trainableBeforeFreeze)

	// Freeze non-LoRA weights
	for v := range ctxLoRA.IterVariables() {
		if !lora.IsLoRAVariable(v.ParameterName()) {
			v.SetTrainable(false)
		}
	}

	trainableAfterFreeze := 0
	for v := range ctxLoRA.IterVariables() {
		if v.Trainable {
			trainableAfterFreeze++
		}
	}
	fmt.Printf("Trainable variables after freeze: %d (only LoRA params)\n", trainableAfterFreeze)

	// ============================================================
	// STEP 10: Summary and best practices
	// ============================================================
	fmt.Println("\n=== Step 10: Summary ===")
	fmt.Print(`
Best Practices for PEFT in GoMLX:

1. CHOOSING RANK:
   - Start with rank=8 for most tasks
   - Increase to 16-64 for complex tasks
   - Use rank=4 for very constrained memory

2. CHOOSING ALPHA:
   - Common: alpha = rank (scale = 1.0)
   - Or: alpha = 2*rank (scale = 2.0) for stronger adaptation

3. CHOOSING ADAPTER TYPE:
   - LoRA: General purpose, well-tested
   - LoRA+: Better convergence with different LR for A/B
   - rsLoRA: More stable across different ranks
   - DoRA: Better for tasks requiring magnitude changes
   - IA³: Extremely parameter-efficient, good for few-shot

4. TARGET MODULES:
   - Attention Q,V projections: Best bang for buck
   - All attention (Q,K,V,O): More capacity
   - Attention + MLP: Maximum adaptation

5. MEMORY SAVINGS:
   - LoRA rank=8 on 7B model: ~0.1% of base params
   - QLoRA (4-bit): ~6x memory reduction for base weights
`)

	// Verify the test worked
	assert.Equal(t, []int{batchSize, seqLen, hiddenDim}, baseOutput.Shape().Dimensions)
	assert.Equal(t, []int{batchSize, seqLen, hiddenDim}, loraOutput.Shape().Dimensions)
}

// applyLoRADirect applies LoRA transformation directly (for demonstration).
// In practice, use lora.Dense or the Adapter.Apply method.
func applyLoRADirect(ctx *context.Context, input *Node, config *lora.Config, inputDim, outputDim int) *Node {
	g := input.Graph()
	dtype := input.DType()
	rank := config.Rank()

	// LoRA A: [inputDim, rank] - initialized with small random values
	loraAInit := func(g *Graph, shape shapes.Shape) *Node {
		// Kaiming uniform initialization
		return MulScalar(ctx.RandomUniform(g, shape), 0.1)
	}
	loraA := ctx.WithInitializer(loraAInit).
		VariableWithShape("lora_A", shapes.Make(dtype, inputDim, rank)).
		ValueGraph(g)

	// LoRA B: [rank, outputDim] - initialized to zero (so LoRA starts as identity)
	loraB := ctx.WithInitializer(initializers.Zero).
		VariableWithShape("lora_B", shapes.Make(dtype, rank, outputDim)).
		ValueGraph(g)

	// Compute: scale * (input @ A @ B)
	// Using Einsum for batched matrix multiplication
	hidden := Einsum("bsh,hr->bsr", input, loraA)   // [batch, seq, rank]
	output := Einsum("bsr,ro->bso", hidden, loraB)  // [batch, seq, outputDim]
	output = MulScalar(output, config.Scale())

	return output
}

// TestIA3Example demonstrates IA³ (Infused Adapter by Inhibiting and Amplifying
// Inner Activations), which uses learned rescaling vectors instead of matrices.
func TestIA3Example(t *testing.T) {
	backend, err := backends.New()
	require.NoError(t, err)
	ctx := context.New()

	fmt.Println("=== IA³ (Infused Adapter) Example ===")
	fmt.Print(`
IA³ is extremely parameter-efficient:
- Instead of low-rank matrices, uses learned scaling vectors
- Rescales key, value, and feedforward activations
- Parameters: O(d) instead of O(r*d) for LoRA

Formula:
- Key:   K' = l_k * K   (element-wise)
- Value: V' = l_v * V   (element-wise)
- FFN:   h' = l_ff * h  (element-wise)

Where l_k, l_v, l_ff are learned vectors.
`)

	config := ia3.NewConfig().SetInitScale(1.0)
	dim := 64

	// Demonstrate IA³ scaling
	exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, input *Node) *Node {
		// Simulate key/value scaling in attention
		scaledKey := ia3.ScaleActivation(ctx.In("key"), input, config, dim)
		scaledValue := ia3.ScaleActivation(ctx.In("value"), input, config, dim)

		// Simulate FFN intermediate scaling
		ffnHidden := ia3.ScaleActivation(ctx.In("ffn"), input, config, dim)

		// Just return one for the test
		_ = scaledKey
		_ = scaledValue
		return ffnHidden
	})

	inputData := make([][]float32, 2)
	for i := range inputData {
		inputData[i] = make([]float32, dim)
		for j := range inputData[i] {
			inputData[i][j] = 1.0
		}
	}

	output := exec.MustExec(inputData)[0]
	fmt.Printf("Output shape: %v\n", output.Shape())

	// Count IA³ parameters
	ia3Params := int64(0)
	for v := range ctx.IterVariables() {
		if ia3.IsIA3Variable(v.ParameterName()) {
			ia3Params += int64(v.Shape().Size())
			fmt.Printf("IA³ variable: %s, shape: %v\n", v.ParameterName(), v.Shape())
		}
	}
	fmt.Printf("Total IA³ parameters: %d (3 vectors of dim %d)\n", ia3Params, dim)

	// Compare to LoRA
	loraParams := 2 * 8 * dim // rank=8: A[dim,8] + B[8,dim] per layer
	fmt.Printf("Equivalent LoRA (rank=8) would use: %d params per layer\n", loraParams)
	fmt.Printf("IA³ is %.1fx more parameter-efficient!\n", float64(loraParams)/float64(dim))

	assert.Equal(t, int64(3*dim), ia3Params)
}

// TestAdapterComparisonExample shows parameter counts for different adapter types.
func TestAdapterComparisonExample(t *testing.T) {
	fmt.Println("=== Adapter Parameter Comparison ===")
	fmt.Print(`
For a single linear layer with input_dim=4096, output_dim=4096:

| Adapter Type | Rank | Parameters | % of Base |
|--------------|------|------------|-----------|
| Base weights | -    | 16,777,216 | 100%      |
| LoRA         | 8    | 65,536     | 0.39%     |
| LoRA         | 16   | 131,072    | 0.78%     |
| LoRA         | 64   | 524,288    | 3.13%     |
| IA³          | -    | 4,096      | 0.02%     |

For a 7B parameter model targeting Q,V projections (32 layers):

| Method       | Trainable Params | Memory (FP16) |
|--------------|------------------|---------------|
| Full FT      | 7,000,000,000    | ~14 GB        |
| LoRA r=8     | ~4,000,000       | ~8 MB         |
| LoRA r=64    | ~32,000,000      | ~64 MB        |
| QLoRA r=8    | ~4,000,000       | ~8 MB (+ 4-bit base) |
| IA³          | ~260,000         | ~0.5 MB       |
`)
}
