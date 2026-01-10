/*
 *	Copyright 2024 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

package seq2seq

import (
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
)

// EncoderGraphFn defines the signature for encoder graph-building functions.
type EncoderGraphFn func(ctx *context.Context, inputIDs, attentionMask *graph.Node) []*graph.Node

// DecoderGraphFn defines the signature for decoder graph-building functions.
// Takes encoder hidden states, encoder attention mask, and decoder input IDs.
// Returns logits tensor.
//
// Note: KV cache parameters will be added when PR #294 lands.
type DecoderGraphFn func(ctx *context.Context, encoderHiddenStates, encoderAttentionMask, decoderInputIDs *graph.Node) *graph.Node

// TransformerEncoderLayer creates a single transformer encoder layer.
// Returns updated hidden states.
func TransformerEncoderLayer(ctx *context.Context, hiddenStates, attentionMask *graph.Node, config *ModelConfig) *graph.Node {
	ctx = ctx.In("encoder_layer")

	// Self-attention
	selfAttnOutput := layers.MultiHeadAttention(
		ctx.In("self_attention"),
		hiddenStates, hiddenStates, hiddenStates,
		config.NumHeads, config.HeadDim,
	)
	if attentionMask != nil {
		selfAttnOutput = selfAttnOutput.SetKeyMask(attentionMask)
	}
	attnOutput := selfAttnOutput.Done()

	// Add & Norm
	hiddenStates = graph.Add(hiddenStates, attnOutput)
	hiddenStates = layers.LayerNormalization(ctx.In("attention_norm"), hiddenStates, -1).Done()

	// Feed-forward
	ffOutput := feedForward(ctx.In("feed_forward"), hiddenStates, config)

	// Add & Norm
	hiddenStates = graph.Add(hiddenStates, ffOutput)
	hiddenStates = layers.LayerNormalization(ctx.In("output_norm"), hiddenStates, -1).Done()

	return hiddenStates
}

// TransformerDecoderLayer creates a single transformer decoder layer.
// Returns updated hidden states.
//
// Note: KV caching is not yet implemented. When PR #294 (layers/attention/kvcache.go)
// lands, this function should be updated to use the unified KV cache abstraction.
// Until then, autoregressive generation recomputes all past states.
func TransformerDecoderLayer(
	ctx *context.Context,
	hiddenStates, encoderHiddenStates, encoderAttentionMask *graph.Node,
	config *ModelConfig,
) *graph.Node {
	ctx = ctx.In("decoder_layer")

	// Self-attention with causal mask
	selfAttnBuilder := layers.MultiHeadAttention(
		ctx.In("self_attention"),
		hiddenStates, hiddenStates, hiddenStates,
		config.NumHeads, config.HeadDim,
	).UseCausalMask()

	selfAttnOutput := selfAttnBuilder.Done()

	// Add & Norm
	hiddenStates = graph.Add(hiddenStates, selfAttnOutput)
	hiddenStates = layers.LayerNormalization(ctx.In("self_attention_norm"), hiddenStates, -1).Done()

	// Cross-attention to encoder
	crossAttnBuilder := layers.MultiHeadAttention(
		ctx.In("cross_attention"),
		hiddenStates, encoderHiddenStates, encoderHiddenStates,
		config.NumHeads, config.HeadDim,
	)
	if encoderAttentionMask != nil {
		crossAttnBuilder = crossAttnBuilder.SetKeyMask(encoderAttentionMask)
	}
	crossAttnOutput := crossAttnBuilder.Done()

	// Add & Norm
	hiddenStates = graph.Add(hiddenStates, crossAttnOutput)
	hiddenStates = layers.LayerNormalization(ctx.In("cross_attention_norm"), hiddenStates, -1).Done()

	// Feed-forward
	ffOutput := feedForward(ctx.In("feed_forward"), hiddenStates, config)

	// Add & Norm
	hiddenStates = graph.Add(hiddenStates, ffOutput)
	hiddenStates = layers.LayerNormalization(ctx.In("output_norm"), hiddenStates, -1).Done()

	return hiddenStates
}

// feedForward implements the feed-forward network in a transformer layer.
func feedForward(ctx *context.Context, x *graph.Node, config *ModelConfig) *graph.Node {
	// Typically: Linear -> Activation -> Linear
	// With hidden size expansion (usually 4x)
	ffDim := config.HiddenSize * 4

	// First linear
	x = layers.Dense(ctx.In("linear1"), x, true, ffDim)

	// Activation (GELU is common in modern transformers)
	x = activations.Gelu(x)

	// Second linear
	x = layers.Dense(ctx.In("linear2"), x, true, config.HiddenSize)

	return x
}

// CreateEmbedding creates token embeddings from input IDs.
// Uses the context's default initializer for the embedding weights.
// Input should be integer tensor of any shape. Output will have shape
// [...inputShape, embeddingDim].
func CreateEmbedding(ctx *context.Context, inputIDs *graph.Node, vocabSize, embeddingDim int) *graph.Node {
	inputShape := inputIDs.Shape()
	if !inputShape.DType.IsInt() {
		panic("can only use CreateEmbedding on integer inputs")
	}

	// Add a last dimension of size 1 if needed, since Gather needs an index dimension.
	input := inputIDs
	if inputShape.IsScalar() || inputShape.Dimensions[inputShape.Rank()-1] != 1 {
		input = graph.InsertAxes(input, -1)
	}

	// Get or create embedding weights using context's initializer.
	g := inputIDs.Graph()
	embeddingVar := ctx.VariableWithShape("embeddings",
		shapes.Make(dtypes.Float32, vocabSize, embeddingDim))
	embeddings := embeddingVar.ValueGraph(g)

	// Gather embeddings for input IDs.
	return graph.Gather(embeddings, input)
}

// CreatePositionalEncoding creates sinusoidal positional encodings.
// Uses the standard formulation: PE(pos, 2i) = sin(pos / 10000^(2i/d)),
// PE(pos, 2i+1) = cos(pos / 10000^(2i/d)).
func CreatePositionalEncoding(g *graph.Graph, seqLen, embeddingDim int, dtype dtypes.DType) *graph.Node {
	// Create position indices [0, 1, ..., seqLen-1]
	positions := graph.IotaFull(g, shapes.Make(dtypes.Int32, seqLen))
	positions = graph.ConvertDType(positions, dtypes.Float32)

	// Create dimension indices for the frequency calculation.
	// We compute frequencies for pairs of dimensions (2i), so we need embeddingDim/2 frequencies.
	halfDim := embeddingDim / 2
	dimIndices := graph.IotaFull(g, shapes.Make(dtypes.Int32, halfDim))
	dimIndices = graph.ConvertDType(dimIndices, dtypes.Float32)

	// Compute frequencies: 1 / (10000 ^ (2i / d))
	dimScale := graph.DivScalar(dimIndices, float64(embeddingDim))
	dimScale = graph.MulScalar(dimScale, 2.0)
	frequencies := graph.Pow(graph.ConstAs(dimScale, 10000.0), dimScale)
	frequencies = graph.Inverse(frequencies)

	// Reshape for broadcasting: positions [seqLen, 1], frequencies [1, halfDim]
	positions = graph.Reshape(positions, seqLen, 1)
	frequencies = graph.Reshape(frequencies, 1, halfDim)

	// Compute angles: [seqLen, halfDim]
	angles := graph.Mul(positions, frequencies)

	// Apply sin to even indices and cos to odd indices.
	sinEncodings := graph.Sin(angles) // [seqLen, halfDim]
	cosEncodings := graph.Cos(angles) // [seqLen, halfDim]

	// Interleave sin and cos: [sin_0, cos_0, sin_1, cos_1, ...]
	// Stack along a new axis, then reshape.
	// Shape after stack: [seqLen, halfDim, 2]
	stacked := graph.Stack([]*graph.Node{sinEncodings, cosEncodings}, -1)
	// Reshape to [seqLen, embeddingDim]
	encodings := graph.Reshape(stacked, seqLen, halfDim*2)

	// Handle odd embedding dimensions by slicing if needed.
	if embeddingDim%2 != 0 {
		encodings = graph.Slice(encodings, graph.AxisRange(), graph.AxisRange(0, embeddingDim))
	}

	if dtype != dtypes.Float32 {
		encodings = graph.ConvertDType(encodings, dtype)
	}

	return encodings
}

// BuildEncoderGraph builds a complete encoder graph.
func BuildEncoderGraph(
	ctx *context.Context,
	inputIDs, attentionMask *graph.Node,
	config *ModelConfig,
) *graph.Node {
	ctx = ctx.In("encoder")
	g := inputIDs.Graph()

	// Embedding layer
	hiddenStates := CreateEmbedding(ctx.In("embeddings"), inputIDs, config.VocabSize, config.HiddenSize)

	// Add positional encoding
	seqLen := inputIDs.Shape().Dimensions[1]
	posEncoding := CreatePositionalEncoding(g, seqLen, config.HiddenSize, config.DType)
	posEncoding = graph.BroadcastToDims(posEncoding, hiddenStates.Shape().Dimensions...)
	hiddenStates = graph.Add(hiddenStates, posEncoding)

	// Encoder layers
	for i := 0; i < config.NumLayers; i++ {
		layerCtx := ctx.Inf("layer_%d", i)
		hiddenStates = TransformerEncoderLayer(layerCtx, hiddenStates, attentionMask, config)
	}

	return hiddenStates
}

// BuildDecoderGraph builds a complete decoder graph (single step).
// Returns logits tensor of shape [batch_size, seq_len, vocab_size].
//
// Note: KV caching will be added when PR #294 lands. Currently recomputes all states.
func BuildDecoderGraph(
	ctx *context.Context,
	encoderHiddenStates, encoderAttentionMask, decoderInputIDs *graph.Node,
	config *ModelConfig,
) *graph.Node {
	ctx = ctx.In("decoder")
	g := decoderInputIDs.Graph()

	// Embedding layer
	hiddenStates := CreateEmbedding(ctx.In("embeddings"), decoderInputIDs, config.VocabSize, config.HiddenSize)

	// Add positional encoding
	seqLen := decoderInputIDs.Shape().Dimensions[1]
	posEncoding := CreatePositionalEncoding(g, seqLen, config.HiddenSize, config.DType)
	posEncoding = graph.BroadcastToDims(posEncoding, hiddenStates.Shape().Dimensions...)
	hiddenStates = graph.Add(hiddenStates, posEncoding)

	// Decoder layers
	for i := 0; i < config.NumLayers; i++ {
		layerCtx := ctx.Inf("layer_%d", i)
		hiddenStates = TransformerDecoderLayer(
			layerCtx, hiddenStates, encoderHiddenStates, encoderAttentionMask, config,
		)
	}

	// Final layer norm
	hiddenStates = layers.LayerNormalization(ctx.In("final_norm"), hiddenStates, -1).Done()

	// LM head: project to vocabulary
	logits := layers.Dense(ctx.In("lm_head"), hiddenStates, false, config.VocabSize)

	return logits
}
