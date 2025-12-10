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
	"github.com/gomlx/gopjrt/dtypes"
)

// EncoderGraphFn defines the signature for encoder graph-building functions.
type EncoderGraphFn func(ctx *context.Context, inputIDs, attentionMask *graph.Node) []*graph.Node

// DecoderGraphFn defines the signature for decoder graph-building functions.
// It takes encoder hidden states, encoder attention mask, decoder input IDs,
// and optionally past KV cache, and returns logits and new KV cache.
type DecoderGraphFn func(ctx *context.Context, encoderHiddenStates, encoderAttentionMask, decoderInputIDs *graph.Node, pastKVCache []*graph.Node) []*graph.Node

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
// Returns updated hidden states and optionally new KV cache tensors.
func TransformerDecoderLayer(
	ctx *context.Context,
	hiddenStates, encoderHiddenStates, encoderAttentionMask *graph.Node,
	pastSelfK, pastSelfV, pastCrossK, pastCrossV *graph.Node,
	config *ModelConfig,
) (*graph.Node, *graph.Node, *graph.Node, *graph.Node, *graph.Node) {
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

	// For now, return nil for KV cache (to be implemented with actual caching)
	return hiddenStates, nil, nil, nil, nil
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
func CreateEmbedding(ctx *context.Context, inputIDs *graph.Node, vocabSize, embeddingDim int) *graph.Node {
	// Get or create embedding weights
	g := inputIDs.Graph()
	embeddingVar := ctx.WithInitializer(nil).VariableWithShape("embeddings",
		shapes.Make(dtypes.Float32, vocabSize, embeddingDim))
	embeddings := embeddingVar.ValueGraph(g)

	// Gather embeddings for input IDs
	return graph.Gather(embeddings, inputIDs)
}

// CreatePositionalEncoding creates positional encodings.
// Uses sinusoidal positional encoding by default.
// Takes a reference node to get the graph.
func CreatePositionalEncoding(g *graph.Graph, seqLen, embeddingDim int, dtype dtypes.DType) *graph.Node {
	// Create position indices [0, 1, ..., seqLen-1]
	positions := graph.IotaFull(g, shapes.Make(dtypes.Int32, seqLen))
	positions = graph.ConvertDType(positions, dtypes.Float32)

	// Create dimension indices
	dimIndices := graph.IotaFull(g, shapes.Make(dtypes.Int32, embeddingDim))
	dimIndices = graph.ConvertDType(dimIndices, dtypes.Float32)

	// Compute frequencies: 1 / (10000 ^ (2i / d))
	dimScale := graph.DivScalar(dimIndices, float64(embeddingDim))
	dimScale = graph.MulScalar(dimScale, 2.0)
	frequencies := graph.Pow(graph.ConstAs(dimScale, 10000.0), dimScale)
	frequencies = graph.Inverse(frequencies)

	// Reshape for broadcasting: positions [seqLen, 1], frequencies [1, embeddingDim]
	positions = graph.Reshape(positions, seqLen, 1)
	frequencies = graph.Reshape(frequencies, 1, embeddingDim)

	// Compute angles
	angles := graph.Mul(positions, frequencies)

	// Apply sin to even indices and cos to odd indices
	// For simplicity, just use sin (can be extended)
	encodings := graph.Sin(angles)

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
func BuildDecoderGraph(
	ctx *context.Context,
	encoderHiddenStates, encoderAttentionMask, decoderInputIDs *graph.Node,
	pastKVCache []*graph.Node,
	config *ModelConfig,
) []*graph.Node {
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
	newKVCache := make([]*graph.Node, 0, config.NumLayers*4)
	for i := 0; i < config.NumLayers; i++ {
		layerCtx := ctx.Inf("layer_%d", i)

		// Get past KV cache for this layer if available
		var pastSelfK, pastSelfV, pastCrossK, pastCrossV *graph.Node
		cacheOffset := i * 4
		if len(pastKVCache) > cacheOffset+3 {
			pastSelfK = pastKVCache[cacheOffset]
			pastSelfV = pastKVCache[cacheOffset+1]
			pastCrossK = pastKVCache[cacheOffset+2]
			pastCrossV = pastKVCache[cacheOffset+3]
		}

		var newSelfK, newSelfV, newCrossK, newCrossV *graph.Node
		hiddenStates, newSelfK, newSelfV, newCrossK, newCrossV = TransformerDecoderLayer(
			layerCtx, hiddenStates, encoderHiddenStates, encoderAttentionMask,
			pastSelfK, pastSelfV, pastCrossK, pastCrossV, config,
		)

		// Collect new KV cache
		newKVCache = append(newKVCache, newSelfK, newSelfV, newCrossK, newCrossV)
	}

	// Final layer norm
	hiddenStates = layers.LayerNormalization(ctx.In("final_norm"), hiddenStates, -1).Done()

	// LM head: project to vocabulary
	logits := layers.Dense(ctx.In("lm_head"), hiddenStates, false, config.VocabSize)

	// Return logits followed by KV cache
	outputs := make([]*graph.Node, 0, 1+len(newKVCache))
	outputs = append(outputs, logits)
	for _, kv := range newKVCache {
		if kv != nil {
			outputs = append(outputs, kv)
		}
	}

	return outputs
}
