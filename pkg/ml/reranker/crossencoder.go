// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package reranker provides components for building and training re-ranking models.
//
// Re-rankers (also called cross-encoders) are models that score the relevance of a document
// to a query by processing them together, allowing rich interaction between query and document tokens.
//
// This package provides:
//   - CrossEncoder: A transformer-based cross-encoder architecture
//   - Dataset interfaces for ranking (pointwise, pairwise, listwise)
//   - Hard negative mining utilities
package reranker

import (
	"fmt"

	"github.com/gomlx/gomlx/internal/exceptions"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
)

// Context parameter names for cross-encoder configuration.
const (
	// ParamNumLayers is the number of transformer encoder layers.
	ParamNumLayers = "crossencoder_num_layers"

	// ParamNumHeads is the number of attention heads.
	ParamNumHeads = "crossencoder_num_heads"

	// ParamHiddenDim is the hidden dimension (embedding dimension).
	ParamHiddenDim = "crossencoder_hidden_dim"

	// ParamIntermediateDim is the intermediate dimension in the FFN.
	// Defaults to 4 * HiddenDim.
	ParamIntermediateDim = "crossencoder_intermediate_dim"

	// ParamDropoutRate is the dropout rate for attention and FFN.
	ParamDropoutRate = "crossencoder_dropout_rate"

	// ParamAttentionDropoutRate is the dropout rate specifically for attention.
	// If not set, uses ParamDropoutRate.
	ParamAttentionDropoutRate = "crossencoder_attention_dropout_rate"

	// ParamActivation is the activation function for the FFN.
	// Default is "gelu".
	ParamActivation = "crossencoder_activation"

	// ParamVocabSize is the vocabulary size for embeddings.
	ParamVocabSize = "crossencoder_vocab_size"

	// ParamMaxSeqLen is the maximum sequence length for positional embeddings.
	ParamMaxSeqLen = "crossencoder_max_seq_len"

	// ParamNumTypeIDs is the number of token type IDs (for segment embeddings).
	// Default is 2 (query and document).
	ParamNumTypeIDs = "crossencoder_num_type_ids"

	// ParamPreNorm indicates whether to use pre-normalization (LayerNorm before attention/FFN).
	// Default is false (post-normalization, like original BERT).
	ParamPreNorm = "crossencoder_pre_norm"
)

// PoolingType defines how to pool the sequence output to a single vector.
type PoolingType string

const (
	// PoolingCLS uses the first token ([CLS]) representation.
	PoolingCLS PoolingType = "cls"

	// PoolingMean computes the mean of all tokens (excluding padding).
	PoolingMean PoolingType = "mean"

	// PoolingMax computes the max of all tokens (excluding padding).
	PoolingMax PoolingType = "max"
)

// CrossEncoderConfig holds the configuration for a cross-encoder model.
type CrossEncoderConfig struct {
	// NumLayers is the number of transformer encoder layers.
	NumLayers int

	// NumHeads is the number of attention heads.
	NumHeads int

	// HiddenDim is the model hidden dimension (embedding dimension).
	HiddenDim int

	// IntermediateDim is the intermediate dimension in the FFN.
	// If 0, defaults to 4 * HiddenDim.
	IntermediateDim int

	// DropoutRate is the dropout rate for attention and FFN.
	DropoutRate float64

	// AttentionDropoutRate is the dropout rate for attention.
	// If 0, uses DropoutRate.
	AttentionDropoutRate float64

	// Activation is the activation function for the FFN.
	// Default is "gelu".
	Activation activations.Type

	// VocabSize is the vocabulary size for embeddings.
	VocabSize int

	// MaxSeqLen is the maximum sequence length for positional embeddings.
	MaxSeqLen int

	// NumTypeIDs is the number of token type IDs (0 to disable).
	// Default is 2 for query/document segments.
	NumTypeIDs int

	// PreNorm indicates whether to use pre-normalization.
	PreNorm bool

	// Pooling defines how to pool the sequence output.
	Pooling PoolingType
}

// CrossEncoderBuilder builds a cross-encoder model using the builder pattern.
type CrossEncoderBuilder struct {
	ctx    *context.Context
	config CrossEncoderConfig
}

// CrossEncoder creates a new cross-encoder builder.
//
// A cross-encoder (also called re-ranker) processes query-document pairs together,
// allowing rich interaction between them through self-attention.
//
// Example usage:
//
//	scores := reranker.CrossEncoder(ctx).
//	    NumLayers(6).
//	    NumHeads(8).
//	    HiddenDim(768).
//	    VocabSize(30522).
//	    MaxSeqLen(512).
//	    Done(tokenIDs, attentionMask, tokenTypeIDs)
func CrossEncoder(ctx *context.Context) *CrossEncoderBuilder {
	return &CrossEncoderBuilder{
		ctx: ctx.In("cross_encoder"),
		config: CrossEncoderConfig{
			NumLayers:   context.GetParamOr(ctx, ParamNumLayers, 6),
			NumHeads:    context.GetParamOr(ctx, ParamNumHeads, 8),
			HiddenDim:   context.GetParamOr(ctx, ParamHiddenDim, 768),
			DropoutRate: context.GetParamOr(ctx, ParamDropoutRate, 0.1),
			Activation:  activations.TypeGelu,
			VocabSize:   context.GetParamOr(ctx, ParamVocabSize, 30522),
			MaxSeqLen:   context.GetParamOr(ctx, ParamMaxSeqLen, 512),
			NumTypeIDs:  context.GetParamOr(ctx, ParamNumTypeIDs, 2),
			PreNorm:     context.GetParamOr(ctx, ParamPreNorm, false),
			Pooling:     PoolingCLS,
		},
	}
}

// FromContext reads all configuration from context parameters.
func (b *CrossEncoderBuilder) FromContext() *CrossEncoderBuilder {
	b.config.NumLayers = context.GetParamOr(b.ctx, ParamNumLayers, b.config.NumLayers)
	b.config.NumHeads = context.GetParamOr(b.ctx, ParamNumHeads, b.config.NumHeads)
	b.config.HiddenDim = context.GetParamOr(b.ctx, ParamHiddenDim, b.config.HiddenDim)
	b.config.IntermediateDim = context.GetParamOr(b.ctx, ParamIntermediateDim, b.config.IntermediateDim)
	b.config.DropoutRate = context.GetParamOr(b.ctx, ParamDropoutRate, b.config.DropoutRate)
	b.config.AttentionDropoutRate = context.GetParamOr(b.ctx, ParamAttentionDropoutRate, b.config.AttentionDropoutRate)
	b.config.VocabSize = context.GetParamOr(b.ctx, ParamVocabSize, b.config.VocabSize)
	b.config.MaxSeqLen = context.GetParamOr(b.ctx, ParamMaxSeqLen, b.config.MaxSeqLen)
	b.config.NumTypeIDs = context.GetParamOr(b.ctx, ParamNumTypeIDs, b.config.NumTypeIDs)
	b.config.PreNorm = context.GetParamOr(b.ctx, ParamPreNorm, b.config.PreNorm)

	if actStr := context.GetParamOr(b.ctx, ParamActivation, ""); actStr != "" {
		act, err := activations.TypeString(actStr)
		if err == nil {
			b.config.Activation = act
		}
	}
	return b
}

// NumLayers sets the number of transformer encoder layers.
func (b *CrossEncoderBuilder) NumLayers(n int) *CrossEncoderBuilder {
	if n <= 0 {
		exceptions.Panicf("NumLayers must be > 0, got %d", n)
	}
	b.config.NumLayers = n
	return b
}

// NumHeads sets the number of attention heads.
func (b *CrossEncoderBuilder) NumHeads(n int) *CrossEncoderBuilder {
	if n <= 0 {
		exceptions.Panicf("NumHeads must be > 0, got %d", n)
	}
	b.config.NumHeads = n
	return b
}

// HiddenDim sets the hidden dimension (embedding dimension).
func (b *CrossEncoderBuilder) HiddenDim(dim int) *CrossEncoderBuilder {
	if dim <= 0 {
		exceptions.Panicf("HiddenDim must be > 0, got %d", dim)
	}
	b.config.HiddenDim = dim
	return b
}

// IntermediateDim sets the FFN intermediate dimension.
func (b *CrossEncoderBuilder) IntermediateDim(dim int) *CrossEncoderBuilder {
	b.config.IntermediateDim = dim
	return b
}

// DropoutRate sets the dropout rate.
func (b *CrossEncoderBuilder) DropoutRate(rate float64) *CrossEncoderBuilder {
	b.config.DropoutRate = rate
	return b
}

// AttentionDropoutRate sets the attention-specific dropout rate.
func (b *CrossEncoderBuilder) AttentionDropoutRate(rate float64) *CrossEncoderBuilder {
	b.config.AttentionDropoutRate = rate
	return b
}

// Activation sets the activation function for FFN.
func (b *CrossEncoderBuilder) Activation(act activations.Type) *CrossEncoderBuilder {
	b.config.Activation = act
	return b
}

// VocabSize sets the vocabulary size.
func (b *CrossEncoderBuilder) VocabSize(size int) *CrossEncoderBuilder {
	if size <= 0 {
		exceptions.Panicf("VocabSize must be > 0, got %d", size)
	}
	b.config.VocabSize = size
	return b
}

// MaxSeqLen sets the maximum sequence length.
func (b *CrossEncoderBuilder) MaxSeqLen(len int) *CrossEncoderBuilder {
	if len <= 0 {
		exceptions.Panicf("MaxSeqLen must be > 0, got %d", len)
	}
	b.config.MaxSeqLen = len
	return b
}

// NumTypeIDs sets the number of token type IDs (0 to disable).
func (b *CrossEncoderBuilder) NumTypeIDs(n int) *CrossEncoderBuilder {
	b.config.NumTypeIDs = n
	return b
}

// PreNorm sets whether to use pre-normalization.
func (b *CrossEncoderBuilder) PreNorm(preNorm bool) *CrossEncoderBuilder {
	b.config.PreNorm = preNorm
	return b
}

// Pooling sets the pooling strategy.
func (b *CrossEncoderBuilder) Pooling(pooling PoolingType) *CrossEncoderBuilder {
	b.config.Pooling = pooling
	return b
}

// Done builds the cross-encoder graph and returns the relevance scores.
//
// Parameters:
//   - tokenIDs: Token IDs of shape [batch_size, seq_len]
//   - attentionMask: Attention mask of shape [batch_size, seq_len] (1 for real tokens, 0 for padding)
//   - tokenTypeIDs: Optional token type IDs of shape [batch_size, seq_len] (nil to skip)
//
// Returns:
//   - scores: Relevance scores of shape [batch_size, 1]
func (b *CrossEncoderBuilder) Done(tokenIDs, attentionMask, tokenTypeIDs *Node) *Node {
	hidden := b.Encode(tokenIDs, attentionMask, tokenTypeIDs)
	pooled := b.Pool(hidden, attentionMask)
	return b.Head(pooled)
}

// Encode applies embeddings and transformer layers, returning the full sequence output.
//
// Parameters:
//   - tokenIDs: Token IDs of shape [batch_size, seq_len]
//   - attentionMask: Attention mask of shape [batch_size, seq_len]
//   - tokenTypeIDs: Optional token type IDs (nil to skip)
//
// Returns:
//   - hidden: Hidden states of shape [batch_size, seq_len, hidden_dim]
func (b *CrossEncoderBuilder) Encode(tokenIDs, attentionMask, tokenTypeIDs *Node) *Node {
	ctx := b.ctx.In("encoder")
	g := tokenIDs.Graph()

	// Embeddings
	hidden := b.embeddings(ctx.In("embeddings"), tokenIDs, tokenTypeIDs)

	// Apply dropout to embeddings
	if b.config.DropoutRate > 0 {
		hidden = layers.DropoutStatic(ctx, hidden, b.config.DropoutRate)
	}

	// Convert attention mask to boolean for use in attention
	attentionMaskBool := NotEqual(attentionMask, ScalarZero(g, attentionMask.DType()))

	// Transformer layers
	headDim := b.config.HiddenDim / b.config.NumHeads
	attDropout := b.config.AttentionDropoutRate
	if attDropout == 0 {
		attDropout = b.config.DropoutRate
	}

	for i := 0; i < b.config.NumLayers; i++ {
		layerCtx := ctx.In(fmt.Sprintf("layer_%d", i))
		hidden = b.encoderLayer(layerCtx, hidden, attentionMaskBool, headDim, attDropout)
	}

	return hidden
}

// embeddings applies token, position, and optionally type embeddings.
func (b *CrossEncoderBuilder) embeddings(ctx *context.Context, tokenIDs, tokenTypeIDs *Node) *Node {
	g := tokenIDs.Graph()
	dtype := dtypes.Float32 // Standard dtype for embeddings
	seqLen := tokenIDs.Shape().Dimensions[1]

	// Token embeddings
	hidden := layers.Embedding(ctx.In("token"), tokenIDs, dtype, b.config.VocabSize, b.config.HiddenDim)
	// Shape: [batch, seq_len, hidden_dim]

	// Position embeddings
	positions := Iota(g, shapes.Make(tokenIDs.DType(), seqLen), 0)
	positions = ExpandDims(positions, 0) // [1, seq_len]
	posEmbed := layers.Embedding(ctx.In("position"), positions, dtype, b.config.MaxSeqLen, b.config.HiddenDim)
	hidden = Add(hidden, posEmbed)

	// Token type embeddings (optional)
	if tokenTypeIDs != nil && b.config.NumTypeIDs > 0 {
		typeEmbed := layers.Embedding(ctx.In("token_type"), tokenTypeIDs, dtype, b.config.NumTypeIDs, b.config.HiddenDim)
		hidden = Add(hidden, typeEmbed)
	}

	// Layer normalization
	hidden = layers.LayerNormalization(ctx.In("norm"), hidden, -1).Done()

	return hidden
}

// encoderLayer applies one transformer encoder layer.
func (b *CrossEncoderBuilder) encoderLayer(ctx *context.Context, hidden, attentionMask *Node, headDim int, attDropout float64) *Node {
	if b.config.PreNorm {
		// Pre-normalization: norm -> attention -> residual -> norm -> ffn -> residual
		normed := layers.LayerNormalization(ctx.In("attention_norm"), hidden, -1).Done()
		attnOutput := layers.MultiHeadAttention(ctx.In("attention"), normed, normed, normed, b.config.NumHeads, headDim).
			SetKeyMask(attentionMask).
			Dropout(attDropout).
			Done()
		if b.config.DropoutRate > 0 {
			attnOutput = layers.DropoutStatic(ctx, attnOutput, b.config.DropoutRate)
		}
		hidden = Add(hidden, attnOutput)

		normed = layers.LayerNormalization(ctx.In("ffn_norm"), hidden, -1).Done()
		ffnOutput := b.ffn(ctx.In("ffn"), normed)
		if b.config.DropoutRate > 0 {
			ffnOutput = layers.DropoutStatic(ctx, ffnOutput, b.config.DropoutRate)
		}
		hidden = Add(hidden, ffnOutput)
	} else {
		// Post-normalization (original BERT style): attention -> residual -> norm -> ffn -> residual -> norm
		attnOutput := layers.MultiHeadAttention(ctx.In("attention"), hidden, hidden, hidden, b.config.NumHeads, headDim).
			SetKeyMask(attentionMask).
			Dropout(attDropout).
			Done()
		if b.config.DropoutRate > 0 {
			attnOutput = layers.DropoutStatic(ctx, attnOutput, b.config.DropoutRate)
		}
		hidden = layers.LayerNormalization(ctx.In("attention_norm"), Add(hidden, attnOutput), -1).Done()

		ffnOutput := b.ffn(ctx.In("ffn"), hidden)
		if b.config.DropoutRate > 0 {
			ffnOutput = layers.DropoutStatic(ctx, ffnOutput, b.config.DropoutRate)
		}
		hidden = layers.LayerNormalization(ctx.In("ffn_norm"), Add(hidden, ffnOutput), -1).Done()
	}

	return hidden
}

// ffn applies the feed-forward network.
func (b *CrossEncoderBuilder) ffn(ctx *context.Context, x *Node) *Node {
	intermediateDim := b.config.IntermediateDim
	if intermediateDim == 0 {
		intermediateDim = 4 * b.config.HiddenDim
	}

	// Two-layer FFN: hidden_dim -> intermediate_dim -> hidden_dim
	x = layers.DenseWithBias(ctx.In("intermediate"), x, intermediateDim)
	x = activations.Apply(b.config.Activation, x)
	x = layers.DenseWithBias(ctx.In("output"), x, b.config.HiddenDim)
	return x
}

// Pool applies pooling to get a single vector representation.
func (b *CrossEncoderBuilder) Pool(hidden, attentionMask *Node) *Node {
	switch b.config.Pooling {
	case PoolingCLS:
		// Use the first token ([CLS]) representation
		return Slice(hidden, AxisRange(0), AxisElem(0), AxisRange())
		// Shape: [batch, hidden_dim]

	case PoolingMean:
		// Mean pooling over non-padding tokens
		maskFloat := ConvertDType(attentionMask, hidden.DType())
		maskExpanded := ExpandDims(maskFloat, -1) // [batch, seq_len, 1]
		masked := Mul(hidden, maskExpanded)
		summed := ReduceSum(masked, 1)                        // [batch, hidden_dim]
		counts := ReduceSum(maskFloat, -1)                    // [batch]
		counts = AddScalar(counts, 1e-9)                      // Avoid division by zero
		return Div(summed, ExpandDims(counts, -1))

	case PoolingMax:
		// Max pooling over non-padding tokens
		// Set padding positions to large negative value
		largeNeg := ConstAs(hidden, -1e9)
		maskBool := NotEqual(attentionMask, ScalarZero(attentionMask.Graph(), attentionMask.DType()))
		masked := Where(
			ExpandDims(maskBool, -1),
			hidden,
			BroadcastToDims(largeNeg, hidden.Shape().Dimensions...),
		)
		return ReduceMax(masked, 1) // [batch, hidden_dim]

	default:
		exceptions.Panicf("Unknown pooling type: %s", b.config.Pooling)
		return nil
	}
}

// Head applies the classification head to produce relevance scores.
func (b *CrossEncoderBuilder) Head(pooled *Node) *Node {
	ctx := b.ctx.In("head")

	// Optional: Apply a dense layer before the output
	hidden := layers.DenseWithBias(ctx.In("dense"), pooled, b.config.HiddenDim)
	hidden = activations.Apply(activations.TypeTanh, hidden)

	// Output projection to single score
	scores := layers.DenseWithBias(ctx.In("output"), hidden, 1)
	return scores
}
