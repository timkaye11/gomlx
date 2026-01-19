// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package reranker

import (
	"fmt"

	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/train"
)

// RankingDataset provides batches for training ranking models.
// This interface extends the basic train.Dataset with ranking-specific methods.
type RankingDataset interface {
	train.Dataset

	// DatasetType returns the type of ranking data this dataset provides.
	DatasetType() DatasetType
}

// DatasetType indicates the format of ranking training data.
type DatasetType int

const (
	// DatasetPointwise provides (query, document, relevance_score) tuples.
	// Used for binary classification (relevant/not relevant) or regression.
	DatasetPointwise DatasetType = iota

	// DatasetPairwise provides (query, positive_doc, negative_doc) tuples.
	// Used with margin-based or pairwise logistic losses.
	DatasetPairwise

	// DatasetListwise provides (query, [documents], [relevance_scores]) tuples.
	// Used with listwise losses like ListMLE, ListNet, or LambdaLoss.
	DatasetListwise
)

// String returns the string representation of the dataset type.
func (dt DatasetType) String() string {
	switch dt {
	case DatasetPointwise:
		return "pointwise"
	case DatasetPairwise:
		return "pairwise"
	case DatasetListwise:
		return "listwise"
	default:
		return "unknown"
	}
}

// PointwiseSample represents a single query-document pair with relevance label.
type PointwiseSample struct {
	// QueryTokens are the tokenized query.
	QueryTokens []int32

	// DocTokens are the tokenized document.
	DocTokens []int32

	// Label is the relevance score (e.g., 0 or 1 for binary, 0-4 for graded).
	Label float32
}

// PairwiseSample represents a query with positive and negative documents.
type PairwiseSample struct {
	// QueryTokens are the tokenized query.
	QueryTokens []int32

	// PositiveTokens are the tokenized positive (relevant) document.
	PositiveTokens []int32

	// NegativeTokens are the tokenized negative (non-relevant) document.
	NegativeTokens []int32
}

// ListwiseSample represents a query with multiple documents and relevance scores.
type ListwiseSample struct {
	// QueryTokens are the tokenized query.
	QueryTokens []int32

	// DocTokensList contains tokenized documents for the query.
	DocTokensList [][]int32

	// Labels are the relevance scores for each document.
	Labels []float32
}

// PointwiseDataset is a simple in-memory pointwise dataset for ranking.
type PointwiseDataset struct {
	samples     []PointwiseSample
	maxSeqLen   int
	sepTokenID  int32
	clsTokenID  int32
	padTokenID  int32
	batchSize   int
	currentIdx  int
	infinite    bool
	shuffleFunc func([]PointwiseSample)
}

// NewPointwiseDataset creates a new pointwise dataset.
//
// Parameters:
//   - samples: The query-document pairs with labels.
//   - maxSeqLen: Maximum sequence length (query + document + special tokens).
//   - clsTokenID: Token ID for [CLS] token.
//   - sepTokenID: Token ID for [SEP] token.
//   - padTokenID: Token ID for [PAD] token.
//   - batchSize: Number of samples per batch.
func NewPointwiseDataset(samples []PointwiseSample, maxSeqLen int, clsTokenID, sepTokenID, padTokenID int32, batchSize int) *PointwiseDataset {
	return &PointwiseDataset{
		samples:    samples,
		maxSeqLen:  maxSeqLen,
		clsTokenID: clsTokenID,
		sepTokenID: sepTokenID,
		padTokenID: padTokenID,
		batchSize:  batchSize,
		infinite:   true,
	}
}

// WithShuffle sets a shuffle function to randomize sample order between epochs.
func (d *PointwiseDataset) WithShuffle(shuffleFunc func([]PointwiseSample)) *PointwiseDataset {
	d.shuffleFunc = shuffleFunc
	return d
}

// SetInfinite sets whether the dataset should loop infinitely.
func (d *PointwiseDataset) SetInfinite(infinite bool) *PointwiseDataset {
	d.infinite = infinite
	return d
}

// Name returns the dataset name.
func (d *PointwiseDataset) Name() string {
	return "pointwise_ranking"
}

// Reset resets the dataset to the beginning.
func (d *PointwiseDataset) Reset() {
	d.currentIdx = 0
	if d.shuffleFunc != nil {
		d.shuffleFunc(d.samples)
	}
}

// Yield returns the next batch of data.
// Returns (inputs, labels, nil) where:
//   - inputs[0]: token_ids [batch, seq_len]
//   - inputs[1]: attention_mask [batch, seq_len]
//   - inputs[2]: token_type_ids [batch, seq_len]
//   - labels[0]: relevance labels [batch, 1]
func (d *PointwiseDataset) Yield() (inputs, labels []*tensors.Tensor, err error) {
	if d.currentIdx >= len(d.samples) {
		if d.infinite {
			d.Reset()
		} else {
			return nil, nil, nil // End of data
		}
	}

	// Get batch indices
	endIdx := d.currentIdx + d.batchSize
	if endIdx > len(d.samples) {
		endIdx = len(d.samples)
	}
	batchSamples := d.samples[d.currentIdx:endIdx]
	actualBatchSize := len(batchSamples)
	d.currentIdx = endIdx

	// Prepare batch tensors
	tokenIDs := make([][]int32, actualBatchSize)
	attentionMask := make([][]int32, actualBatchSize)
	tokenTypeIDs := make([][]int32, actualBatchSize)
	labelValues := make([]float32, actualBatchSize)

	for i, sample := range batchSamples {
		tokenIDs[i], attentionMask[i], tokenTypeIDs[i] = d.encodeQueryDoc(sample.QueryTokens, sample.DocTokens)
		labelValues[i] = sample.Label
	}

	// Convert to tensors
	inputs = []*tensors.Tensor{
		tensors.FromFlatDataAndDimensions(flatten2DInt32(tokenIDs), actualBatchSize, d.maxSeqLen),
		tensors.FromFlatDataAndDimensions(flatten2DInt32(attentionMask), actualBatchSize, d.maxSeqLen),
		tensors.FromFlatDataAndDimensions(flatten2DInt32(tokenTypeIDs), actualBatchSize, d.maxSeqLen),
	}
	labels = []*tensors.Tensor{
		tensors.FromFlatDataAndDimensions(labelValues, actualBatchSize, 1),
	}

	return inputs, labels, nil
}

// DatasetType returns DatasetPointwise.
func (d *PointwiseDataset) DatasetType() DatasetType {
	return DatasetPointwise
}

// encodeQueryDoc combines query and document tokens into BERT format.
// Format: [CLS] query [SEP] document [SEP] [PAD]...
func (d *PointwiseDataset) encodeQueryDoc(query, doc []int32) (tokenIDs, attentionMask, tokenTypeIDs []int32) {
	return encodeQueryDocPair(query, doc, d.maxSeqLen, d.clsTokenID, d.sepTokenID, d.padTokenID)
}

// PairwiseDataset is a simple in-memory pairwise dataset for ranking.
type PairwiseDataset struct {
	samples     []PairwiseSample
	maxSeqLen   int
	sepTokenID  int32
	clsTokenID  int32
	padTokenID  int32
	batchSize   int
	currentIdx  int
	infinite    bool
	shuffleFunc func([]PairwiseSample)
}

// NewPairwiseDataset creates a new pairwise dataset.
func NewPairwiseDataset(samples []PairwiseSample, maxSeqLen int, clsTokenID, sepTokenID, padTokenID int32, batchSize int) *PairwiseDataset {
	return &PairwiseDataset{
		samples:    samples,
		maxSeqLen:  maxSeqLen,
		clsTokenID: clsTokenID,
		sepTokenID: sepTokenID,
		padTokenID: padTokenID,
		batchSize:  batchSize,
		infinite:   true,
	}
}

// WithShuffle sets a shuffle function to randomize sample order between epochs.
func (d *PairwiseDataset) WithShuffle(shuffleFunc func([]PairwiseSample)) *PairwiseDataset {
	d.shuffleFunc = shuffleFunc
	return d
}

// SetInfinite sets whether the dataset should loop infinitely.
func (d *PairwiseDataset) SetInfinite(infinite bool) *PairwiseDataset {
	d.infinite = infinite
	return d
}

// Name returns the dataset name.
func (d *PairwiseDataset) Name() string {
	return "pairwise_ranking"
}

// Reset resets the dataset to the beginning.
func (d *PairwiseDataset) Reset() {
	d.currentIdx = 0
	if d.shuffleFunc != nil {
		d.shuffleFunc(d.samples)
	}
}

// Yield returns the next batch of data for pairwise training.
// Returns (inputs, nil, nil) where:
//   - inputs[0]: positive_token_ids [batch, seq_len]
//   - inputs[1]: positive_attention_mask [batch, seq_len]
//   - inputs[2]: positive_token_type_ids [batch, seq_len]
//   - inputs[3]: negative_token_ids [batch, seq_len]
//   - inputs[4]: negative_attention_mask [batch, seq_len]
//   - inputs[5]: negative_token_type_ids [batch, seq_len]
func (d *PairwiseDataset) Yield() (inputs, labels []*tensors.Tensor, err error) {
	if d.currentIdx >= len(d.samples) {
		if d.infinite {
			d.Reset()
		} else {
			return nil, nil, nil
		}
	}

	endIdx := d.currentIdx + d.batchSize
	if endIdx > len(d.samples) {
		endIdx = len(d.samples)
	}
	batchSamples := d.samples[d.currentIdx:endIdx]
	actualBatchSize := len(batchSamples)
	d.currentIdx = endIdx

	// Prepare positive pairs
	posTokenIDs := make([][]int32, actualBatchSize)
	posAttentionMask := make([][]int32, actualBatchSize)
	posTokenTypeIDs := make([][]int32, actualBatchSize)

	// Prepare negative pairs
	negTokenIDs := make([][]int32, actualBatchSize)
	negAttentionMask := make([][]int32, actualBatchSize)
	negTokenTypeIDs := make([][]int32, actualBatchSize)

	for i, sample := range batchSamples {
		posTokenIDs[i], posAttentionMask[i], posTokenTypeIDs[i] = d.encodeQueryDoc(sample.QueryTokens, sample.PositiveTokens)
		negTokenIDs[i], negAttentionMask[i], negTokenTypeIDs[i] = d.encodeQueryDoc(sample.QueryTokens, sample.NegativeTokens)
	}

	inputs = []*tensors.Tensor{
		tensors.FromFlatDataAndDimensions(flatten2DInt32(posTokenIDs), actualBatchSize, d.maxSeqLen),
		tensors.FromFlatDataAndDimensions(flatten2DInt32(posAttentionMask), actualBatchSize, d.maxSeqLen),
		tensors.FromFlatDataAndDimensions(flatten2DInt32(posTokenTypeIDs), actualBatchSize, d.maxSeqLen),
		tensors.FromFlatDataAndDimensions(flatten2DInt32(negTokenIDs), actualBatchSize, d.maxSeqLen),
		tensors.FromFlatDataAndDimensions(flatten2DInt32(negAttentionMask), actualBatchSize, d.maxSeqLen),
		tensors.FromFlatDataAndDimensions(flatten2DInt32(negTokenTypeIDs), actualBatchSize, d.maxSeqLen),
	}

	return inputs, nil, nil
}

// DatasetType returns DatasetPairwise.
func (d *PairwiseDataset) DatasetType() DatasetType {
	return DatasetPairwise
}

// encodeQueryDoc combines query and document tokens into BERT format.
func (d *PairwiseDataset) encodeQueryDoc(query, doc []int32) (tokenIDs, attentionMask, tokenTypeIDs []int32) {
	return encodeQueryDocPair(query, doc, d.maxSeqLen, d.clsTokenID, d.sepTokenID, d.padTokenID)
}

// Helper functions

// encodeQueryDocPair combines query and document tokens into BERT format.
// Format: [CLS] query [SEP] document [SEP] [PAD]...
//
// This shared helper is used by both PointwiseDataset and PairwiseDataset to avoid code duplication.
func encodeQueryDocPair(query, doc []int32, maxSeqLen int, clsTokenID, sepTokenID, padTokenID int32) (tokenIDs, attentionMask, tokenTypeIDs []int32) {
	tokenIDs = make([]int32, maxSeqLen)
	attentionMask = make([]int32, maxSeqLen)
	tokenTypeIDs = make([]int32, maxSeqLen)

	// Start with [CLS]
	idx := 0
	tokenIDs[idx] = clsTokenID
	attentionMask[idx] = 1
	tokenTypeIDs[idx] = 0
	idx++

	// Add query tokens
	for _, t := range query {
		if idx >= maxSeqLen-2 { // Leave room for [SEP] tokens
			break
		}
		tokenIDs[idx] = t
		attentionMask[idx] = 1
		tokenTypeIDs[idx] = 0
		idx++
	}

	// Add [SEP] after query
	if idx < maxSeqLen {
		tokenIDs[idx] = sepTokenID
		attentionMask[idx] = 1
		tokenTypeIDs[idx] = 0
		idx++
	}

	// Add document tokens
	for _, t := range doc {
		if idx >= maxSeqLen-1 { // Leave room for final [SEP]
			break
		}
		tokenIDs[idx] = t
		attentionMask[idx] = 1
		tokenTypeIDs[idx] = 1 // Different segment for document
		idx++
	}

	// Add final [SEP]
	if idx < maxSeqLen {
		tokenIDs[idx] = sepTokenID
		attentionMask[idx] = 1
		tokenTypeIDs[idx] = 1
		idx++
	}

	// Pad the rest
	for ; idx < maxSeqLen; idx++ {
		tokenIDs[idx] = padTokenID
		attentionMask[idx] = 0
		tokenTypeIDs[idx] = 0
	}

	return tokenIDs, attentionMask, tokenTypeIDs
}

func flatten2DInt32(data [][]int32) []int32 {
	if len(data) == 0 {
		return nil
	}
	seqLen := len(data[0])
	flat := make([]int32, len(data)*seqLen)
	for i, row := range data {
		// Validate that all rows have the same length as the first row
		if len(row) != seqLen {
			panic(fmt.Sprintf("flatten2DInt32: inconsistent row lengths - expected %d, got %d at row %d", seqLen, len(row), i))
		}
		copy(flat[i*seqLen:], row)
	}
	return flat
}
