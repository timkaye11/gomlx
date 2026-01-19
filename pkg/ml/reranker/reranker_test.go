// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package reranker

import (
	"testing"

	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCrossEncoderBuilder(t *testing.T) {
	ctx := context.New()

	// Test builder configuration
	builder := CrossEncoder(ctx).
		NumLayers(2).
		NumHeads(4).
		HiddenDim(64).
		VocabSize(1000).
		MaxSeqLen(32).
		DropoutRate(0.0). // Disable dropout for testing
		Pooling(PoolingCLS)

	assert.Equal(t, 2, builder.config.NumLayers)
	assert.Equal(t, 4, builder.config.NumHeads)
	assert.Equal(t, 64, builder.config.HiddenDim)
	assert.Equal(t, 1000, builder.config.VocabSize)
	assert.Equal(t, 32, builder.config.MaxSeqLen)
	assert.Equal(t, PoolingCLS, builder.config.Pooling)
}

func TestPoolingTypes(t *testing.T) {
	// Test pooling type string conversion
	assert.Equal(t, PoolingCLS, PoolingType("cls"))
	assert.Equal(t, PoolingMean, PoolingType("mean"))
	assert.Equal(t, PoolingMax, PoolingType("max"))
}

func TestCrossEncoderConfig(t *testing.T) {
	ctx := context.New()

	// Test default values
	builder := CrossEncoder(ctx)
	assert.Equal(t, 6, builder.config.NumLayers)
	assert.Equal(t, 8, builder.config.NumHeads)
	assert.Equal(t, 768, builder.config.HiddenDim)
	assert.Equal(t, 0.1, builder.config.DropoutRate)
	assert.Equal(t, 30522, builder.config.VocabSize)
	assert.Equal(t, 512, builder.config.MaxSeqLen)
	assert.Equal(t, 2, builder.config.NumTypeIDs)

	// Test context-based configuration
	ctx2 := context.New()
	ctx2.SetParam(ParamNumLayers, 4)
	ctx2.SetParam(ParamNumHeads, 12)
	ctx2.SetParam(ParamHiddenDim, 384)

	builder2 := CrossEncoder(ctx2).FromContext()
	assert.Equal(t, 4, builder2.config.NumLayers)
	assert.Equal(t, 12, builder2.config.NumHeads)
	assert.Equal(t, 384, builder2.config.HiddenDim)
}

func TestPreNormConfig(t *testing.T) {
	ctx := context.New()

	// Test post-norm (default)
	builder := CrossEncoder(ctx)
	assert.False(t, builder.config.PreNorm)

	// Test pre-norm
	builder = CrossEncoder(ctx).PreNorm(true)
	assert.True(t, builder.config.PreNorm)
}

func TestPointwiseDataset(t *testing.T) {
	samples := []PointwiseSample{
		{QueryTokens: []int32{1, 2, 3}, DocTokens: []int32{4, 5, 6}, Label: 1.0},
		{QueryTokens: []int32{7, 8}, DocTokens: []int32{9, 10, 11, 12}, Label: 0.0},
		{QueryTokens: []int32{13, 14, 15}, DocTokens: []int32{16}, Label: 1.0},
	}

	ds := NewPointwiseDataset(samples, 20, 101, 102, 0, 2) // CLS=101, SEP=102, PAD=0
	ds.SetInfinite(false)

	assert.Equal(t, "pointwise_ranking", ds.Name())
	assert.Equal(t, DatasetPointwise, ds.DatasetType())

	// First batch
	inputs, labels, err := ds.Yield()
	require.NoError(t, err)
	require.Len(t, inputs, 3)
	require.Len(t, labels, 1)

	// Check shapes
	assert.Equal(t, []int{2, 20}, inputs[0].Shape().Dimensions) // token_ids
	assert.Equal(t, []int{2, 20}, inputs[1].Shape().Dimensions) // attention_mask
	assert.Equal(t, []int{2, 20}, inputs[2].Shape().Dimensions) // token_type_ids
	assert.Equal(t, []int{2, 1}, labels[0].Shape().Dimensions)  // labels

	// Second batch (partial)
	inputs, labels, err = ds.Yield()
	require.NoError(t, err)
	require.Len(t, inputs, 3)
	assert.Equal(t, []int{1, 20}, inputs[0].Shape().Dimensions) // Only 1 sample left

	// End of data
	inputs, labels, err = ds.Yield()
	require.NoError(t, err)
	assert.Nil(t, inputs)
	assert.Nil(t, labels)
}

func TestPointwiseDatasetInfinite(t *testing.T) {
	samples := []PointwiseSample{
		{QueryTokens: []int32{1, 2}, DocTokens: []int32{3, 4}, Label: 1.0},
	}

	ds := NewPointwiseDataset(samples, 10, 101, 102, 0, 1)
	ds.SetInfinite(true)

	// Should loop infinitely
	for i := 0; i < 5; i++ {
		inputs, _, err := ds.Yield()
		require.NoError(t, err)
		require.NotNil(t, inputs)
	}
}

func TestPointwiseDatasetEncoding(t *testing.T) {
	samples := []PointwiseSample{
		{QueryTokens: []int32{1, 2, 3}, DocTokens: []int32{4, 5}, Label: 1.0},
	}

	ds := NewPointwiseDataset(samples, 12, 101, 102, 0, 1)

	inputs, _, err := ds.Yield()
	require.NoError(t, err)

	// Extract token_ids from first sample - value is [][]int32 because shape is [1, 12]
	tokenIDsAll := inputs[0].Value().([][]int32)
	attentionMaskAll := inputs[1].Value().([][]int32)
	tokenTypeIDsAll := inputs[2].Value().([][]int32)

	tokenIDs := tokenIDsAll[0]
	attentionMask := attentionMaskAll[0]
	tokenTypeIDs := tokenTypeIDsAll[0]

	// Expected: [CLS] 1 2 3 [SEP] 4 5 [SEP] [PAD] [PAD] [PAD] [PAD]
	assert.Equal(t, int32(101), tokenIDs[0])  // CLS
	assert.Equal(t, int32(1), tokenIDs[1])    // Query token
	assert.Equal(t, int32(2), tokenIDs[2])    // Query token
	assert.Equal(t, int32(3), tokenIDs[3])    // Query token
	assert.Equal(t, int32(102), tokenIDs[4])  // SEP
	assert.Equal(t, int32(4), tokenIDs[5])    // Doc token
	assert.Equal(t, int32(5), tokenIDs[6])    // Doc token
	assert.Equal(t, int32(102), tokenIDs[7])  // SEP
	assert.Equal(t, int32(0), tokenIDs[8])    // PAD
	assert.Equal(t, int32(0), tokenIDs[9])    // PAD
	assert.Equal(t, int32(0), tokenIDs[10])   // PAD
	assert.Equal(t, int32(0), tokenIDs[11])   // PAD

	// Check attention mask
	for i := 0; i < 8; i++ {
		assert.Equal(t, int32(1), attentionMask[i], "attention mask at %d", i)
	}
	for i := 8; i < 12; i++ {
		assert.Equal(t, int32(0), attentionMask[i], "attention mask at %d", i)
	}

	// Check token type IDs (0 for query, 1 for document)
	for i := 0; i <= 4; i++ { // CLS, query tokens, first SEP
		assert.Equal(t, int32(0), tokenTypeIDs[i], "token type at %d", i)
	}
	for i := 5; i <= 7; i++ { // Doc tokens, final SEP
		assert.Equal(t, int32(1), tokenTypeIDs[i], "token type at %d", i)
	}
}

func TestPairwiseDataset(t *testing.T) {
	samples := []PairwiseSample{
		{
			QueryTokens:    []int32{1, 2},
			PositiveTokens: []int32{3, 4},
			NegativeTokens: []int32{5, 6},
		},
		{
			QueryTokens:    []int32{7, 8},
			PositiveTokens: []int32{9, 10},
			NegativeTokens: []int32{11, 12},
		},
	}

	ds := NewPairwiseDataset(samples, 15, 101, 102, 0, 2)
	ds.SetInfinite(false)

	assert.Equal(t, "pairwise_ranking", ds.Name())
	assert.Equal(t, DatasetPairwise, ds.DatasetType())

	inputs, labels, err := ds.Yield()
	require.NoError(t, err)
	require.Len(t, inputs, 6) // 3 for positive, 3 for negative
	assert.Nil(t, labels)     // Pairwise doesn't have explicit labels

	// Check shapes
	assert.Equal(t, []int{2, 15}, inputs[0].Shape().Dimensions) // pos_token_ids
	assert.Equal(t, []int{2, 15}, inputs[1].Shape().Dimensions) // pos_attention_mask
	assert.Equal(t, []int{2, 15}, inputs[2].Shape().Dimensions) // pos_token_type_ids
	assert.Equal(t, []int{2, 15}, inputs[3].Shape().Dimensions) // neg_token_ids
	assert.Equal(t, []int{2, 15}, inputs[4].Shape().Dimensions) // neg_attention_mask
	assert.Equal(t, []int{2, 15}, inputs[5].Shape().Dimensions) // neg_token_type_ids
}

func TestDatasetTypeString(t *testing.T) {
	assert.Equal(t, "pointwise", DatasetPointwise.String())
	assert.Equal(t, "pairwise", DatasetPairwise.String())
	assert.Equal(t, "listwise", DatasetListwise.String())
	assert.Equal(t, "unknown", DatasetType(-1).String())
}

func TestHardNegativeMiner(t *testing.T) {
	config := HardNegativeConfig{
		NumHardNegatives:   2,
		NumRandomNegatives: 1,
		MinScore:           0.2,
		MaxScore:           0.7,
	}

	miner := NewHardNegativeMiner(config, nil)

	query := QueryDocuments{
		QueryID: 1,
		Documents: []ScoredDocument{
			{DocID: 1, Score: 0.9},  // Positive (score >= 0.7)
			{DocID: 2, Score: 0.8},  // Positive
			{DocID: 3, Score: 0.6},  // Hard negative (0.2 <= score < 0.7)
			{DocID: 4, Score: 0.5},  // Hard negative
			{DocID: 5, Score: 0.3},  // Hard negative
			{DocID: 6, Score: 0.15}, // Easy negative (score < 0.2)
			{DocID: 7, Score: 0.1},  // Easy negative
		},
	}

	result := miner.Mine(query)

	assert.Equal(t, 1, result.QueryID)
	assert.Len(t, result.HardNegatives, 2)
	assert.Len(t, result.EasyNegatives, 1)

	// Hard negatives should be sorted by score descending
	assert.Equal(t, 3, result.HardNegatives[0].DocID)
	assert.Equal(t, float32(0.6), result.HardNegatives[0].Score)
	assert.Equal(t, 4, result.HardNegatives[1].DocID)
	assert.Equal(t, float32(0.5), result.HardNegatives[1].Score)
}

func TestHardNegativeMinerWithRandom(t *testing.T) {
	config := DefaultHardNegativeConfig()

	// Create a mock random source
	mockRng := &mockRandom{shuffleCalled: false}
	miner := NewHardNegativeMiner(config, mockRng)

	query := QueryDocuments{
		QueryID: 1,
		Documents: []ScoredDocument{
			{DocID: 1, Score: 0.6},  // Positive
			{DocID: 2, Score: 0.3},  // Hard negative
			{DocID: 3, Score: 0.05}, // Easy negative
		},
	}

	miner.Mine(query)
	assert.True(t, mockRng.shuffleCalled)
}

type mockRandom struct {
	shuffleCalled bool
}

func (m *mockRandom) Intn(n int) int {
	return 0
}

func (m *mockRandom) Shuffle(n int, swap func(i, j int)) {
	m.shuffleCalled = true
}

func TestMineAll(t *testing.T) {
	config := DefaultHardNegativeConfig()
	miner := NewHardNegativeMiner(config, nil)

	queries := []QueryDocuments{
		{QueryID: 1, Documents: []ScoredDocument{{DocID: 1, Score: 0.3}}},
		{QueryID: 2, Documents: []ScoredDocument{{DocID: 2, Score: 0.2}}},
	}

	results := miner.MineAll(queries)
	assert.Len(t, results, 2)
	assert.Equal(t, 1, results[0].QueryID)
	assert.Equal(t, 2, results[1].QueryID)
}

func TestInBatchNegatives(t *testing.T) {
	// Simulated score matrix: scores[i][j] = score of doc j to query i
	scores := [][]float32{
		{1.0, 0.5, 0.3}, // Query 0: doc 0 is positive, docs 1,2 are negatives
		{0.4, 1.0, 0.6}, // Query 1: doc 1 is positive, docs 0,2 are negatives
		{0.2, 0.7, 1.0}, // Query 2: doc 2 is positive, docs 0,1 are negatives
	}

	negatives := InBatchNegatives(scores, 2)

	assert.Len(t, negatives, 3)

	// Query 0: hardest negatives are doc 1 (0.5) and doc 2 (0.3)
	assert.Equal(t, []int{1, 2}, negatives[0])

	// Query 1: hardest negatives are doc 2 (0.6) and doc 0 (0.4)
	assert.Equal(t, []int{2, 0}, negatives[1])

	// Query 2: hardest negatives are doc 1 (0.7) and doc 0 (0.2)
	assert.Equal(t, []int{1, 0}, negatives[2])
}

func TestInBatchNegativesLimited(t *testing.T) {
	scores := [][]float32{
		{1.0, 0.5, 0.3, 0.8},
		{0.4, 1.0, 0.6, 0.2},
		{0.2, 0.7, 1.0, 0.9},
		{0.3, 0.4, 0.5, 1.0},
	}

	// Request only 1 negative
	negatives := InBatchNegatives(scores, 1)

	assert.Len(t, negatives, 4)
	for _, negs := range negatives {
		assert.Len(t, negs, 1)
	}

	// Query 0: hardest negative is doc 3 (0.8)
	assert.Equal(t, []int{3}, negatives[0])
}
