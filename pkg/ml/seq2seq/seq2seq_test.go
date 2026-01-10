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
	"math"
	"math/rand"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/tensors"

	_ "github.com/gomlx/gomlx/backends/default"
)

func TestSoftmax(t *testing.T) {
	tests := []struct {
		name   string
		logits []float32
	}{
		{
			name:   "simple",
			logits: []float32{1.0, 2.0, 3.0},
		},
		{
			name:   "larger values",
			logits: []float32{10.0, 20.0, 30.0},
		},
		{
			name:   "negative values",
			logits: []float32{-1.0, 0.0, 1.0},
		},
		{
			name:   "all same",
			logits: []float32{5.0, 5.0, 5.0, 5.0},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			probs := softmax(tc.logits)

			// Check that probabilities sum to 1.
			var sum float32
			for _, p := range probs {
				sum += p
			}
			if math.Abs(float64(sum-1.0)) > 1e-5 {
				t.Errorf("softmax probabilities sum to %f, expected 1.0", sum)
			}

			// Check that all probabilities are positive.
			for i, p := range probs {
				if p < 0 {
					t.Errorf("softmax prob[%d] = %f, expected >= 0", i, p)
				}
			}

			// Check that larger logits produce larger probabilities.
			for i := 0; i < len(tc.logits)-1; i++ {
				if tc.logits[i] < tc.logits[i+1] && probs[i] > probs[i+1] {
					t.Errorf("expected probs[%d] <= probs[%d] since logits[%d] <= logits[%d]",
						i, i+1, i, i+1)
				}
			}
		})
	}
}

func TestSampleTopP(t *testing.T) {
	// Test with deterministic probabilities.
	rng := rand.New(rand.NewSource(42))
	probs := []float32{0.7, 0.2, 0.1}

	// With topP = 0.5, should almost always return index 0.
	counts := make(map[int]int)
	for i := 0; i < 1000; i++ {
		idx := sampleFromTopProbs(probs, 0.5, 0, rng)
		counts[idx]++
	}

	// Index 0 should have the vast majority of samples.
	if counts[0] < 900 {
		t.Errorf("expected index 0 to dominate with topP=0.5, got counts: %v", counts)
	}
}

func TestSampleTopK(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	probs := []float32{0.1, 0.3, 0.2, 0.4}

	// With topK = 2, should only sample indices 1 and 3 (top 2 by probability).
	counts := make(map[int]int)
	for i := 0; i < 1000; i++ {
		idx := sampleFromTopProbs(probs, 1.0, 2, rng)
		counts[idx]++
	}

	// Should not sample indices 0 or 2.
	if counts[0] > 0 || counts[2] > 0 {
		t.Errorf("topK=2 should not sample indices 0 or 2, got counts: %v", counts)
	}
}

func TestArgmaxFromLogits(t *testing.T) {
	tests := []struct {
		name     string
		logits   []float32
		dims     []int
		expected []int32
	}{
		{
			name:     "simple 2D",
			logits:   []float32{1.0, 3.0, 2.0, 4.0, 1.0, 2.0},
			dims:     []int{2, 3},
			expected: []int32{1, 0},
		},
		{
			name:     "3D with seq_len=1",
			logits:   []float32{1.0, 5.0, 2.0, 2.0, 1.0, 3.0},
			dims:     []int{2, 1, 3},
			expected: []int32{1, 2},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			logitsTensor := tensors.FromFlatDataAndDimensions(tc.logits, tc.dims...)

			tokens, err := argmaxFromLogits(logitsTensor)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if len(tokens) != len(tc.expected) {
				t.Fatalf("expected %d tokens, got %d", len(tc.expected), len(tokens))
			}

			for i, tok := range tokens {
				if tok != tc.expected[i] {
					t.Errorf("token[%d] = %d, expected %d", i, tok, tc.expected[i])
				}
			}
		})
	}
}

func TestKVCache(t *testing.T) {
	numLayers := 4
	kv := NewKVCache(numLayers)

	if kv.NumLayers != numLayers {
		t.Errorf("expected NumLayers=%d, got %d", numLayers, kv.NumLayers)
	}

	if len(kv.SelfAttentionKeys) != numLayers {
		t.Errorf("expected %d self-attention key slots, got %d", numLayers, len(kv.SelfAttentionKeys))
	}

	// Test setting and getting.
	dummyTensor := tensors.FromFlatDataAndDimensions([]float32{1.0, 2.0, 3.0, 4.0}, 1, 2, 1, 2)
	kv.SetSelfAttention(0, dummyTensor, dummyTensor)

	keys, values := kv.GetSelfAttentionKV()
	if keys[0] != dummyTensor || values[0] != dummyTensor {
		t.Error("self-attention KV not set correctly")
	}

	// Test finalize.
	kv.Finalize()
	if kv.SelfAttentionKeys[0] != nil {
		t.Error("finalize should nil out tensors")
	}
}

func TestModelConfig(t *testing.T) {
	config := DefaultModelConfig()

	if config.DType != dtypes.Float32 {
		t.Errorf("expected default DType=Float32, got %s", config.DType)
	}

	if config.MaxLength != 512 {
		t.Errorf("expected default MaxLength=512, got %d", config.MaxLength)
	}
}

func TestGenerationConfig(t *testing.T) {
	config := DefaultGenerationConfig()

	if config.Temperature != 1.0 {
		t.Errorf("expected default Temperature=1.0, got %f", config.Temperature)
	}

	if config.TopP != 1.0 {
		t.Errorf("expected default TopP=1.0, got %f", config.TopP)
	}

	if config.DoSample {
		t.Error("expected DoSample=false by default")
	}
}

func TestApplyRepetitionPenalty(t *testing.T) {
	logits := []float32{1.0, 2.0, 3.0, 4.0, 5.0}
	generatedIDs := []int32{1, 3}
	penalty := float32(2.0)

	ApplyRepetitionPenalty(logits, generatedIDs, penalty)

	// Check that penalized positions are reduced.
	if logits[1] != 1.0 { // 2.0 / 2.0 = 1.0
		t.Errorf("expected logits[1]=1.0, got %f", logits[1])
	}
	if logits[3] != 2.0 { // 4.0 / 2.0 = 2.0
		t.Errorf("expected logits[3]=2.0, got %f", logits[3])
	}

	// Check that non-penalized positions are unchanged.
	if logits[0] != 1.0 || logits[2] != 3.0 || logits[4] != 5.0 {
		t.Error("non-penalized positions should be unchanged")
	}
}

func TestTensorUtilities(t *testing.T) {
	t.Run("CreateInt32Tensor", func(t *testing.T) {
		data := []int32{1, 2, 3, 4, 5, 6}
		tensor := CreateInt32Tensor(data, 2, 3)
		if tensor.Shape().Rank() != 2 {
			t.Errorf("expected rank 2, got %d", tensor.Shape().Rank())
		}
		if tensor.Shape().Dimensions[0] != 2 || tensor.Shape().Dimensions[1] != 3 {
			t.Errorf("unexpected shape: %v", tensor.Shape())
		}
	})

	t.Run("CreateFloat32Tensor", func(t *testing.T) {
		data := []float32{1.0, 2.0, 3.0, 4.0}
		tensor := CreateFloat32Tensor(data, 4)
		if tensor.Shape().Rank() != 1 {
			t.Errorf("expected rank 1, got %d", tensor.Shape().Rank())
		}
	})

	t.Run("CreateZerosTensor", func(t *testing.T) {
		tensor := CreateZerosTensor(dtypes.Float32, 2, 3)
		if tensor.Shape().DType != dtypes.Float32 {
			t.Errorf("expected Float32, got %s", tensor.Shape().DType)
		}
	})

	t.Run("CreateOnesTensor", func(t *testing.T) {
		tensor := CreateOnesTensor(dtypes.Float32, 3, 2)
		data, err := TensorToFloat32Slice(tensor)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		for i, v := range data {
			if v != 1.0 {
				t.Errorf("expected data[%d]=1.0, got %f", i, v)
			}
		}
	})
}

func TestFloat16Conversion(t *testing.T) {
	tests := []struct {
		name     string
		fp16     uint16
		expected float32
	}{
		{"zero", 0x0000, 0.0},
		{"one", 0x3C00, 1.0},
		{"negative one", 0xBC00, -1.0},
		{"two", 0x4000, 2.0},
		{"half", 0x3800, 0.5},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := float16ToFloat32(tc.fp16)
			if math.Abs(float64(result-tc.expected)) > 1e-3 {
				t.Errorf("float16ToFloat32(0x%04X) = %f, expected %f", tc.fp16, result, tc.expected)
			}
		})
	}
}

func TestValidateKVCacheShape(t *testing.T) {
	// Create a valid KV cache tensor.
	batchSize, numHeads, seqLen, headDim := 2, 8, 10, 64
	tensor := CreateZerosTensor(dtypes.Float32, batchSize, numHeads, seqLen, headDim)

	if !ValidateKVCacheShape(tensor, batchSize, numHeads, headDim) {
		t.Error("expected valid KV cache shape")
	}

	// Test with wrong batch size.
	if ValidateKVCacheShape(tensor, batchSize+1, numHeads, headDim) {
		t.Error("expected invalid with wrong batch size")
	}

	// Test with wrong rank.
	wrongRankTensor := CreateZerosTensor(dtypes.Float32, batchSize, numHeads, seqLen)
	if ValidateKVCacheShape(wrongRankTensor, batchSize, numHeads, headDim) {
		t.Error("expected invalid with wrong rank")
	}
}

func TestBatchBasics(t *testing.T) {
	// Note: This test doesn't use a real backend, so it just tests the struct.
	inputIDs := tensors.FromFlatDataAndDimensions([]int32{1, 2, 3, 4, 5, 6}, 2, 3)

	batch := &Batch{
		InputIDs:      inputIDs,
		batchSize:     2,
		encoderSeqLen: 3,
	}

	if batch.BatchSize() != 2 {
		t.Errorf("expected BatchSize=2, got %d", batch.BatchSize())
	}

	if batch.EncoderSeqLen() != 3 {
		t.Errorf("expected EncoderSeqLen=3, got %d", batch.EncoderSeqLen())
	}

	if batch.EncoderRun() {
		t.Error("encoder should not be run initially")
	}

	if batch.DecoderInitialized() {
		t.Error("decoder should not be initialized initially")
	}
}

func TestExtractLogitsData(t *testing.T) {
	t.Run("2D float32", func(t *testing.T) {
		logits := tensors.FromFlatDataAndDimensions([]float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, 2, 3)
		batchSize, vocabSize, data, err := extractLogitsData(logits)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if batchSize != 2 || vocabSize != 3 {
			t.Errorf("expected batchSize=2, vocabSize=3, got %d, %d", batchSize, vocabSize)
		}
		if len(data) != 2 {
			t.Fatalf("expected 2 batch slices, got %d", len(data))
		}
		// Check first batch
		if data[0][0] != 1.0 || data[0][1] != 2.0 || data[0][2] != 3.0 {
			t.Errorf("unexpected batch 0 data: %v", data[0])
		}
	})

	t.Run("3D float32 last position", func(t *testing.T) {
		// Shape: [2, 2, 3] - batch=2, seq_len=2, vocab=3
		logits := tensors.FromFlatDataAndDimensions([]float32{
			// batch 0, pos 0
			1.0, 2.0, 3.0,
			// batch 0, pos 1 (last)
			4.0, 5.0, 6.0,
			// batch 1, pos 0
			7.0, 8.0, 9.0,
			// batch 1, pos 1 (last)
			10.0, 11.0, 12.0,
		}, 2, 2, 3)
		batchSize, vocabSize, data, err := extractLogitsData(logits)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if batchSize != 2 || vocabSize != 3 {
			t.Errorf("expected batchSize=2, vocabSize=3, got %d, %d", batchSize, vocabSize)
		}
		// Should extract last position for each batch
		if data[0][0] != 4.0 || data[0][1] != 5.0 || data[0][2] != 6.0 {
			t.Errorf("expected batch 0 to be [4, 5, 6], got %v", data[0])
		}
		if data[1][0] != 10.0 || data[1][1] != 11.0 || data[1][2] != 12.0 {
			t.Errorf("expected batch 1 to be [10, 11, 12], got %v", data[1])
		}
	})

	t.Run("unsupported rank", func(t *testing.T) {
		logits := tensors.FromFlatDataAndDimensions([]float32{1.0, 2.0, 3.0}, 3)
		_, _, _, err := extractLogitsData(logits)
		if err == nil {
			t.Error("expected error for rank-1 tensor")
		}
	})
}

func TestTensorConversionErrors(t *testing.T) {
	t.Run("TensorToFloat32Slice unsupported dtype", func(t *testing.T) {
		tensor := tensors.FromFlatDataAndDimensions([]int32{1, 2, 3}, 3)
		_, err := TensorToFloat32Slice(tensor)
		if err == nil {
			t.Error("expected error for int32 tensor")
		}
	})

	t.Run("TensorToInt32Slice unsupported dtype", func(t *testing.T) {
		tensor := tensors.FromFlatDataAndDimensions([]float32{1.0, 2.0, 3.0}, 3)
		_, err := TensorToInt32Slice(tensor)
		if err == nil {
			t.Error("expected error for float32 tensor")
		}
	})
}

func TestPositionalEncoding(t *testing.T) {
	seqLen := 4
	embeddingDim := 8

	graphtest.RunTestGraphFn(t, "PositionalEncoding",
		func(g *graph.Graph) (inputs, outputs []*graph.Node) {
			pe := CreatePositionalEncoding(g, seqLen, embeddingDim, dtypes.Float32)
			pe.AssertDims(seqLen, embeddingDim)

			// Return the shape dimensions to verify output shape.
			shapeNode := graph.Const(g, []int32{int32(seqLen), int32(embeddingDim)})
			return nil, []*graph.Node{shapeNode}
		},
		[]any{
			[]int32{4, 8},
		},
		0.0,
	)
}

func TestPositionalEncodingProperties(t *testing.T) {
	seqLen := 10
	embeddingDim := 16

	graphtest.RunTestGraphFn(t, "PositionalEncodingRange",
		func(g *graph.Graph) (inputs, outputs []*graph.Node) {
			pe := CreatePositionalEncoding(g, seqLen, embeddingDim, dtypes.Float32)

			// Check that values are in [-1, 1] range (sin/cos output).
			minVal := graph.ReduceMin(pe)
			maxVal := graph.ReduceMax(pe)

			// Check min >= -1 and max <= 1 using comparisons that return 1 if true.
			// ge(-1) means minVal >= -1
			minOk := graph.GreaterOrEqual(minVal, graph.Scalar(g, dtypes.Float32, -1.0))
			maxOk := graph.LessOrEqual(maxVal, graph.Scalar(g, dtypes.Float32, 1.0))

			return nil, []*graph.Node{minOk, maxOk}
		},
		[]any{
			true, // min >= -1
			true, // max <= 1
		},
		0.0,
	)
}

// Note: TransformerEncoderLayer, TransformerDecoderLayer, and CreateEmbedding
// create context variables that require context.Exec for proper initialization.
// Shape inference is tested implicitly via the AssertDims calls during graph
// building, which happen in the positional encoding and other tests.
// Full integration tests would require setting up a complete Model with Exec.

func TestSampleFromTopProbs(t *testing.T) {
	// Test with deterministic RNG.
	rng := rand.New(rand.NewSource(42))
	probs := []float32{0.1, 0.2, 0.3, 0.4}

	// With high topP, should sample from all tokens.
	counts := make(map[int]int)
	for i := 0; i < 1000; i++ {
		idx := sampleFromTopProbs(probs, 1.0, 0, rng)
		counts[idx]++
	}

	// All indices should be sampled (with probability proportional to their probs).
	for i := 0; i < 4; i++ {
		if counts[i] == 0 {
			t.Errorf("expected index %d to be sampled at least once", i)
		}
	}

	// Higher probability tokens should be sampled more.
	if counts[0] > counts[3] {
		t.Errorf("expected index 3 (p=0.4) to be sampled more than index 0 (p=0.1), got %d vs %d",
			counts[3], counts[0])
	}
}

func TestSampleFromTopProbsWithTopK(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	probs := []float32{0.1, 0.2, 0.3, 0.4}

	// With topK=2, should only sample from indices 2 and 3.
	counts := make(map[int]int)
	for i := 0; i < 1000; i++ {
		idx := sampleFromTopProbs(probs, 1.0, 2, rng)
		counts[idx]++
	}

	// Should not sample indices 0 or 1.
	if counts[0] > 0 || counts[1] > 0 {
		t.Errorf("topK=2 should not sample indices 0 or 1, got counts: %v", counts)
	}

	// Should sample both 2 and 3.
	if counts[2] == 0 || counts[3] == 0 {
		t.Errorf("expected both indices 2 and 3 to be sampled, got counts: %v", counts)
	}
}

func TestSampleFromTopProbsWithTopP(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	probs := []float32{0.05, 0.1, 0.35, 0.5}

	// With topP=0.6, should only sample from indices 2 and 3 (cumulative 0.85 > 0.6).
	// Actually with sorted order: 0.5, 0.35, 0.1, 0.05
	// Cumulative: 0.5 (idx 3), 0.85 (idx 2) -> cutoff at 0.6 means only idx 3 most of the time.
	counts := make(map[int]int)
	for i := 0; i < 1000; i++ {
		idx := sampleFromTopProbs(probs, 0.6, 0, rng)
		counts[idx]++
	}

	// Index 3 (highest prob) should dominate.
	if counts[3] < 500 {
		t.Errorf("expected index 3 to dominate with topP=0.6, got counts: %v", counts)
	}
}

func TestBuildGreedyPostProcessGraph(t *testing.T) {
	// Test 2D logits [batch_size, vocab_size]
	t.Run("2D logits", func(t *testing.T) {
		graphtest.RunTestGraphFn(t, "Greedy2D",
			func(g *graph.Graph) (inputs, outputs []*graph.Node) {
				// Batch size 2, vocab size 4
				// Batch 0: max at index 2 (value 3.0)
				// Batch 1: max at index 0 (value 4.0)
				logits := graph.Const(g, [][]float32{
					{1.0, 2.0, 3.0, 0.5},
					{4.0, 1.0, 2.0, 3.0},
				})
				tokenIDs := buildGreedyPostProcessGraph(logits)
				return nil, []*graph.Node{tokenIDs}
			},
			[]any{
				[]int32{2, 0},
			},
			0.0,
		)
	})

	// Test 3D logits [batch_size, seq_len, vocab_size]
	t.Run("3D logits", func(t *testing.T) {
		graphtest.RunTestGraphFn(t, "Greedy3D",
			func(g *graph.Graph) (inputs, outputs []*graph.Node) {
				// Batch size 2, seq_len 1, vocab size 3
				logits := graph.Const(g, [][][]float32{
					{{1.0, 5.0, 2.0}}, // max at index 1
					{{3.0, 1.0, 2.0}}, // max at index 0
				})
				tokenIDs := buildGreedyPostProcessGraph(logits)
				return nil, []*graph.Node{tokenIDs}
			},
			[]any{
				[]int32{1, 0},
			},
			0.0,
		)
	})
}

func TestBuildSamplingPostProcessGraph(t *testing.T) {
	// Test that TopK returns correct shape and values
	t.Run("TopK shape and values", func(t *testing.T) {
		graphtest.RunTestGraphFn(t, "SamplingTopK",
			func(g *graph.Graph) (inputs, outputs []*graph.Node) {
				// Batch size 1, vocab size 5
				// After softmax with temp=1, we get probabilities
				logits := graph.Const(g, [][]float32{
					{0.0, 1.0, 2.0, 3.0, 4.0}, // Highest at index 4
				})
				temperature := graph.Scalar(g, dtypes.Float32, 1.0)

				topKProbs, topKIndices := buildSamplingPostProcessGraph(logits, temperature, 3)

				// Check shapes
				topKProbs.AssertDims(1, 3)
				topKIndices.AssertDims(1, 3)

				// The top 3 indices should be 4, 3, 2 (in descending prob order)
				return nil, []*graph.Node{topKIndices}
			},
			[]any{
				[][]int32{{4, 3, 2}},
			},
			0.0,
		)
	})

	// Test temperature scaling
	t.Run("Temperature scaling", func(t *testing.T) {
		graphtest.RunTestGraphFn(t, "SamplingTemp",
			func(g *graph.Graph) (inputs, outputs []*graph.Node) {
				// With high temperature, probabilities should be more uniform
				logits := graph.Const(g, [][]float32{
					{0.0, 10.0}, // Without temp: very skewed. With high temp: more uniform
				})

				// Low temperature (sharp distribution)
				tempLow := graph.Scalar(g, dtypes.Float32, 0.1)
				probsLow, _ := buildSamplingPostProcessGraph(logits, tempLow, 2)
				maxProbLow := graph.ReduceMax(probsLow)

				// High temperature (flatter distribution)
				tempHigh := graph.Scalar(g, dtypes.Float32, 10.0)
				probsHigh, _ := buildSamplingPostProcessGraph(logits, tempHigh, 2)
				maxProbHigh := graph.ReduceMax(probsHigh)

				// Low temp should have higher max prob than high temp
				lowMoreConcentrated := graph.GreaterThan(maxProbLow, maxProbHigh)

				return nil, []*graph.Node{lowMoreConcentrated}
			},
			[]any{
				true,
			},
			0.0,
		)
	})
}
