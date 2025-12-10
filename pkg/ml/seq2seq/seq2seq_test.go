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
	"testing"

	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gopjrt/dtypes"
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
	probs := []float32{0.7, 0.2, 0.1}

	// With topP = 0.5, should almost always return index 0.
	counts := make(map[int]int)
	for i := 0; i < 1000; i++ {
		idx := sampleTopP(probs, 0.5)
		counts[idx]++
	}

	// Index 0 should have the vast majority of samples.
	if counts[0] < 900 {
		t.Errorf("expected index 0 to dominate with topP=0.5, got counts: %v", counts)
	}
}

func TestSampleTopK(t *testing.T) {
	probs := []float32{0.1, 0.3, 0.2, 0.4}

	// With topK = 2, should only sample indices 1 and 3 (top 2 by probability).
	counts := make(map[int]int)
	for i := 0; i < 1000; i++ {
		idx := sampleTopK(probs, 2)
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
