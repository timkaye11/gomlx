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
	"sort"

	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/pkg/errors"
)

// indexedProb pairs a token index with its probability.
// Used internally for sorting tokens by probability during sampling.
type indexedProb struct {
	index int     // Token index in vocabulary
	prob  float32 // Probability of this token
}

// extractLogitsData extracts logits tensor data as float32 slices per batch item.
// Returns the batch size, vocab size, and a slice of float32 slices (one per batch item).
// For 3D tensors [batch, seq_len, vocab], extracts only the last position.
func extractLogitsData(logits *tensors.Tensor) (batchSize, vocabSize int, batchLogits [][]float32, err error) {
	shape := logits.Shape()
	if shape.Rank() < 2 || shape.Rank() > 3 {
		return 0, 0, nil, errors.Errorf("expected logits rank 2 or 3, got %d", shape.Rank())
	}

	batchSize = shape.Dimensions[0]
	vocabSize = shape.Dimensions[shape.Rank()-1]

	// Use TensorToFloat32Slice for dtype conversion.
	logitsData, err := TensorToFloat32Slice(logits)
	if err != nil {
		return 0, 0, nil, errors.WithMessage(err, "failed to convert logits to float32")
	}

	// Split into per-batch slices, taking only the last position for 3D tensors.
	batchLogits = make([][]float32, batchSize)
	for batch := 0; batch < batchSize; batch++ {
		var offset int
		if shape.Rank() == 3 {
			seqLen := shape.Dimensions[1]
			offset = batch*seqLen*vocabSize + (seqLen-1)*vocabSize
		} else {
			offset = batch * vocabSize
		}
		batchLogits[batch] = logitsData[offset : offset+vocabSize]
	}

	return batchSize, vocabSize, batchLogits, nil
}

// GenerationConfig holds parameters for text generation.
type GenerationConfig struct {
	// MaxLength is the maximum number of tokens to generate.
	MaxLength int

	// MinLength is the minimum number of tokens to generate before allowing EOS.
	MinLength int

	// Temperature controls randomness in sampling (1.0 = no change, <1 = more deterministic).
	Temperature float64

	// TopP (nucleus sampling) - cumulative probability threshold.
	TopP float64

	// TopK - number of highest probability tokens to consider.
	TopK int

	// DoSample enables sampling mode (vs greedy).
	DoSample bool

	// NumBeams for beam search (1 = no beam search).
	NumBeams int

	// RepetitionPenalty to discourage repeating tokens.
	RepetitionPenalty float64

	// LengthPenalty for beam search scoring.
	LengthPenalty float64

	// EarlyStopping for beam search.
	EarlyStopping bool

	// EOSTokenID marks end of sequence.
	EOSTokenID int32

	// PadTokenID for padding.
	PadTokenID int32

	// ForcedBOSTokenID if set, forces this as first token.
	ForcedBOSTokenID int32

	// ForcedEOSTokenID if set, forces EOS at max_length.
	ForcedEOSTokenID int32

	// Rand is the random number generator for sampling. If nil, uses a default
	// seeded from the current time. Set this for deterministic/reproducible generation.
	Rand *rand.Rand
}

// DefaultGenerationConfig returns default generation parameters.
func DefaultGenerationConfig() *GenerationConfig {
	return &GenerationConfig{
		MaxLength:         64,
		MinLength:         0,
		Temperature:       1.0,
		TopP:              1.0,
		TopK:              0,
		DoSample:          false,
		NumBeams:          1,
		RepetitionPenalty: 1.0,
		LengthPenalty:     1.0,
		EarlyStopping:     false,
		EOSTokenID:        1,
		PadTokenID:        0,
		ForcedBOSTokenID:  -1,
		ForcedEOSTokenID:  -1,
	}
}

// GenerateGreedy generates output tokens using greedy decoding (argmax at each step).
// It's a simple and fast generation method that always picks the most likely next token.
func (b *Batch) GenerateGreedy(maxLength int) ([][]int32, error) {
	config := &GenerationConfig{
		MaxLength:  maxLength,
		EOSTokenID: b.model.Config.EOSTokenID,
		PadTokenID: b.model.Config.PadTokenID,
	}
	return b.Generate(config)
}

// GenerateSampling generates output tokens using sampling with temperature and top-p.
func (b *Batch) GenerateSampling(maxLength int, temperature, topP float64) ([][]int32, error) {
	config := &GenerationConfig{
		MaxLength:   maxLength,
		Temperature: temperature,
		TopP:        topP,
		DoSample:    true,
		EOSTokenID:  b.model.Config.EOSTokenID,
		PadTokenID:  b.model.Config.PadTokenID,
	}
	return b.Generate(config)
}

// Generate runs the generation loop with the given configuration.
func (b *Batch) Generate(config *GenerationConfig) ([][]int32, error) {
	// Run encoder if not done yet.
	if !b.encoderRun {
		if err := b.RunEncoder(); err != nil {
			return nil, errors.WithMessage(err, "failed to run encoder")
		}
	}

	// Run decoder init if not done yet.
	if !b.decoderInitd {
		if err := b.RunDecoderInit(); err != nil {
			return nil, errors.WithMessage(err, "failed to initialize decoder")
		}
	}

	// Initialize generation state.
	batchSize := b.batchSize
	generatedIDs := make([][]int32, batchSize)
	for i := range generatedIDs {
		generatedIDs[i] = make([]int32, 0, config.MaxLength)
	}

	// Track which sequences are finished.
	finished := make([]bool, batchSize)
	numFinished := 0

	// Create initial decoder input.
	startTokenID := b.model.Config.DecoderStartTokenID
	currentIDs := make([]int32, batchSize)
	for i := range currentIDs {
		currentIDs[i] = startTokenID
	}

	// Handle forced BOS token.
	if config.ForcedBOSTokenID >= 0 {
		for i := range currentIDs {
			generatedIDs[i] = append(generatedIDs[i], config.ForcedBOSTokenID)
		}
	}

	// Determine effective TopK for sampling mode.
	// Default to 50 if not specified, which covers most sampling scenarios.
	effectiveTopK := config.TopK
	if effectiveTopK <= 0 {
		effectiveTopK = 50
	}

	// Use provided Rand or create a default one for sampling mode.
	rng := config.Rand
	if rng == nil {
		rng = rand.New(rand.NewSource(rand.Int63()))
	}

	// Generation loop.
	for step := 0; step < config.MaxLength && numFinished < batchSize; step++ {
		// Create input tensor for this step.
		inputTensor := tensors.FromFlatDataAndDimensions(currentIDs, batchSize, 1)

		var nextTokens []int32
		var err error

		if config.DoSample {
			// Sampling mode: use on-device softmax + TopK, then sample on CPU.
			// This transfers only O(k) data instead of O(vocab_size).
			samplingOut, samplingErr := b.RunDecoderStepSampling(inputTensor, config.Temperature, effectiveTopK)
			if samplingErr != nil {
				inputTensor.FinalizeAll()
				return nil, errors.WithMessagef(samplingErr, "decoder step %d failed", step)
			}

			// Sample from TopK output on CPU.
			nextTokens, err = sampleFromTopKOutput(samplingOut, config, rng)
			samplingOut.Finalize()
		} else {
			// Greedy mode: use on-device argmax.
			// This transfers only O(batch_size) data instead of O(vocab_size).
			tokenIDsTensor, greedyErr := b.RunDecoderStepGreedy(inputTensor)
			if greedyErr != nil {
				inputTensor.FinalizeAll()
				return nil, errors.WithMessagef(greedyErr, "decoder step %d failed", step)
			}

			nextTokens, err = TensorToInt32Slice(tokenIDsTensor)
			tokenIDsTensor.FinalizeAll()
		}

		if err != nil {
			inputTensor.FinalizeAll()
			return nil, errors.WithMessage(err, "failed to get next tokens")
		}

		// Update generated sequences.
		for i := 0; i < batchSize; i++ {
			if finished[i] {
				continue
			}

			token := nextTokens[i]
			generatedIDs[i] = append(generatedIDs[i], token)
			currentIDs[i] = token

			// Check for EOS.
			if token == config.EOSTokenID && step >= config.MinLength {
				finished[i] = true
				numFinished++
			}
		}

		// Cleanup input tensor.
		inputTensor.FinalizeAll()
	}

	// Handle forced EOS at max length.
	if config.ForcedEOSTokenID >= 0 {
		for i := 0; i < batchSize; i++ {
			if !finished[i] {
				generatedIDs[i] = append(generatedIDs[i], config.ForcedEOSTokenID)
			}
		}
	}

	b.generatedIDs = generatedIDs
	b.generationDone = true

	return generatedIDs, nil
}

// argmaxFromLogits extracts the argmax token ID from logits for each batch item.
// logits shape: [batch_size, 1, vocab_size] or [batch_size, vocab_size]
func argmaxFromLogits(logits *tensors.Tensor) ([]int32, error) {
	batchSize, vocabSize, batchLogits, err := extractLogitsData(logits)
	if err != nil {
		return nil, err
	}

	// Find argmax for each batch item.
	tokens := make([]int32, batchSize)
	for batch := 0; batch < batchSize; batch++ {
		logitsSlice := batchLogits[batch]
		maxIdx := 0
		maxVal := logitsSlice[0]
		for v := 1; v < vocabSize; v++ {
			if logitsSlice[v] > maxVal {
				maxVal = logitsSlice[v]
				maxIdx = v
			}
		}
		tokens[batch] = int32(maxIdx)
	}

	return tokens, nil
}

// sampleFromLogits samples token IDs from logits with temperature and top-p/top-k.
func sampleFromLogits(logits *tensors.Tensor, config *GenerationConfig) ([]int32, error) {
	batchSize, _, batchLogits, err := extractLogitsData(logits)
	if err != nil {
		return nil, err
	}

	// Use provided Rand or create a default one.
	rng := config.Rand
	if rng == nil {
		rng = rand.New(rand.NewSource(rand.Int63()))
	}

	tokens := make([]int32, batchSize)
	for batch := 0; batch < batchSize; batch++ {
		logitsSlice := batchLogits[batch]

		// Apply temperature.
		if config.Temperature != 1.0 && config.Temperature > 0 {
			for i := range logitsSlice {
				logitsSlice[i] /= float32(config.Temperature)
			}
		}

		// Convert to probabilities with softmax.
		probs := softmax(logitsSlice)

		// Apply top-p (nucleus) or top-k sampling.
		var sampledToken int
		if config.TopP < 1.0 {
			sampledToken = sampleFromTopProbs(probs, config.TopP, 0, rng)
		} else if config.TopK > 0 {
			sampledToken = sampleFromTopProbs(probs, 1.0, config.TopK, rng)
		} else {
			sampledToken = sampleWithRng(probs, rng)
		}

		tokens[batch] = int32(sampledToken)
	}

	return tokens, nil
}

// softmax computes softmax over a slice of logits.
func softmax(logits []float32) []float32 {
	// Find max for numerical stability.
	maxVal := logits[0]
	for _, v := range logits[1:] {
		if v > maxVal {
			maxVal = v
		}
	}

	// Compute exp and sum.
	probs := make([]float32, len(logits))
	var sum float32
	for i, v := range logits {
		exp := float32(math.Exp(float64(v - maxVal)))
		probs[i] = exp
		sum += exp
	}

	// Normalize.
	for i := range probs {
		probs[i] /= sum
	}

	return probs
}

// sampleFromTopProbs implements both top-p (nucleus) and top-k sampling in a unified way.
// If topP < 1.0, uses nucleus sampling. If topK > 0, uses top-k sampling.
// Both can be combined (apply top-k first, then top-p within those).
func sampleFromTopProbs(probs []float32, topP float64, topK int, rng *rand.Rand) int {
	// Create indexed probabilities and sort by probability descending.
	indexed := make([]indexedProb, len(probs))
	for i, p := range probs {
		indexed[i] = indexedProb{i, p}
	}
	sort.Slice(indexed, func(i, j int) bool {
		return indexed[i].prob > indexed[j].prob
	})

	// Apply top-k filter if specified.
	candidates := indexed
	if topK > 0 && topK < len(indexed) {
		candidates = indexed[:topK]
	}

	// Apply top-p (nucleus) filter if specified.
	if topP < 1.0 {
		var cumSum float32
		cutoff := len(candidates)
		for i, ip := range candidates {
			cumSum += ip.prob
			if float64(cumSum) >= topP {
				cutoff = i + 1
				break
			}
		}
		candidates = candidates[:cutoff]
	}

	// Compute sum of remaining probabilities.
	var sum float32
	for _, ip := range candidates {
		sum += ip.prob
	}

	// Sample from candidates.
	r := rng.Float32() * sum
	var cumProb float32
	for _, ip := range candidates {
		cumProb += ip.prob
		if r <= cumProb {
			return ip.index
		}
	}

	// Fallback to most likely.
	return candidates[0].index
}

// sampleWithRng samples a token index from a probability distribution using the given RNG.
func sampleWithRng(probs []float32, rng *rand.Rand) int {
	r := rng.Float32()
	var cumProb float32
	for i, p := range probs {
		cumProb += p
		if r <= cumProb {
			return i
		}
	}
	return len(probs) - 1
}

// sampleFromTopKOutput samples tokens from TopK probabilities/indices returned by on-device processing.
// This is much faster than the old sampleFromLogits because k << vocab_size.
// The TopK values are already sorted in descending order by probability.
func sampleFromTopKOutput(out *SamplingOutput, config *GenerationConfig, rng *rand.Rand) ([]int32, error) {
	// Extract TopK data from tensors.
	probs, err := TensorToFloat32Slice(out.TopKProbs)
	if err != nil {
		return nil, errors.WithMessage(err, "failed to extract TopK probabilities")
	}
	indices, err := TensorToInt32Slice(out.TopKIndices)
	if err != nil {
		return nil, errors.WithMessage(err, "failed to extract TopK indices")
	}

	shape := out.TopKProbs.Shape()
	batchSize := shape.Dimensions[0]
	k := shape.Dimensions[1]

	tokens := make([]int32, batchSize)

	for batch := 0; batch < batchSize; batch++ {
		offset := batch * k
		batchProbs := probs[offset : offset+k]
		batchIndices := indices[offset : offset+k]

		// Apply TopP filtering within TopK (already sorted by probability descending).
		var candidates []indexedProb
		if config.TopP < 1.0 {
			var cumSum float32
			for i := 0; i < k; i++ {
				candidates = append(candidates, indexedProb{
					index: int(batchIndices[i]),
					prob:  batchProbs[i],
				})
				cumSum += batchProbs[i]
				if float64(cumSum) >= config.TopP {
					break
				}
			}
		} else {
			for i := 0; i < k; i++ {
				candidates = append(candidates, indexedProb{
					index: int(batchIndices[i]),
					prob:  batchProbs[i],
				})
			}
		}

		// Normalize and sample.
		var sum float32
		for _, c := range candidates {
			sum += c.prob
		}

		r := rng.Float32() * sum
		var cumProb float32
		sampled := false
		for _, c := range candidates {
			cumProb += c.prob
			if r <= cumProb {
				tokens[batch] = int32(c.index)
				sampled = true
				break
			}
		}

		// Fallback to most likely if not sampled.
		if !sampled && len(candidates) > 0 {
			tokens[batch] = int32(candidates[0].index)
		}
	}

	return tokens, nil
}

// sampleTopP implements nucleus (top-p) sampling.
// Deprecated: Use sampleFromTopProbs instead. Kept for backward compatibility.
func sampleTopP(probs []float32, topP float32) int {
	return sampleFromTopProbs(probs, float64(topP), 0, rand.New(rand.NewSource(rand.Int63())))
}

// sampleTopK samples from the top-k most likely tokens.
// Deprecated: Use sampleFromTopProbs instead. Kept for backward compatibility.
func sampleTopK(probs []float32, topK int) int {
	return sampleFromTopProbs(probs, 1.0, topK, rand.New(rand.NewSource(rand.Int63())))
}

// sampleFromProbs samples a token index from a probability distribution.
// Deprecated: Use sampleWithRng instead. Kept for backward compatibility.
func sampleFromProbs(probs []float32) int {
	return sampleWithRng(probs, rand.New(rand.NewSource(rand.Int63())))
}

// ApplyRepetitionPenalty modifies logits to penalize already-generated tokens.
func ApplyRepetitionPenalty(logits []float32, generatedIDs []int32, penalty float32) {
	if penalty == 1.0 {
		return
	}

	for _, tokenID := range generatedIDs {
		idx := int(tokenID)
		if idx < 0 || idx >= len(logits) {
			continue
		}

		if logits[idx] > 0 {
			logits[idx] /= penalty
		} else {
			logits[idx] *= penalty
		}
	}
}
