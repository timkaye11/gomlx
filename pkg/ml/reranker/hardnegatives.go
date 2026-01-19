// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package reranker

import (
	"sort"
)

// HardNegativeConfig configures hard negative mining.
type HardNegativeConfig struct {
	// NumHardNegatives is the number of hard negatives to sample per query.
	NumHardNegatives int

	// NumRandomNegatives is the number of random negatives to include.
	NumRandomNegatives int

	// MinScore is the minimum score for a document to be considered a hard negative.
	// Documents with scores below this are considered easy negatives.
	MinScore float32

	// MaxScore is the maximum score for a hard negative.
	// Documents with scores above this are considered positives.
	MaxScore float32
}

// DefaultHardNegativeConfig returns a default configuration.
func DefaultHardNegativeConfig() HardNegativeConfig {
	return HardNegativeConfig{
		NumHardNegatives:   3,
		NumRandomNegatives: 1,
		MinScore:           0.1,
		MaxScore:           0.5,
	}
}

// ScoredDocument represents a document with its relevance score.
type ScoredDocument struct {
	DocID int
	Score float32
}

// QueryDocuments represents a query with its candidate documents.
type QueryDocuments struct {
	QueryID   int
	Documents []ScoredDocument
}

// HardNegativeMiner mines hard negatives from scored document lists.
type HardNegativeMiner struct {
	config HardNegativeConfig
	rng    RandomSource
}

// RandomSource provides random number generation for sampling.
type RandomSource interface {
	// Intn returns a random int in [0, n).
	Intn(n int) int
	// Shuffle randomly shuffles a slice.
	Shuffle(n int, swap func(i, j int))
}

// NewHardNegativeMiner creates a new hard negative miner.
func NewHardNegativeMiner(config HardNegativeConfig, rng RandomSource) *HardNegativeMiner {
	return &HardNegativeMiner{
		config: config,
		rng:    rng,
	}
}

// MinedNegatives represents the result of hard negative mining for a query.
type MinedNegatives struct {
	QueryID       int
	HardNegatives []ScoredDocument
	EasyNegatives []ScoredDocument
}

// Mine performs hard negative mining for a single query.
// It selects hard negatives (high-scoring but non-relevant documents) and
// easy negatives (random non-relevant documents).
//
// Documents are categorized as:
//   - Positive: score >= MaxScore
//   - Hard negative: MinScore <= score < MaxScore
//   - Easy negative: score < MinScore
func (m *HardNegativeMiner) Mine(query QueryDocuments) MinedNegatives {
	var hardCandidates, easyCandidates []ScoredDocument

	// Categorize documents by score
	for _, doc := range query.Documents {
		if doc.Score >= m.config.MaxScore {
			// This is a positive, skip
			continue
		} else if doc.Score >= m.config.MinScore {
			// This is a hard negative candidate
			hardCandidates = append(hardCandidates, doc)
		} else {
			// This is an easy negative candidate
			easyCandidates = append(easyCandidates, doc)
		}
	}

	result := MinedNegatives{
		QueryID: query.QueryID,
	}

	// Sort hard candidates by score (descending) to get hardest negatives
	sort.Slice(hardCandidates, func(i, j int) bool {
		return hardCandidates[i].Score > hardCandidates[j].Score
	})

	// Select top hard negatives
	numHard := min(m.config.NumHardNegatives, len(hardCandidates))
	result.HardNegatives = hardCandidates[:numHard]

	// Randomly sample easy negatives
	if m.rng != nil && len(easyCandidates) > 0 {
		m.rng.Shuffle(len(easyCandidates), func(i, j int) {
			easyCandidates[i], easyCandidates[j] = easyCandidates[j], easyCandidates[i]
		})
	}
	numEasy := min(m.config.NumRandomNegatives, len(easyCandidates))
	result.EasyNegatives = easyCandidates[:numEasy]

	return result
}

// MineAll performs hard negative mining for multiple queries.
func (m *HardNegativeMiner) MineAll(queries []QueryDocuments) []MinedNegatives {
	results := make([]MinedNegatives, len(queries))
	for i, query := range queries {
		results[i] = m.Mine(query)
	}
	return results
}

// InBatchNegatives generates hard negatives from within a batch.
// This is an efficient alternative to explicit hard negative mining that uses
// other queries' positives as negatives.
//
// Given a batch of (query, positive_doc) pairs, for each query, the positive
// documents of other queries become in-batch negatives.
//
// Parameters:
//   - scores: A matrix of shape [batch_size, batch_size] where scores[i][j] is
//     the relevance score of document j to query i.
//
// Returns:
//   - hardNegIndices: For each query, the indices of the hardest in-batch negatives.
func InBatchNegatives(scores [][]float32, numNegatives int) [][]int {
	batchSize := len(scores)
	result := make([][]int, batchSize)

	for i := range batchSize {
		// Collect (index, score) pairs for all non-diagonal entries
		type scoredIdx struct {
			idx   int
			score float32
		}
		candidates := make([]scoredIdx, 0, batchSize-1)
		for j := range batchSize {
			if i != j { // Exclude the true positive
				candidates = append(candidates, scoredIdx{idx: j, score: scores[i][j]})
			}
		}

		// Sort by score descending (hardest negatives first)
		sort.Slice(candidates, func(a, b int) bool {
			return candidates[a].score > candidates[b].score
		})

		// Select top negatives
		numToSelect := min(numNegatives, len(candidates))
		result[i] = make([]int, numToSelect)
		for k := 0; k < numToSelect; k++ {
			result[i][k] = candidates[k].idx
		}
	}

	return result
}

// CrossEncoderScorer scores query-document pairs using a cross-encoder model.
// This interface allows for scoring candidates to identify hard negatives.
type CrossEncoderScorer interface {
	// ScoreBatch scores a batch of query-document pairs.
	// Returns relevance scores for each pair.
	ScoreBatch(queryTokens, docTokens [][]int32) ([]float32, error)
}

// MineWithScorer uses a cross-encoder to identify hard negatives.
// It scores all candidate documents and selects the highest-scoring non-relevant ones.
//
// Parameters:
//   - scorer: The cross-encoder model to use for scoring.
//   - queryTokens: Tokenized query.
//   - candidateTokens: Tokenized candidate documents.
//   - positiveIndices: Indices of known positive documents (to exclude).
//   - numHardNegatives: Number of hard negatives to select.
//
// Returns:
//   - The indices of selected hard negatives.
func MineWithScorer(scorer CrossEncoderScorer, queryTokens []int32, candidateTokens [][]int32, positiveIndices []int, numHardNegatives int) ([]int, error) {
	if len(candidateTokens) == 0 {
		return nil, nil
	}

	// Build positive set for fast lookup
	positiveSet := make(map[int]bool)
	for _, idx := range positiveIndices {
		positiveSet[idx] = true
	}

	// Score all candidates
	queryBatch := make([][]int32, len(candidateTokens))
	for i := range candidateTokens {
		queryBatch[i] = queryTokens
	}

	scores, err := scorer.ScoreBatch(queryBatch, candidateTokens)
	if err != nil {
		return nil, err
	}

	// Collect non-positive candidates with their scores
	type scoredCandidate struct {
		idx   int
		score float32
	}
	var negatives []scoredCandidate
	for i, score := range scores {
		if !positiveSet[i] {
			negatives = append(negatives, scoredCandidate{idx: i, score: score})
		}
	}

	// Sort by score descending
	sort.Slice(negatives, func(i, j int) bool {
		return negatives[i].score > negatives[j].score
	})

	// Select top hard negatives
	numToSelect := min(numHardNegatives, len(negatives))
	result := make([]int, numToSelect)
	for i := 0; i < numToSelect; i++ {
		result[i] = negatives[i].idx
	}

	return result, nil
}
