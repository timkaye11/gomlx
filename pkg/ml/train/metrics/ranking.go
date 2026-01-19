// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package metrics

import (
	"fmt"
	"math"

	. "github.com/gomlx/gomlx/internal/exceptions"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
)

// Metric types for ranking metrics.
const (
	// NDCGMetricType is the type for NDCG metrics.
	NDCGMetricType = "ndcg"

	// MRRMetricType is the type for MRR metrics.
	MRRMetricType = "mrr"

	// MAPMetricType is the type for MAP metrics.
	MAPMetricType = "map"

	// PrecisionMetricType is the type for precision metrics.
	PrecisionMetricType = "precision"

	// RecallMetricType is the type for recall metrics.
	RecallMetricType = "recall"
)

// NDCGGraph computes the Normalized Discounted Cumulative Gain (NDCG) metric.
//
// NDCG measures ranking quality by comparing the model's ranking to the ideal ranking.
// It accounts for both relevance and position, giving more credit to highly relevant
// items ranked higher in the list.
//
// Formula:
//
//	DCG@K = sum_{i=1}^{K} (2^rel_i - 1) / log2(i + 1)
//	IDCG@K = DCG@K for the ideal ranking
//	NDCG@K = DCG@K / IDCG@K
//
// Parameters:
//   - labels[0]: Relevance scores (higher = more relevant), shape [batch, list_size]
//   - predictions[0]: Model scores, shape [batch, list_size]
//   - k: Number of top items to consider (0 = all items)
//
// Returns a scalar NDCG value averaged over the batch.
func NDCGGraph(k int) BaseMetricGraph {
	return func(_ *context.Context, labels, predictions []*Node) *Node {
		return ndcgCompute(labels, predictions, k)
	}
}

func ndcgCompute(labels, predictions []*Node, k int) *Node {
	labels0 := labels[0]
	scores := predictions[0]
	dtype := scores.DType()
	g := scores.Graph()

	if labels0.Rank() != 2 || scores.Rank() != 2 {
		Panicf("NDCG expects labels and predictions of rank 2 [batch, list_size], got %s and %s",
			labels0.Shape(), scores.Shape())
	}

	listSize := scores.Shape().Dimensions[1]
	if k <= 0 || k > listSize {
		k = listSize
	}

	// Convert labels to same dtype
	relevance := ConvertDType(labels0, dtype)

	// Sort by model scores (descending) to get predicted ranking
	// We negate and sort ascending to get descending order
	negScores := Neg(scores)
	sortedNegScores := Sort(negScores, -1, true) // ascending
	sortedScores := Neg(sortedNegScores)

	// Get indices that would sort the scores (descending)
	// We'll use a trick: compare sorted scores with original to find positions
	// For NDCG, we need to gather relevance by score ranking

	// Create pairwise comparison to find where each score ends up
	// This is approximate but works for distinct scores
	scoresExpanded := InsertAxes(scores, -1)           // [batch, list_size, 1]
	sortedExpanded := InsertAxes(sortedScores, -2)     // [batch, 1, list_size]
	matches := Equal(scoresExpanded, sortedExpanded)   // [batch, list_size, list_size]

	// Use soft attention to gather relevance values by rank
	// matches[i,j] = 1 if scores[i] == sortedScores[j]
	matchesFloat := ConvertDType(matches, dtype)
	// Normalize to handle potential ties
	matchSum := ReduceAndKeep(matchesFloat, ReduceSum, -2)
	matchSum = MaxScalar(matchSum, 1.0)
	normalizedMatches := Div(matchesFloat, matchSum)

	// Transpose and multiply to get relevance in sorted order
	// rankedRel[j] = sum_i relevance[i] * matches[i,j]
	relExpanded := InsertAxes(relevance, -1)          // [batch, list_size, 1]
	rankedRel := ReduceSum(Mul(relExpanded, normalizedMatches), -2) // [batch, list_size]

	// Only consider top K
	if k < listSize {
		// Create mask for top K positions
		positions := Iota(g, shapes.Make(dtype, scores.Shape().Dimensions...), -1)
		topKMask := LessThan(positions, Scalar(g, dtype, float64(k)))
		rankedRel = Where(topKMask, rankedRel, ZerosLike(rankedRel))
	}

	// Compute DCG
	// gains = 2^rel - 1
	twoConst := Scalar(g, dtype, 2.0)
	gains := Sub(Pow(twoConst, rankedRel), ScalarOne(g, dtype))

	// discounts = 1 / log2(rank + 2) where rank is 0-indexed
	// So for rank 0: 1/log2(2) = 1, rank 1: 1/log2(3), etc.
	ranks := Iota(g, shapes.Make(dtype, scores.Shape().Dimensions...), -1)
	ranks = ConvertDType(ranks, dtype)
	log2 := Scalar(g, dtype, math.Log(2.0))
	discounts := Div(ScalarOne(g, dtype), Div(Log(AddScalar(ranks, 2.0)), log2))

	dcg := ReduceSum(Mul(gains, discounts), -1)

	// Compute IDCG (ideal DCG) - sort relevance descending
	sortedRel := Neg(Sort(Neg(relevance), -1, true))
	if k < listSize {
		positions := Iota(g, shapes.Make(dtype, scores.Shape().Dimensions...), -1)
		topKMask := LessThan(positions, Scalar(g, dtype, float64(k)))
		sortedRel = Where(topKMask, sortedRel, ZerosLike(sortedRel))
	}

	idealGains := Sub(Pow(twoConst, sortedRel), ScalarOne(g, dtype))
	idcg := ReduceSum(Mul(idealGains, discounts), -1)

	// NDCG = DCG / IDCG
	eps := Scalar(g, dtype, 1e-10)
	ndcg := Div(dcg, Max(idcg, eps))

	// Average over batch
	return ReduceAllMean(ndcg)
}

// NewMeanNDCG creates an NDCG metric that reports the mean NDCG@K across batches.
// If k <= 0, all items in the list are considered.
func NewMeanNDCG(name, shortName string, k int) *MeanMetric {
	return NewMeanMetric(name, shortName, NDCGMetricType, NDCGGraph(k), rankingPPrint)
}

// MRRGraph computes the Mean Reciprocal Rank (MRR) metric.
//
// MRR is the average of the reciprocal ranks of the first relevant item for each query.
// For binary relevance (0/1), it finds the rank of the first relevant item.
//
// Formula:
//
//	RR = 1 / rank_of_first_relevant_item
//	MRR = mean(RR) over all queries
//
// Parameters:
//   - labels[0]: Binary relevance labels (1 = relevant), shape [batch, list_size]
//   - predictions[0]: Model scores, shape [batch, list_size]
//
// Returns a scalar MRR value averaged over the batch.
func MRRGraph(_ *context.Context, labels, predictions []*Node) *Node {
	labels0 := labels[0]
	scores := predictions[0]
	dtype := scores.DType()
	g := scores.Graph()

	if labels0.Rank() != 2 || scores.Rank() != 2 {
		Panicf("MRR expects labels and predictions of rank 2 [batch, list_size], got %s and %s",
			labels0.Shape(), scores.Shape())
	}

	// Convert labels to same dtype
	relevance := ConvertDType(labels0, dtype)

	// Sort by model scores (descending) to get predicted ranking
	negScores := Neg(scores)
	sortedNegScores := Sort(negScores, -1, true)
	sortedScores := Neg(sortedNegScores)

	// Get relevance in sorted order using soft attention
	scoresExpanded := InsertAxes(scores, -1)
	sortedExpanded := InsertAxes(sortedScores, -2)
	matches := Equal(scoresExpanded, sortedExpanded)
	matchesFloat := ConvertDType(matches, dtype)
	matchSum := ReduceAndKeep(matchesFloat, ReduceSum, -2)
	matchSum = MaxScalar(matchSum, 1.0)
	normalizedMatches := Div(matchesFloat, matchSum)

	relExpanded := InsertAxes(relevance, -1)
	rankedRel := ReduceSum(Mul(relExpanded, normalizedMatches), -2)

	// Find the first relevant item in the ranked list
	// Use cumsum to detect the first occurrence
	isRelevant := GreaterThan(rankedRel, ScalarZero(g, dtype))
	isRelevantFloat := ConvertDType(isRelevant, dtype)

	// Cumulative sum to count relevant items seen so far
	cumRelevant := CumSum(isRelevantFloat, -1)

	// First relevant item is where cumRelevant == 1 AND isRelevant == 1
	isFirstRelevant := And(Equal(cumRelevant, ScalarOne(g, dtype)), isRelevant)
	isFirstRelevantFloat := ConvertDType(isFirstRelevant, dtype)

	// Ranks are 1-indexed for MRR
	ranks := Iota(g, shapes.Make(dtype, scores.Shape().Dimensions...), -1)
	ranks = AddScalar(ConvertDType(ranks, dtype), 1.0)

	// Reciprocal rank: sum(isFirstRelevant * (1/rank))
	reciprocalRanks := Mul(isFirstRelevantFloat, Reciprocal(ranks))
	rr := ReduceSum(reciprocalRanks, -1)

	// Average over batch
	return ReduceAllMean(rr)
}

// NewMeanMRR creates an MRR metric that reports the mean MRR across batches.
func NewMeanMRR(name, shortName string) *MeanMetric {
	return NewMeanMetric(name, shortName, MRRMetricType, MRRGraph, rankingPPrint)
}

// MAPGraph computes the Mean Average Precision (MAP) metric.
//
// MAP is the mean of the Average Precision (AP) for each query.
// AP is the average of precisions computed at each relevant item in the ranked list.
//
// Formula:
//
//	Precision@k = (relevant items in top k) / k
//	AP = sum_{k: item_k is relevant} Precision@k / total_relevant
//	MAP = mean(AP) over all queries
//
// Parameters:
//   - labels[0]: Binary relevance labels (1 = relevant), shape [batch, list_size]
//   - predictions[0]: Model scores, shape [batch, list_size]
//
// Returns a scalar MAP value averaged over the batch.
func MAPGraph(_ *context.Context, labels, predictions []*Node) *Node {
	labels0 := labels[0]
	scores := predictions[0]
	dtype := scores.DType()
	g := scores.Graph()

	if labels0.Rank() != 2 || scores.Rank() != 2 {
		Panicf("MAP expects labels and predictions of rank 2 [batch, list_size], got %s and %s",
			labels0.Shape(), scores.Shape())
	}

	// Convert labels to same dtype
	relevance := ConvertDType(labels0, dtype)

	// Sort by model scores (descending) to get predicted ranking
	negScores := Neg(scores)
	sortedNegScores := Sort(negScores, -1, true)
	sortedScores := Neg(sortedNegScores)

	// Get relevance in sorted order using soft attention
	scoresExpanded := InsertAxes(scores, -1)
	sortedExpanded := InsertAxes(sortedScores, -2)
	matches := Equal(scoresExpanded, sortedExpanded)
	matchesFloat := ConvertDType(matches, dtype)
	matchSum := ReduceAndKeep(matchesFloat, ReduceSum, -2)
	matchSum = MaxScalar(matchSum, 1.0)
	normalizedMatches := Div(matchesFloat, matchSum)

	relExpanded := InsertAxes(relevance, -1)
	rankedRel := ReduceSum(Mul(relExpanded, normalizedMatches), -2)

	// Compute precision at each position
	isRelevant := GreaterThan(rankedRel, ScalarZero(g, dtype))
	isRelevantFloat := ConvertDType(isRelevant, dtype)

	// Cumulative sum of relevant items
	cumRelevant := CumSum(isRelevantFloat, -1)

	// Ranks (1-indexed)
	ranks := Iota(g, shapes.Make(dtype, scores.Shape().Dimensions...), -1)
	ranks = AddScalar(ConvertDType(ranks, dtype), 1.0)

	// Precision at each position: cumRelevant / rank
	precisionAtK := Div(cumRelevant, ranks)

	// AP: sum of precision at relevant positions / total relevant
	precisionAtRelevant := Mul(precisionAtK, isRelevantFloat)
	sumPrecision := ReduceSum(precisionAtRelevant, -1)

	totalRelevant := ReduceSum(relevance, -1)
	totalRelevant = MaxScalar(totalRelevant, 1.0) // Avoid division by zero

	ap := Div(sumPrecision, totalRelevant)

	// Average over batch
	return ReduceAllMean(ap)
}

// NewMeanMAP creates a MAP metric that reports the mean MAP across batches.
func NewMeanMAP(name, shortName string) *MeanMetric {
	return NewMeanMetric(name, shortName, MAPMetricType, MAPGraph, rankingPPrint)
}

// PrecisionAtKGraph computes the Precision@K metric.
//
// Precision@K is the proportion of relevant items among the top K items.
//
// Formula:
//
//	Precision@K = (number of relevant items in top K) / K
//
// Parameters:
//   - labels[0]: Binary relevance labels (1 = relevant), shape [batch, list_size]
//   - predictions[0]: Model scores, shape [batch, list_size]
//   - k: Number of top items to consider
//
// Returns a scalar Precision@K value averaged over the batch.
func PrecisionAtKGraph(k int) BaseMetricGraph {
	return func(_ *context.Context, labels, predictions []*Node) *Node {
		return precisionAtKCompute(labels, predictions, k)
	}
}

func precisionAtKCompute(labels, predictions []*Node, k int) *Node {
	labels0 := labels[0]
	scores := predictions[0]
	dtype := scores.DType()
	g := scores.Graph()

	if labels0.Rank() != 2 || scores.Rank() != 2 {
		Panicf("Precision@K expects labels and predictions of rank 2 [batch, list_size], got %s and %s",
			labels0.Shape(), scores.Shape())
	}

	listSize := scores.Shape().Dimensions[1]
	if k <= 0 {
		k = listSize
	}
	if k > listSize {
		k = listSize
	}

	// Convert labels to same dtype
	relevance := ConvertDType(labels0, dtype)

	// Sort by model scores (descending) to get predicted ranking
	negScores := Neg(scores)
	sortedNegScores := Sort(negScores, -1, true)
	sortedScores := Neg(sortedNegScores)

	// Get relevance in sorted order using soft attention
	scoresExpanded := InsertAxes(scores, -1)
	sortedExpanded := InsertAxes(sortedScores, -2)
	matches := Equal(scoresExpanded, sortedExpanded)
	matchesFloat := ConvertDType(matches, dtype)
	matchSum := ReduceAndKeep(matchesFloat, ReduceSum, -2)
	matchSum = MaxScalar(matchSum, 1.0)
	normalizedMatches := Div(matchesFloat, matchSum)

	relExpanded := InsertAxes(relevance, -1)
	rankedRel := ReduceSum(Mul(relExpanded, normalizedMatches), -2)

	// Mask to only consider top K positions
	positions := Iota(g, shapes.Make(dtype, scores.Shape().Dimensions...), -1)
	topKMask := LessThan(positions, Scalar(g, dtype, float64(k)))

	// Count relevant items in top K
	isRelevant := GreaterThan(rankedRel, ScalarZero(g, dtype))
	isRelevantFloat := ConvertDType(isRelevant, dtype)
	topKRelevant := Where(topKMask, isRelevantFloat, ZerosLike(isRelevantFloat))
	relevantInTopK := ReduceSum(topKRelevant, -1)

	// Precision = relevant_in_top_k / k
	precision := DivScalar(relevantInTopK, float64(k))

	// Average over batch
	return ReduceAllMean(precision)
}

// NewMeanPrecisionAtK creates a Precision@K metric that reports the mean Precision@K across batches.
func NewMeanPrecisionAtK(name, shortName string, k int) *MeanMetric {
	return NewMeanMetric(name, shortName, PrecisionMetricType, PrecisionAtKGraph(k), rankingPPrint)
}

// RecallAtKGraph computes the Recall@K metric.
//
// Recall@K is the proportion of all relevant items that are in the top K items.
//
// Formula:
//
//	Recall@K = (number of relevant items in top K) / (total number of relevant items)
//
// Parameters:
//   - labels[0]: Binary relevance labels (1 = relevant), shape [batch, list_size]
//   - predictions[0]: Model scores, shape [batch, list_size]
//   - k: Number of top items to consider
//
// Returns a scalar Recall@K value averaged over the batch.
func RecallAtKGraph(k int) BaseMetricGraph {
	return func(_ *context.Context, labels, predictions []*Node) *Node {
		return recallAtKCompute(labels, predictions, k)
	}
}

func recallAtKCompute(labels, predictions []*Node, k int) *Node {
	labels0 := labels[0]
	scores := predictions[0]
	dtype := scores.DType()
	g := scores.Graph()

	if labels0.Rank() != 2 || scores.Rank() != 2 {
		Panicf("Recall@K expects labels and predictions of rank 2 [batch, list_size], got %s and %s",
			labels0.Shape(), scores.Shape())
	}

	listSize := scores.Shape().Dimensions[1]
	if k <= 0 {
		k = listSize
	}
	if k > listSize {
		k = listSize
	}

	// Convert labels to same dtype
	relevance := ConvertDType(labels0, dtype)

	// Sort by model scores (descending) to get predicted ranking
	negScores := Neg(scores)
	sortedNegScores := Sort(negScores, -1, true)
	sortedScores := Neg(sortedNegScores)

	// Get relevance in sorted order using soft attention
	scoresExpanded := InsertAxes(scores, -1)
	sortedExpanded := InsertAxes(sortedScores, -2)
	matches := Equal(scoresExpanded, sortedExpanded)
	matchesFloat := ConvertDType(matches, dtype)
	matchSum := ReduceAndKeep(matchesFloat, ReduceSum, -2)
	matchSum = MaxScalar(matchSum, 1.0)
	normalizedMatches := Div(matchesFloat, matchSum)

	relExpanded := InsertAxes(relevance, -1)
	rankedRel := ReduceSum(Mul(relExpanded, normalizedMatches), -2)

	// Mask to only consider top K positions
	positions := Iota(g, shapes.Make(dtype, scores.Shape().Dimensions...), -1)
	topKMask := LessThan(positions, Scalar(g, dtype, float64(k)))

	// Count relevant items in top K
	isRelevant := GreaterThan(rankedRel, ScalarZero(g, dtype))
	isRelevantFloat := ConvertDType(isRelevant, dtype)
	topKRelevant := Where(topKMask, isRelevantFloat, ZerosLike(isRelevantFloat))
	relevantInTopK := ReduceSum(topKRelevant, -1)

	// Total relevant items
	totalRelevant := ReduceSum(relevance, -1)
	totalRelevant = MaxScalar(totalRelevant, 1.0) // Avoid division by zero

	// Recall = relevant_in_top_k / total_relevant
	recall := Div(relevantInTopK, totalRelevant)

	// Average over batch
	return ReduceAllMean(recall)
}

// NewMeanRecallAtK creates a Recall@K metric that reports the mean Recall@K across batches.
func NewMeanRecallAtK(name, shortName string, k int) *MeanMetric {
	return NewMeanMetric(name, shortName, RecallMetricType, RecallAtKGraph(k), rankingPPrint)
}

// rankingPPrint formats ranking metrics for display.
func rankingPPrint(value *tensors.Tensor) string {
	v := shapes.ConvertTo[float64](value.Value())
	return fmt.Sprintf("%.4f", v)
}
