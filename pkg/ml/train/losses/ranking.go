// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package losses

import (
	"math"

	. "github.com/gomlx/gomlx/internal/exceptions"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
)

// Ranking loss hyperparameters
var (
	// ParamMarginRankingMargin is the margin for MarginRankingLoss.
	// Default is 1.0.
	ParamMarginRankingMargin = "margin_ranking_margin"

	// ParamListMLETemperature is the temperature for ListMLE softmax.
	// Default is 1.0.
	ParamListMLETemperature = "listmle_temperature"

	// ParamListNetTemperature is the temperature for ListNet softmax.
	// Default is 1.0.
	ParamListNetTemperature = "listnet_temperature"

	// ParamPairwiseLogisticSigma is the sigma (scaling) parameter for PairwiseLogisticLoss.
	// Default is 1.0.
	ParamPairwiseLogisticSigma = "pairwise_logistic_sigma"
)

// PairwiseLogisticLoss computes the pairwise logistic loss for ranking.
//
// This loss is commonly used for training cross-encoder re-rankers. For each pair
// of (positive, negative) items, it computes:
//
//	loss = log(1 + exp(sigma * (score_negative - score_positive)))
//
// This encourages the model to score positive items higher than negative items.
//
// Parameters:
//   - labels[0]: Binary relevance labels, shape [batch, list_size] where 1 = relevant, 0 = not relevant
//   - predictions[0]: Model scores, shape [batch, list_size]
//   - sigma: Scaling factor (default 1.0), controls the steepness of the loss
//
// The loss is computed over all valid positive-negative pairs within each list.
//
// References:
//   - "From RankNet to LambdaRank to LambdaMART: An Overview" (Burges, 2010)
func PairwiseLogisticLoss(labels, predictions []*Node, sigma float64) *Node {
	labels0 := labels[0]
	scores := predictions[0]
	dtype := scores.DType()
	g := scores.Graph()

	if labels0.Rank() != 2 || scores.Rank() != 2 {
		Panicf("PairwiseLogisticLoss expects labels and predictions of rank 2 [batch, list_size], got %s and %s",
			labels0.Shape(), scores.Shape())
	}

	// Convert labels to same dtype as scores
	relevance := ConvertDType(labels0, dtype)

	// Create pairwise difference matrix: scores[i] - scores[j]
	// Shape: [batch, list_size, list_size]
	scoresI := InsertAxes(scores, -1)         // [batch, list_size, 1]
	scoresJ := InsertAxes(scores, -2)         // [batch, 1, list_size]
	pairwiseDiff := Sub(scoresI, scoresJ)     // [batch, list_size, list_size]

	// Create valid pair mask: relevance[i] > relevance[j] (positive vs negative pairs)
	relI := InsertAxes(relevance, -1)         // [batch, list_size, 1]
	relJ := InsertAxes(relevance, -2)         // [batch, 1, list_size]
	validPairs := GreaterThan(relI, relJ)     // [batch, list_size, list_size]

	// Compute pairwise logistic loss: log(1 + exp(-sigma * (score_pos - score_neg)))
	// For valid pairs, we want score_pos > score_neg, so diff = score_pos - score_neg should be positive
	// loss = log(1 + exp(-sigma * diff))
	sigmaNode := Scalar(g, dtype, sigma)
	negScaledDiff := Mul(Neg(pairwiseDiff), sigmaNode)
	pairwiseLoss := Log1P(Exp(negScaledDiff))

	// Zero out invalid pairs
	zero := ScalarZero(g, dtype)
	pairwiseLoss = Where(validPairs, pairwiseLoss, BroadcastToShape(zero, pairwiseLoss.Shape()))

	// Count valid pairs per batch
	validCount := ReduceSum(ConvertDType(validPairs, dtype), -1, -2)
	validCount = MaxScalar(validCount, 1.0) // Avoid division by zero

	// Sum loss over pairs and normalize
	loss := ReduceSum(pairwiseLoss, -1, -2)
	loss = Div(loss, validCount)

	// Factor in weights and mask
	weightsShape := shapes.Make(dtype, labels0.Shape().Dimensions[:1]...)
	weights, mask := CheckExtraLabelsForWeightsAndMask(weightsShape, labels[1:])
	if weights != nil {
		loss = Mul(loss, weights)
	}
	if mask != nil {
		loss = Where(mask, loss, ZerosLike(loss))
		if !loss.IsScalar() {
			loss = MaskedReduceAllMean(loss, mask)
		}
	} else if !loss.IsScalar() {
		loss = ReduceAllMean(loss)
	}

	return loss
}

// MakePairwiseLogisticLossFromContext creates a PairwiseLogisticLoss from context parameters.
func MakePairwiseLogisticLossFromContext(ctx *context.Context) LossFn {
	sigma := context.GetParamOr(ctx, ParamPairwiseLogisticSigma, 1.0)
	return func(labels, predictions []*Node) *Node {
		return PairwiseLogisticLoss(labels, predictions, sigma)
	}
}

// MarginRankingLoss computes a margin-based ranking loss.
//
// For each pair of (positive, negative) items:
//
//	loss = max(0, margin - (score_positive - score_negative))
//
// This loss penalizes when the positive score isn't at least `margin` higher than the negative score.
//
// Parameters:
//   - labels[0]: Binary relevance labels, shape [batch, list_size] where 1 = relevant, 0 = not relevant
//   - predictions[0]: Model scores, shape [batch, list_size]
//   - margin: Target margin between positive and negative scores (default 1.0)
//
// References:
//   - Commonly used in learning-to-rank and metric learning
func MarginRankingLoss(labels, predictions []*Node, margin float64) *Node {
	labels0 := labels[0]
	scores := predictions[0]
	dtype := scores.DType()
	g := scores.Graph()

	if labels0.Rank() != 2 || scores.Rank() != 2 {
		Panicf("MarginRankingLoss expects labels and predictions of rank 2 [batch, list_size], got %s and %s",
			labels0.Shape(), scores.Shape())
	}

	// Convert labels to same dtype as scores
	relevance := ConvertDType(labels0, dtype)

	// Create pairwise difference matrix
	scoresI := InsertAxes(scores, -1)
	scoresJ := InsertAxes(scores, -2)
	pairwiseDiff := Sub(scoresI, scoresJ)

	// Create valid pair mask: relevance[i] > relevance[j]
	relI := InsertAxes(relevance, -1)
	relJ := InsertAxes(relevance, -2)
	validPairs := GreaterThan(relI, relJ)

	// Compute margin loss: max(0, margin - diff)
	marginNode := Scalar(g, dtype, margin)
	pairwiseLoss := MaxScalar(Sub(marginNode, pairwiseDiff), 0.0)

	// Zero out invalid pairs
	zero := ScalarZero(g, dtype)
	pairwiseLoss = Where(validPairs, pairwiseLoss, BroadcastToShape(zero, pairwiseLoss.Shape()))

	// Count valid pairs and normalize
	validCount := ReduceSum(ConvertDType(validPairs, dtype), -1, -2)
	validCount = MaxScalar(validCount, 1.0)

	loss := ReduceSum(pairwiseLoss, -1, -2)
	loss = Div(loss, validCount)

	// Factor in weights and mask
	weightsShape := shapes.Make(dtype, labels0.Shape().Dimensions[:1]...)
	weights, mask := CheckExtraLabelsForWeightsAndMask(weightsShape, labels[1:])
	if weights != nil {
		loss = Mul(loss, weights)
	}
	if mask != nil {
		loss = Where(mask, loss, ZerosLike(loss))
		if !loss.IsScalar() {
			loss = MaskedReduceAllMean(loss, mask)
		}
	} else if !loss.IsScalar() {
		loss = ReduceAllMean(loss)
	}

	return loss
}

// MakeMarginRankingLossFromContext creates a MarginRankingLoss from context parameters.
func MakeMarginRankingLossFromContext(ctx *context.Context) LossFn {
	margin := context.GetParamOr(ctx, ParamMarginRankingMargin, 1.0)
	return func(labels, predictions []*Node) *Node {
		return MarginRankingLoss(labels, predictions, margin)
	}
}

// ListMLELoss computes the ListMLE (Listwise Maximum Likelihood Estimation) loss.
//
// ListMLE treats ranking as maximizing the likelihood of the ground-truth permutation.
// This implementation uses a softmax-based approximation that is differentiable and
// doesn't require explicit sorting operations.
//
// The loss uses relevance-weighted softmax probabilities:
//
//	loss = -sum_i rel_i * log_softmax(scores)_i
//
// This is equivalent to cross-entropy where the target distribution is normalized relevance.
//
// Parameters:
//   - labels[0]: Relevance scores (higher = more relevant), shape [batch, list_size]
//   - predictions[0]: Model scores, shape [batch, list_size]
//   - temperature: Softmax temperature (default 1.0)
//
// References:
//   - "Listwise Approach to Learning to Rank: Theory and Algorithm" (Xia et al., 2008)
func ListMLELoss(labels, predictions []*Node, temperature float64) *Node {
	labels0 := labels[0]
	scores := predictions[0]
	dtype := scores.DType()
	g := scores.Graph()

	if labels0.Rank() != 2 || scores.Rank() != 2 {
		Panicf("ListMLELoss expects labels and predictions of rank 2 [batch, list_size], got %s and %s",
			labels0.Shape(), scores.Shape())
	}

	// Convert labels to same dtype
	relevance := ConvertDType(labels0, dtype)

	// Scale scores by temperature
	scaledScores := scores
	if temperature != 1.0 {
		scaledScores = DivScalar(scores, temperature)
	}

	// Compute log_softmax of scores for numerical stability
	eps := epsilonForDType(g, dtype)
	maxScore := ReduceAndKeep(scaledScores, ReduceMax, -1)
	shiftedScores := Sub(scaledScores, maxScore)
	logSumExp := Log(Max(ReduceAndKeep(Exp(shiftedScores), ReduceSum, -1), eps))
	logSoftmax := Sub(shiftedScores, logSumExp)

	// Normalize relevance to create target distribution
	relSum := ReduceAndKeep(relevance, ReduceSum, -1)
	relSum = MaxScalar(relSum, 1e-10) // Avoid division by zero
	normalizedRel := Div(relevance, relSum)

	// Cross-entropy loss: -sum(normalized_rel * log_softmax(scores))
	loss := Neg(ReduceSum(Mul(normalizedRel, logSoftmax), -1))

	// Factor in weights and mask
	weightsShape := shapes.Make(dtype, labels0.Shape().Dimensions[:1]...)
	weights, mask := CheckExtraLabelsForWeightsAndMask(weightsShape, labels[1:])
	if weights != nil {
		loss = Mul(loss, weights)
	}
	if mask != nil {
		loss = Where(mask, loss, ZerosLike(loss))
		if !loss.IsScalar() {
			loss = MaskedReduceAllMean(loss, mask)
		}
	} else if !loss.IsScalar() {
		loss = ReduceAllMean(loss)
	}

	return loss
}

// MakeListMLELossFromContext creates a ListMLELoss from context parameters.
func MakeListMLELossFromContext(ctx *context.Context) LossFn {
	temperature := context.GetParamOr(ctx, ParamListMLETemperature, 1.0)
	return func(labels, predictions []*Node) *Node {
		return ListMLELoss(labels, predictions, temperature)
	}
}

// ListNetLoss computes the ListNet cross-entropy loss between score distributions.
//
// ListNet treats ranking as matching the probability distribution of scores to
// the distribution of relevance labels. It uses cross-entropy between:
//   - P_y: softmax of relevance labels (ground truth distribution)
//   - P_s: softmax of model scores (predicted distribution)
//
// The loss is:
//
//	loss = -sum_i P_y(i) * log(P_s(i))
//
// Parameters:
//   - labels[0]: Relevance scores (higher = more relevant), shape [batch, list_size]
//   - predictions[0]: Model scores, shape [batch, list_size]
//   - temperature: Softmax temperature (default 1.0)
//
// References:
//   - "Learning to Rank: From Pairwise Approach to Listwise Approach" (Cao et al., 2007)
func ListNetLoss(labels, predictions []*Node, temperature float64) *Node {
	labels0 := labels[0]
	scores := predictions[0]
	dtype := scores.DType()
	g := scores.Graph()

	if labels0.Rank() != 2 || scores.Rank() != 2 {
		Panicf("ListNetLoss expects labels and predictions of rank 2 [batch, list_size], got %s and %s",
			labels0.Shape(), scores.Shape())
	}

	// Convert labels to same dtype
	relevance := ConvertDType(labels0, dtype)

	// Scale by temperature
	if temperature != 1.0 {
		relevance = DivScalar(relevance, temperature)
		scores = DivScalar(scores, temperature)
	}

	// Compute softmax distributions
	// P_y = softmax(relevance)
	// P_s = softmax(scores)
	pyLogits := relevance
	psLogits := scores

	// For numerical stability, use log_softmax
	eps := epsilonForDType(g, dtype)

	// Compute softmax of relevance (target distribution)
	pyMax := ReduceAndKeep(pyLogits, ReduceMax, -1)
	pyShifted := Sub(pyLogits, pyMax)
	pyExp := Exp(pyShifted)
	pySum := ReduceAndKeep(pyExp, ReduceSum, -1)
	py := Div(pyExp, pySum)

	// Compute log_softmax of scores (predicted log-probabilities)
	psMax := ReduceAndKeep(psLogits, ReduceMax, -1)
	psShifted := Sub(psLogits, psMax)
	psLogSum := Log(Max(ReduceAndKeep(Exp(psShifted), ReduceSum, -1), eps))
	logPs := Sub(psShifted, psLogSum)

	// Cross-entropy: -sum(P_y * log(P_s))
	loss := Neg(ReduceSum(Mul(py, logPs), -1))

	// Factor in weights and mask
	weightsShape := shapes.Make(dtype, labels0.Shape().Dimensions[:1]...)
	weights, mask := CheckExtraLabelsForWeightsAndMask(weightsShape, labels[1:])
	if weights != nil {
		loss = Mul(loss, weights)
	}
	if mask != nil {
		loss = Where(mask, loss, ZerosLike(loss))
		if !loss.IsScalar() {
			loss = MaskedReduceAllMean(loss, mask)
		}
	} else if !loss.IsScalar() {
		loss = ReduceAllMean(loss)
	}

	return loss
}

// MakeListNetLossFromContext creates a ListNetLoss from context parameters.
func MakeListNetLossFromContext(ctx *context.Context) LossFn {
	temperature := context.GetParamOr(ctx, ParamListNetTemperature, 1.0)
	return func(labels, predictions []*Node) *Node {
		return ListNetLoss(labels, predictions, temperature)
	}
}

// ApproxNDCGLoss computes an approximate NDCG loss using a differentiable surrogate.
//
// This loss uses a sigmoid-based soft ranking function to approximate NDCG,
// making it differentiable for gradient-based optimization.
//
// The loss is: 1 - ApproxNDCG, where ApproxNDCG uses soft ranks computed via:
//
//	soft_rank[i] = 1 + sum_j sigmoid(sigma * (s_j - s_i))
//
// Parameters:
//   - labels[0]: Relevance scores (higher = more relevant), shape [batch, list_size]
//   - predictions[0]: Model scores, shape [batch, list_size]
//   - sigma: Steepness of the sigmoid for soft ranking (default 1.0)
//
// References:
//   - "A General Approximation Framework for Direct Optimization of Information Retrieval Measures"
//     (Qin et al., 2010)
func ApproxNDCGLoss(labels, predictions []*Node, sigma float64) *Node {
	labels0 := labels[0]
	scores := predictions[0]
	dtype := scores.DType()
	g := scores.Graph()

	if labels0.Rank() != 2 || scores.Rank() != 2 {
		Panicf("ApproxNDCGLoss expects labels and predictions of rank 2 [batch, list_size], got %s and %s",
			labels0.Shape(), scores.Shape())
	}

	listSize := scores.Shape().Dimensions[1]

	// Convert labels to same dtype
	relevance := ConvertDType(labels0, dtype)

	// Compute soft ranks using sigmoid
	// soft_rank[i] = 1 + sum_{j != i} sigmoid(sigma * (s_j - s_i))
	scoresI := InsertAxes(scores, -1)     // [batch, list_size, 1]
	scoresJ := InsertAxes(scores, -2)     // [batch, 1, list_size]
	scoreDiff := Sub(scoresJ, scoresI)    // [batch, list_size, list_size]

	// Sigmoid with sigma scaling
	sigmaNode := Scalar(g, dtype, sigma)
	sigmoidDiff := Sigmoid(Mul(scoreDiff, sigmaNode))

	// Sum over j, but exclude i==j (diagonal)
	// Create mask to zero out diagonal - broadcast 2D diagonal to 3D batch shape
	diagMask2D := Diagonal(g, listSize)                              // [list_size, list_size] bool
	diagMask2D = LogicalNot(diagMask2D)                              // True for off-diagonal
	diagMask3D := InsertAxes(diagMask2D, 0)                          // [1, list_size, list_size]
	diagMaskFloat := ConvertDType(diagMask3D, dtype)                 // Broadcast will happen automatically
	maskedSigmoid := Mul(sigmoidDiff, diagMaskFloat)

	// soft_rank = 1 + sum_j (masked sigmoid)
	softRanks := AddScalar(ReduceSum(maskedSigmoid, -1), 1.0)

	// Compute DCG with soft ranks
	// DCG = sum_i (2^rel_i - 1) / log2(1 + soft_rank_i)
	gains := OneMinus(Pow(Scalar(g, dtype, 2.0), Neg(relevance))) // 2^rel - 1 = 1 - 2^(-rel) for stability when rel=0
	gains = Mul(relevance, relevance) // Simple: rel^2 as gain, works for binary relevance

	// Actually use standard gain: (2^rel - 1)
	twoConst := Scalar(g, dtype, 2.0)
	gains = OneMinus(Pow(twoConst, relevance))
	gains = Neg(gains) // 2^rel - 1

	// Discounts: 1 / log2(1 + rank)
	log2e := Scalar(g, dtype, 1.0/math.Log(2.0))
	discounts := Div(ScalarOne(g, dtype), Mul(Log(OnePlus(softRanks)), log2e))

	dcg := ReduceSum(Mul(gains, discounts), -1)

	// Compute ideal DCG (sort by relevance)
	sortedRel := Neg(Sort(Neg(relevance), -1, true)) // Sort descending
	idealRanks := Iota(g, scores.Shape(), -1)        // 0, 1, 2, ... along last axis
	idealRanks = AddScalar(ConvertDType(idealRanks, dtype), 1.0)
	idealDiscounts := Div(ScalarOne(g, dtype), Mul(Log(OnePlus(idealRanks)), log2e))

	idealGains := OneMinus(Pow(twoConst, sortedRel))
	idealGains = Neg(idealGains)

	idcg := ReduceSum(Mul(idealGains, idealDiscounts), -1)

	// NDCG = DCG / IDCG
	eps := epsilonForDType(g, dtype)
	ndcg := Div(dcg, Max(idcg, eps))

	// Loss = 1 - NDCG
	loss := OneMinus(ndcg)

	// Factor in weights and mask
	weightsShape := shapes.Make(dtype, labels0.Shape().Dimensions[:1]...)
	weights, mask := CheckExtraLabelsForWeightsAndMask(weightsShape, labels[1:])
	if weights != nil {
		loss = Mul(loss, weights)
	}
	if mask != nil {
		loss = Where(mask, loss, ZerosLike(loss))
		if !loss.IsScalar() {
			loss = MaskedReduceAllMean(loss, mask)
		}
	} else if !loss.IsScalar() {
		loss = ReduceAllMean(loss)
	}

	return loss
}

// MakeApproxNDCGLossFromContext creates an ApproxNDCGLoss from context parameters.
func MakeApproxNDCGLossFromContext(ctx *context.Context) LossFn {
	sigma := context.GetParamOr(ctx, ParamPairwiseLogisticSigma, 1.0)
	return func(labels, predictions []*Node) *Node {
		return ApproxNDCGLoss(labels, predictions, sigma)
	}
}
