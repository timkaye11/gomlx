// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package losses

import (
	"fmt"
	"math"
	"testing"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/stretchr/testify/require"

	_ "github.com/gomlx/gomlx/backends/default"
)

func TestPairwiseLogisticLoss(t *testing.T) {
	graphtest.RunTestGraphFn(t, "PairwiseLogisticLoss",
		func(g *Graph) (inputs, outputs []*Node) {
			// Simple test case: batch of 2, list size of 3
			// First batch: item 0 is relevant, items 1,2 are not
			// Second batch: item 2 is relevant, items 0,1 are not
			labels := Const(g, [][]float32{
				{1, 0, 0}, // First query: item 0 relevant
				{0, 0, 1}, // Second query: item 2 relevant
			})
			scores := Const(g, [][]float32{
				{0.9, 0.1, 0.2}, // Correct ranking for first query
				{0.3, 0.1, 0.8}, // Correct ranking for second query
			})
			inputs = []*Node{labels, scores}
			outputs = []*Node{
				PairwiseLogisticLoss([]*Node{labels}, []*Node{scores}, 1.0),
			}
			return
		}, []any{
			// Expected: loss should be small since rankings are correct
			// For batch 1: pairs (0,1) with diff 0.8, (0,2) with diff 0.7
			// loss = log(1 + exp(-0.8)) + log(1 + exp(-0.7)) ≈ 0.371 + 0.403 = 0.774 / 2 pairs = 0.387
			// For batch 2: pairs (2,0) with diff 0.5, (2,1) with diff 0.7
			// loss = log(1 + exp(-0.5)) + log(1 + exp(-0.7)) ≈ 0.474 + 0.403 = 0.877 / 2 pairs = 0.438
			// Mean = (0.387 + 0.438) / 2 = 0.412
			float32(0.412),
		}, 0.05)
}

func TestPairwiseLogisticLossIncorrectRanking(t *testing.T) {
	graphtest.RunTestGraphFn(t, "PairwiseLogisticLossIncorrect",
		func(g *Graph) (inputs, outputs []*Node) {
			// Test case where ranking is incorrect
			labels := Const(g, [][]float32{
				{1, 0, 0}, // Item 0 is relevant
			})
			scores := Const(g, [][]float32{
				{0.1, 0.9, 0.5}, // Wrong: irrelevant item 1 scored highest
			})
			inputs = []*Node{labels, scores}
			outputs = []*Node{
				PairwiseLogisticLoss([]*Node{labels}, []*Node{scores}, 1.0),
			}
			return
		}, []any{
			// Loss should be higher for incorrect ranking
			// pairs (0,1) with diff -0.8: loss = log(1 + exp(0.8)) ≈ 1.17
			// pairs (0,2) with diff -0.4: loss = log(1 + exp(0.4)) ≈ 0.91
			// Mean = (1.17 + 0.91) / 2 = 1.04
			float32(1.04),
		}, 0.1)
}

func TestMarginRankingLoss(t *testing.T) {
	graphtest.RunTestGraphFn(t, "MarginRankingLoss",
		func(g *Graph) (inputs, outputs []*Node) {
			labels := Const(g, [][]float32{
				{1, 0, 0},
				{0, 0, 1},
			})
			scores := Const(g, [][]float32{
				{0.9, 0.1, 0.2}, // Good ranking
				{0.3, 0.1, 0.8}, // Good ranking
			})
			inputs = []*Node{labels, scores}
			outputs = []*Node{
				MarginRankingLoss([]*Node{labels}, []*Node{scores}, 0.5),
			}
			return
		}, []any{
			// With margin 0.5:
			// Batch 1: pairs (0,1) diff=0.8 > 0.5, (0,2) diff=0.7 > 0.5 -> loss = 0
			// Batch 2: pairs (2,0) diff=0.5 = 0.5, (2,1) diff=0.7 > 0.5 -> loss = 0
			// Total: 0
			float32(0.0),
		}, 0.01)
}

func TestMarginRankingLossWithViolation(t *testing.T) {
	graphtest.RunTestGraphFn(t, "MarginRankingLossViolation",
		func(g *Graph) (inputs, outputs []*Node) {
			labels := Const(g, [][]float32{
				{1, 0},
			})
			scores := Const(g, [][]float32{
				{0.5, 0.4}, // Margin of 0.1, less than required 0.5
			})
			inputs = []*Node{labels, scores}
			outputs = []*Node{
				MarginRankingLoss([]*Node{labels}, []*Node{scores}, 0.5),
			}
			return
		}, []any{
			// diff = 0.1, margin = 0.5
			// loss = max(0, 0.5 - 0.1) = 0.4
			float32(0.4),
		}, 0.01)
}

func TestListMLELoss(t *testing.T) {
	graphtest.RunTestGraphFn(t, "ListMLELoss",
		func(g *Graph) (inputs, outputs []*Node) {
			// Relevance scores
			labels := Const(g, [][]float32{
				{3, 2, 1}, // Decreasing relevance
			})
			scores := Const(g, [][]float32{
				{1.0, 0.5, 0.2}, // Matching score order (correct ranking)
			})
			inputs = []*Node{labels, scores}
			outputs = []*Node{
				ListMLELoss([]*Node{labels}, []*Node{scores}, 1.0),
			}
			return
		}, []any{
			// Loss should be relatively low for correct ranking
			float32(1.01), // Approximate expected value
		}, 0.2)
}

func TestListNetLoss(t *testing.T) {
	graphtest.RunTestGraphFn(t, "ListNetLoss",
		func(g *Graph) (inputs, outputs []*Node) {
			// ListNet uses softmax of relevance as target distribution, not one-hot
			// For relevance [1,0,0]: softmax([1,0,0]) ≈ [0.576, 0.212, 0.212]
			// For scores [10,0,0]: log_softmax([10,0,0]) ≈ [0, -10, -10]
			// Cross-entropy ≈ -(0.576*0 + 0.212*(-10) + 0.212*(-10)) ≈ 4.24
			labels := Const(g, [][]float32{
				{1, 0, 0}, // Relevance scores (will be softmaxed)
			})
			scores := Const(g, [][]float32{
				{10.0, 0.0, 0.0}, // Clear preference for item 0
			})
			inputs = []*Node{labels, scores}
			outputs = []*Node{
				ListNetLoss([]*Node{labels}, []*Node{scores}, 1.0),
			}
			return
		}, []any{
			float32(4.24), // Cross-entropy between softmax(relevance) and log_softmax(scores)
		}, 0.1)
}

func TestListNetLossHighEntropy(t *testing.T) {
	graphtest.RunTestGraphFn(t, "ListNetLossHighEntropy",
		func(g *Graph) (inputs, outputs []*Node) {
			labels := Const(g, [][]float32{
				{1, 0, 0},
			})
			scores := Const(g, [][]float32{
				{0.0, 0.0, 0.0}, // Uniform scores
			})
			inputs = []*Node{labels, scores}
			outputs = []*Node{
				ListNetLoss([]*Node{labels}, []*Node{scores}, 1.0),
			}
			return
		}, []any{
			// Cross-entropy between one-hot and uniform should be log(3) ≈ 1.1
			float32(math.Log(3)),
		}, 0.05)
}

func TestApproxNDCGLoss(t *testing.T) {
	graphtest.RunTestGraphFn(t, "ApproxNDCGLoss",
		func(g *Graph) (inputs, outputs []*Node) {
			// Correct ranking order
			// Note: ApproxNDCG uses soft ranking with sigmoid, so even perfect ordering
			// has some approximation error depending on sigma
			labels := Const(g, [][]float32{
				{3, 2, 1, 0}, // Decreasing relevance
			})
			scores := Const(g, [][]float32{
				{1.0, 0.7, 0.4, 0.1}, // Correct order
			})
			inputs = []*Node{labels, scores}
			outputs = []*Node{
				ApproxNDCGLoss([]*Node{labels}, []*Node{scores}, 1.0),
			}
			return
		}, []any{
			// With soft ranking (sigma=1.0), even correct order has some loss
			// Loss ≈ 0.30 due to soft rank approximation
			float32(0.30),
		}, 0.05)
}

func TestApproxNDCGLossWrongOrder(t *testing.T) {
	graphtest.RunTestGraphFn(t, "ApproxNDCGLossWrongOrder",
		func(g *Graph) (inputs, outputs []*Node) {
			// Reversed ranking
			labels := Const(g, [][]float32{
				{3, 2, 1, 0}, // Decreasing relevance
			})
			scores := Const(g, [][]float32{
				{0.1, 0.4, 0.7, 1.0}, // Reversed order
			})
			inputs = []*Node{labels, scores}
			outputs = []*Node{
				ApproxNDCGLoss([]*Node{labels}, []*Node{scores}, 1.0),
			}
			return
		}, []any{
			// Wrong ranking should have higher loss than correct ranking (0.30)
			float32(0.39), // Higher loss for reversed order
		}, 0.05)
}

func TestRankingLossGradients(t *testing.T) {
	// Test that all ranking losses are differentiable
	manager := graphtest.BuildTestBackend()

	// Test PairwiseLogisticLoss gradient
	t.Run("PairwiseLogisticGradient", func(t *testing.T) {
		g := NewGraph(manager, "PairwiseLogisticGradient")
		labels := Const(g, [][]float64{{1, 0}})
		scores := Const(g, [][]float64{{0.6, 0.4}})
		loss := PairwiseLogisticLoss([]*Node{labels}, []*Node{scores}, 1.0)
		grads := Gradient(loss, scores)
		g.Compile(loss, grads[0])
		results := g.Run()
		fmt.Printf("PairwiseLogistic: loss=%v, grad=%v\n", results[0], results[1])
		// Just verify gradient exists and is reasonable
		require.NotNil(t, results[1])
	})

	// Test MarginRankingLoss gradient
	t.Run("MarginRankingGradient", func(t *testing.T) {
		g := NewGraph(manager, "MarginRankingGradient")
		labels := Const(g, [][]float64{{1, 0}})
		scores := Const(g, [][]float64{{0.6, 0.4}})
		loss := MarginRankingLoss([]*Node{labels}, []*Node{scores}, 0.5)
		grads := Gradient(loss, scores)
		g.Compile(loss, grads[0])
		results := g.Run()
		fmt.Printf("MarginRanking: loss=%v, grad=%v\n", results[0], results[1])
		require.NotNil(t, results[1])
	})

	// Test ListMLELoss gradient
	t.Run("ListMLEGradient", func(t *testing.T) {
		g := NewGraph(manager, "ListMLEGradient")
		labels := Const(g, [][]float64{{3, 2, 1}})
		scores := Const(g, [][]float64{{0.8, 0.5, 0.2}})
		loss := ListMLELoss([]*Node{labels}, []*Node{scores}, 1.0)
		grads := Gradient(loss, scores)
		g.Compile(loss, grads[0])
		results := g.Run()
		fmt.Printf("ListMLE: loss=%v, grad=%v\n", results[0], results[1])
		require.NotNil(t, results[1])
	})

	// Test ListNetLoss gradient
	t.Run("ListNetGradient", func(t *testing.T) {
		g := NewGraph(manager, "ListNetGradient")
		labels := Const(g, [][]float64{{3, 2, 1}})
		scores := Const(g, [][]float64{{0.8, 0.5, 0.2}})
		loss := ListNetLoss([]*Node{labels}, []*Node{scores}, 1.0)
		grads := Gradient(loss, scores)
		g.Compile(loss, grads[0])
		results := g.Run()
		fmt.Printf("ListNet: loss=%v, grad=%v\n", results[0], results[1])
		require.NotNil(t, results[1])
	})

	// Test ApproxNDCGLoss gradient
	t.Run("ApproxNDCGGradient", func(t *testing.T) {
		g := NewGraph(manager, "ApproxNDCGGradient")
		labels := Const(g, [][]float64{{3, 2, 1}})
		scores := Const(g, [][]float64{{0.8, 0.5, 0.2}})
		loss := ApproxNDCGLoss([]*Node{labels}, []*Node{scores}, 1.0)
		grads := Gradient(loss, scores)
		g.Compile(loss, grads[0])
		results := g.Run()
		fmt.Printf("ApproxNDCG: loss=%v, grad=%v\n", results[0], results[1])
		require.NotNil(t, results[1])
	})
}
