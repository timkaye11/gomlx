// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package metrics

import (
	"testing"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"

	_ "github.com/gomlx/gomlx/backends/default"
)

func TestNDCGMetric(t *testing.T) {
	graphtest.RunTestGraphFn(t, "NDCG",
		func(g *Graph) (inputs, outputs []*Node) {
			// Perfect ranking: items already sorted by relevance
			labels := Const(g, [][]float32{
				{3, 2, 1, 0}, // Decreasing relevance
			})
			scores := Const(g, [][]float32{
				{1.0, 0.7, 0.4, 0.1}, // Correct order
			})
			inputs = []*Node{labels, scores}
			outputs = []*Node{
				ndcgCompute([]*Node{labels}, []*Node{scores}, 0), // k=0 means all
			}
			return
		}, []any{
			// Perfect ranking should have NDCG = 1.0
			float32(1.0),
		}, 0.01)
}

func TestNDCGMetricWrongOrder(t *testing.T) {
	graphtest.RunTestGraphFn(t, "NDCG_WrongOrder",
		func(g *Graph) (inputs, outputs []*Node) {
			// Completely reversed ranking
			labels := Const(g, [][]float32{
				{3, 2, 1, 0}, // Decreasing relevance
			})
			scores := Const(g, [][]float32{
				{0.1, 0.4, 0.7, 1.0}, // Reversed order
			})
			inputs = []*Node{labels, scores}
			outputs = []*Node{
				ndcgCompute([]*Node{labels}, []*Node{scores}, 0),
			}
			return
		}, []any{
			// Reversed ranking should have lower NDCG
			// IDCG = (2^3-1)/log2(2) + (2^2-1)/log2(3) + (2^1-1)/log2(4) + (2^0-1)/log2(5)
			//      = 7/1 + 3/1.585 + 1/2 + 0 = 7 + 1.893 + 0.5 = 9.393
			// DCG with reversed order: items at positions 3,2,1,0 get relevances 0,1,2,3
			float32(0.56), // Approximate NDCG for reversed order
		}, 0.1)
}

func TestNDCGAtK(t *testing.T) {
	graphtest.RunTestGraphFn(t, "NDCG@3",
		func(g *Graph) (inputs, outputs []*Node) {
			// Test NDCG@K with K=3
			labels := Const(g, [][]float32{
				{3, 2, 1, 0}, // Decreasing relevance
			})
			scores := Const(g, [][]float32{
				{1.0, 0.7, 0.4, 0.1}, // Correct order
			})
			inputs = []*Node{labels, scores}
			outputs = []*Node{
				ndcgCompute([]*Node{labels}, []*Node{scores}, 3), // Only consider top 3
			}
			return
		}, []any{
			// Perfect ranking at top 3 should still be 1.0
			float32(1.0),
		}, 0.01)
}

func TestMRRMetric(t *testing.T) {
	graphtest.RunTestGraphFn(t, "MRR",
		func(g *Graph) (inputs, outputs []*Node) {
			// First relevant item is at position 0
			labels := Const(g, [][]float32{
				{1, 0, 0}, // First item is relevant
			})
			scores := Const(g, [][]float32{
				{0.9, 0.5, 0.3}, // Correct order
			})
			inputs = []*Node{labels, scores}
			outputs = []*Node{
				MRRGraph(nil, []*Node{labels}, []*Node{scores}),
			}
			return
		}, []any{
			// First relevant at rank 1, RR = 1/1 = 1.0
			float32(1.0),
		}, 0.01)
}

func TestMRRMetricSecondPosition(t *testing.T) {
	graphtest.RunTestGraphFn(t, "MRR_SecondPos",
		func(g *Graph) (inputs, outputs []*Node) {
			// First relevant item is at position 1
			labels := Const(g, [][]float32{
				{0, 1, 0}, // Second item is relevant
			})
			scores := Const(g, [][]float32{
				{0.9, 0.5, 0.3}, // Item 0 ranked first, but item 1 is relevant
			})
			inputs = []*Node{labels, scores}
			outputs = []*Node{
				MRRGraph(nil, []*Node{labels}, []*Node{scores}),
			}
			return
		}, []any{
			// First relevant at rank 2, RR = 1/2 = 0.5
			float32(0.5),
		}, 0.01)
}

func TestMRRMetricBatch(t *testing.T) {
	graphtest.RunTestGraphFn(t, "MRR_Batch",
		func(g *Graph) (inputs, outputs []*Node) {
			// Batch with different positions for first relevant
			labels := Const(g, [][]float32{
				{1, 0, 0}, // First relevant at position 0
				{0, 1, 0}, // First relevant at position 1
				{0, 0, 1}, // First relevant at position 2
			})
			scores := Const(g, [][]float32{
				{0.9, 0.5, 0.3}, // Correct ranking for batch 0
				{0.9, 0.5, 0.3}, // Relevant item ranked second
				{0.9, 0.5, 0.3}, // Relevant item ranked third
			})
			inputs = []*Node{labels, scores}
			outputs = []*Node{
				MRRGraph(nil, []*Node{labels}, []*Node{scores}),
			}
			return
		}, []any{
			// MRR = (1/1 + 1/2 + 1/3) / 3 = (1 + 0.5 + 0.333) / 3 = 0.611
			float32(0.611),
		}, 0.02)
}

func TestMAPMetric(t *testing.T) {
	graphtest.RunTestGraphFn(t, "MAP",
		func(g *Graph) (inputs, outputs []*Node) {
			// Two relevant items at positions 0 and 2
			labels := Const(g, [][]float32{
				{1, 0, 1, 0}, // Items 0 and 2 are relevant
			})
			scores := Const(g, [][]float32{
				{1.0, 0.8, 0.6, 0.4}, // Item 0 and 2 correctly ranked
			})
			inputs = []*Node{labels, scores}
			outputs = []*Node{
				MAPGraph(nil, []*Node{labels}, []*Node{scores}),
			}
			return
		}, []any{
			// P@1 = 1/1 = 1 (relevant at pos 0)
			// P@3 = 2/3 = 0.667 (relevant at pos 2)
			// AP = (1 + 0.667) / 2 = 0.833
			float32(0.833),
		}, 0.05)
}

func TestMAPMetricPerfect(t *testing.T) {
	graphtest.RunTestGraphFn(t, "MAP_Perfect",
		func(g *Graph) (inputs, outputs []*Node) {
			// All relevant items at the top
			labels := Const(g, [][]float32{
				{1, 1, 0, 0}, // First two items are relevant
			})
			scores := Const(g, [][]float32{
				{1.0, 0.9, 0.5, 0.3}, // Relevant items ranked first
			})
			inputs = []*Node{labels, scores}
			outputs = []*Node{
				MAPGraph(nil, []*Node{labels}, []*Node{scores}),
			}
			return
		}, []any{
			// P@1 = 1/1 = 1, P@2 = 2/2 = 1
			// AP = (1 + 1) / 2 = 1.0
			float32(1.0),
		}, 0.01)
}

func TestPrecisionAtK(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Precision@3",
		func(g *Graph) (inputs, outputs []*Node) {
			labels := Const(g, [][]float32{
				{1, 1, 0, 0, 1}, // Items 0, 1, 4 are relevant
			})
			scores := Const(g, [][]float32{
				{0.9, 0.8, 0.7, 0.6, 0.5}, // Items 0, 1, 2 in top 3
			})
			inputs = []*Node{labels, scores}
			outputs = []*Node{
				precisionAtKCompute([]*Node{labels}, []*Node{scores}, 3),
			}
			return
		}, []any{
			// Top 3: items 0, 1, 2 -> 2 relevant out of 3
			// Precision@3 = 2/3 = 0.667
			float32(0.667),
		}, 0.02)
}

func TestPrecisionAtKPerfect(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Precision@2_Perfect",
		func(g *Graph) (inputs, outputs []*Node) {
			labels := Const(g, [][]float32{
				{1, 1, 0, 0}, // First two are relevant
			})
			scores := Const(g, [][]float32{
				{0.9, 0.8, 0.3, 0.1}, // Relevant items at top
			})
			inputs = []*Node{labels, scores}
			outputs = []*Node{
				precisionAtKCompute([]*Node{labels}, []*Node{scores}, 2),
			}
			return
		}, []any{
			// Top 2 are both relevant: 2/2 = 1.0
			float32(1.0),
		}, 0.01)
}

func TestRecallAtK(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Recall@3",
		func(g *Graph) (inputs, outputs []*Node) {
			labels := Const(g, [][]float32{
				{1, 1, 0, 0, 1}, // Items 0, 1, 4 are relevant (3 total)
			})
			scores := Const(g, [][]float32{
				{0.9, 0.8, 0.7, 0.6, 0.5}, // Items 0, 1, 2 in top 3
			})
			inputs = []*Node{labels, scores}
			outputs = []*Node{
				recallAtKCompute([]*Node{labels}, []*Node{scores}, 3),
			}
			return
		}, []any{
			// Top 3 contains 2 relevant out of 3 total
			// Recall@3 = 2/3 = 0.667
			float32(0.667),
		}, 0.02)
}

func TestRecallAtKPerfect(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Recall@3_Perfect",
		func(g *Graph) (inputs, outputs []*Node) {
			labels := Const(g, [][]float32{
				{1, 1, 1, 0, 0}, // First three are relevant
			})
			scores := Const(g, [][]float32{
				{0.9, 0.8, 0.7, 0.3, 0.1}, // All relevant items at top
			})
			inputs = []*Node{labels, scores}
			outputs = []*Node{
				recallAtKCompute([]*Node{labels}, []*Node{scores}, 3),
			}
			return
		}, []any{
			// Top 3 contains all 3 relevant items
			// Recall@3 = 3/3 = 1.0
			float32(1.0),
		}, 0.01)
}

func TestRecallAtKBatch(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Recall@2_Batch",
		func(g *Graph) (inputs, outputs []*Node) {
			labels := Const(g, [][]float32{
				{1, 1, 0, 0}, // 2 relevant
				{1, 0, 1, 0}, // 2 relevant
			})
			scores := Const(g, [][]float32{
				{0.9, 0.8, 0.3, 0.1}, // Both relevant in top 2
				{0.9, 0.8, 0.3, 0.1}, // Only 1 relevant in top 2
			})
			inputs = []*Node{labels, scores}
			outputs = []*Node{
				recallAtKCompute([]*Node{labels}, []*Node{scores}, 2),
			}
			return
		}, []any{
			// Batch 0: 2/2 = 1.0
			// Batch 1: 1/2 = 0.5
			// Mean = 0.75
			float32(0.75),
		}, 0.02)
}
