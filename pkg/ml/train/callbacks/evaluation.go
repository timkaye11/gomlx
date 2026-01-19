// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package callbacks

import (
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/train"
)

// EvalMetricsKey is the key used to store evaluation metrics in Loop.SharedData.
const EvalMetricsKey = "eval_metrics"

// PeriodicEvalConfig configures the periodic evaluation callback.
type PeriodicEvalConfig struct {
	// Verbose prints evaluation metrics when evaluation completes.
	Verbose bool
}

// PeriodicEval runs evaluation on a dataset and stores results in SharedData.
// This allows other callbacks (like EarlyStopping) to access evaluation metrics.
type PeriodicEval struct {
	evalDS  train.Dataset
	config  PeriodicEvalConfig
	evalNum int
}

// NewPeriodicEval creates a callback that runs evaluation on the given dataset.
//
// Example:
//
//	periodicEval := callbacks.NewPeriodicEval(evalDataset, callbacks.PeriodicEvalConfig{
//	    Verbose: true,
//	})
//	train.EveryNSteps(loop, 100, "periodic_eval", 50, periodicEval.OnStepFn)
//
//	// Then add early stopping that uses the eval metrics
//	earlyStop := callbacks.NewEarlyStopping(callbacks.EarlyStoppingConfig{
//	    MetricName: "eval_loss",
//	    Patience:   5,
//	})
//	train.EveryNSteps(loop, 100, "early_stop", 51, earlyStop.OnStepFn) // Priority > 50
func NewPeriodicEval(evalDS train.Dataset, config PeriodicEvalConfig) *PeriodicEval {
	return &PeriodicEval{
		evalDS: evalDS,
		config: config,
	}
}

// OnStepFn implements train.OnStepFn.
// It runs evaluation and stores the results in loop.SharedData[EvalMetricsKey].
func (pe *PeriodicEval) OnStepFn(loop *train.Loop, _ []*tensors.Tensor) error {
	evalMetrics, err := loop.Trainer.Eval(pe.evalDS)
	if err != nil {
		return err
	}

	// Store in SharedData for other callbacks to use
	loop.SharedData[EvalMetricsKey] = evalMetrics

	pe.evalNum++

	if pe.config.Verbose {
		pe.printMetrics(loop, evalMetrics)
	}

	return nil
}

// printMetrics prints the evaluation metrics.
func (pe *PeriodicEval) printMetrics(loop *train.Loop, evalMetrics []*tensors.Tensor) {
	metrics := loop.Trainer.EvalMetrics()
	if len(metrics) == 0 {
		return
	}

	// Print header on first eval
	if pe.evalNum == 1 {
		for i, m := range metrics {
			if i > 0 {
				print(", ")
			}
			print(m.Name())
		}
		println()
	}

	// Print values
	for i, m := range metrics {
		if i > 0 {
			print(", ")
		}
		if i < len(evalMetrics) {
			print(m.PrettyPrint(evalMetrics[i]))
		}
	}
	println()
}

// EvalCount returns the number of evaluations performed.
func (pe *PeriodicEval) EvalCount() int {
	return pe.evalNum
}

// CombinedEvalAndEarlyStop creates both a periodic evaluation callback and
// an early stopping callback that uses the evaluation results.
//
// Returns the evaluation callback (to run first) and early stopping callback (to run second).
// Use them with consecutive priorities to ensure proper ordering.
//
// Example:
//
//	evalCb, earlyCb := callbacks.CombinedEvalAndEarlyStop(evalDS, callbacks.EarlyStoppingConfig{
//	    MetricName: "eval_loss",
//	    Patience:   5,
//	    Mode:       callbacks.ModeMin,
//	    Verbose:    true,
//	})
//	train.EveryNSteps(loop, 100, "periodic_eval", 50, evalCb.OnStepFn)
//	train.EveryNSteps(loop, 100, "early_stop", 51, earlyCb.OnStepFn)
func CombinedEvalAndEarlyStop(evalDS train.Dataset, esConfig EarlyStoppingConfig) (*PeriodicEval, *EarlyStopping) {
	evalCb := NewPeriodicEval(evalDS, PeriodicEvalConfig{Verbose: esConfig.Verbose})
	earlyCb := NewEarlyStopping(esConfig)
	return evalCb, earlyCb
}

// CombinedEvalAndBestModel creates both a periodic evaluation callback and
// a best model checkpoint callback that uses the evaluation results.
//
// Returns the evaluation callback (to run first) and best model callback (to run second).
//
// Example:
//
//	evalCb, bestCb := callbacks.CombinedEvalAndBestModel(evalDS, checkpoint, callbacks.BestModelCheckpointConfig{
//	    MetricName: "eval_ndcg",
//	    Mode:       callbacks.ModeMax,
//	    Verbose:    true,
//	})
//	train.EveryNSteps(loop, 100, "periodic_eval", 50, evalCb.OnStepFn)
//	train.EveryNSteps(loop, 100, "best_model", 51, bestCb.OnStepFn)
func CombinedEvalAndBestModel(evalDS train.Dataset, checkpoint Checkpointer, bmConfig BestModelCheckpointConfig) (*PeriodicEval, *BestModelCheckpoint) {
	evalCb := NewPeriodicEval(evalDS, PeriodicEvalConfig{Verbose: bmConfig.Verbose})
	bestCb := NewBestModelCheckpointWithInterface(checkpoint, bmConfig)
	return evalCb, bestCb
}

// Checkpointer is an interface for checkpoint handlers.
// This allows for dependency injection and testing.
type Checkpointer interface {
	Save() error
}

// NewBestModelCheckpointWithInterface creates a BestModelCheckpoint with a Checkpointer interface.
func NewBestModelCheckpointWithInterface(checkpoint Checkpointer, config BestModelCheckpointConfig) *BestModelCheckpoint {
	bm := NewBestModelCheckpoint(nil, config)
	bm.checkpointInterface = checkpoint
	return bm
}
