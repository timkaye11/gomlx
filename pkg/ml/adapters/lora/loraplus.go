// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package lora

import (
	"strings"

	"github.com/gomlx/gomlx/pkg/ml/context"
)

// LoRA+ (LoRA Plus) uses different learning rates for the A and B matrices.
// Research shows that using a higher learning rate for B improves convergence.
//
// Reference: "LoRA+: Efficient Low Rank Adaptation of Large Models"
// https://arxiv.org/abs/2402.12354
//
// Usage:
//
//	config := lora.NewConfig().SetUseLoRAPlus(true).SetLoRAPlusRatio(16.0)
//	// Store config in context for optimizer to use
//	ctx.SetParam(lora.ParamLoRAPlusConfig, config)
//
//	// In your optimizer or training loop, use GetLearningRateMultiplier
//	// to scale the learning rate per variable.

// ParamLoRAPlusConfig is the context parameter key for storing LoRA+ configuration.
// Set this to a *Config with UseLoRAPlus enabled.
const ParamLoRAPlusConfig = "lora_plus_config"

// GetLearningRateMultiplier returns the learning rate multiplier for a variable.
// For LoRA+ enabled configs:
//   - Returns 1.0 for lora_A matrices (and non-LoRA variables)
//   - Returns config.LoRAPlusRatio for lora_B matrices
//
// If LoRA+ is not enabled or config is nil, returns 1.0.
func GetLearningRateMultiplier(ctx *context.Context, v *context.Variable) float64 {
	configAny, found := ctx.GetParam(ParamLoRAPlusConfig)
	if !found {
		return 1.0
	}

	config, ok := configAny.(*Config)
	if !ok || config == nil || !config.UseLoRAPlus {
		return 1.0
	}

	return GetLearningRateMultiplierForName(config, v.ParameterName())
}

// GetLearningRateMultiplierForName returns the multiplier based on variable name.
func GetLearningRateMultiplierForName(config *Config, varName string) float64 {
	if config == nil || !config.UseLoRAPlus {
		return 1.0
	}

	if IsLoRABVariable(varName) {
		return config.LoRAPlusRatio
	}
	return 1.0
}

// IsLoRAAVariable returns true if the variable name indicates a LoRA A matrix.
func IsLoRAAVariable(name string) bool {
	return strings.Contains(name, "lora_A")
}

// IsLoRABVariable returns true if the variable name indicates a LoRA B matrix.
func IsLoRABVariable(name string) bool {
	return strings.Contains(name, "lora_B")
}

// IsLoRAVariable returns true if the variable name indicates any LoRA variable (A or B).
func IsLoRAVariable(name string) bool {
	return IsLoRAAVariable(name) || IsLoRABVariable(name)
}

// PartitionVariables separates context variables into LoRA A, LoRA B, and other variables.
// This is useful for creating optimizer parameter groups with different learning rates.
func PartitionVariables(ctx *context.Context) (loraA, loraB, other []*context.Variable) {
	for v := range ctx.IterVariables() {
		name := v.ParameterName()
		switch {
		case IsLoRAAVariable(name):
			loraA = append(loraA, v)
		case IsLoRABVariable(name):
			loraB = append(loraB, v)
		default:
			other = append(other, v)
		}
	}
	return
}

// GetLoRAVariables returns all LoRA variables (both A and B) from the context.
func GetLoRAVariables(ctx *context.Context) []*context.Variable {
	var loraVars []*context.Variable
	for v := range ctx.IterVariables() {
		if IsLoRAVariable(v.ParameterName()) {
			loraVars = append(loraVars, v)
		}
	}
	return loraVars
}

// CountLoRAParameters counts the total number of LoRA parameters.
func CountLoRAParameters(ctx *context.Context) int64 {
	var count int64
	for v := range ctx.IterVariables() {
		if IsLoRAVariable(v.ParameterName()) {
			count += int64(v.Shape().Size())
		}
	}
	return count
}

// SetupLoRAPlus stores the LoRA+ config in the context for optimizers to use.
// This is a convenience function that sets ParamLoRAPlusConfig.
func SetupLoRAPlus(ctx *context.Context, config *Config) {
	if config != nil && config.UseLoRAPlus {
		ctx.SetParam(ParamLoRAPlusConfig, config)
	}
}
