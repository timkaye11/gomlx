// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package lora

import (
	"github.com/gomlx/gomlx/pkg/ml/adapters"
	"github.com/pkg/errors"
)

// Factory creates LoRA adapter instances.
type Factory struct{}

// NewFactory creates a new LoRA factory.
func NewFactory() *Factory {
	return &Factory{}
}

// Type returns the adapter type this factory creates.
func (f *Factory) Type() adapters.AdapterType {
	return adapters.AdapterTypeLoRA
}

// Create creates a new LoRA adapter with the given name and configuration.
func (f *Factory) Create(name string, config adapters.AdapterConfig) (adapters.Adapter, error) {
	loraConfig, ok := config.(*Config)
	if !ok {
		return nil, errors.Errorf("expected *lora.Config, got %T", config)
	}
	return New(name, loraConfig)
}

func init() {
	// Register the LoRA factory with the global registry
	adapters.MustRegister(NewFactory())
}
