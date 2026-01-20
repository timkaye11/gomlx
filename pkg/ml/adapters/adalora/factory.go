// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package adalora

import (
	"github.com/gomlx/gomlx/pkg/ml/adapters"
	"github.com/pkg/errors"
)

// Factory implements adapters.AdapterFactory for AdaLoRA.
type Factory struct{}

// NewFactory creates a new AdaLoRA factory.
func NewFactory() *Factory {
	return &Factory{}
}

// Type returns the adapter type this factory creates.
func (f *Factory) Type() adapters.AdapterType {
	return adapters.AdapterTypeAdaLoRA
}

// Create creates a new AdaLoRA adapter with the given name and configuration.
func (f *Factory) Create(name string, config adapters.AdapterConfig) (adapters.Adapter, error) {
	adaLoRAConfig, ok := config.(*Config)
	if !ok {
		return nil, errors.Errorf("expected *adalora.Config, got %T", config)
	}
	return New(name, adaLoRAConfig)
}

// DefaultConfig returns a default AdaLoRA configuration.
func (f *Factory) DefaultConfig() adapters.AdapterConfig {
	return NewConfig()
}

func init() {
	// Register AdaLoRA factory with the global registry
	_ = adapters.Register(NewFactory())
}
