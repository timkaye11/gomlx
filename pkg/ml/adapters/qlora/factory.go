// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package qlora

import (
	"github.com/gomlx/gomlx/pkg/ml/adapters"
	"github.com/pkg/errors"
)

// Factory implements adapters.AdapterFactory for QLoRA.
type Factory struct{}

// NewFactory creates a new QLoRA factory.
func NewFactory() *Factory {
	return &Factory{}
}

// Type returns the adapter type this factory creates.
func (f *Factory) Type() adapters.AdapterType {
	return adapters.AdapterTypeQLoRA
}

// Create creates a new QLoRA adapter with the given name and configuration.
func (f *Factory) Create(name string, config adapters.AdapterConfig) (adapters.Adapter, error) {
	qloraConfig, ok := config.(*Config)
	if !ok {
		return nil, errors.Errorf("expected *qlora.Config, got %T", config)
	}
	return New(name, qloraConfig)
}

// DefaultConfig returns a default QLoRA configuration.
func (f *Factory) DefaultConfig() adapters.AdapterConfig {
	return NewConfig()
}

func init() {
	// Register QLoRA factory with the global registry
	_ = adapters.Register(NewFactory())
}
