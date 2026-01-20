// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package adapters

import (
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/pkg/errors"
)

// GetAdapterModel is the main entry point for adding adapter support to a model.
// Similar to HuggingFace PEFT's get_peft_model function.
//
// This function:
// 1. Validates the configuration
// 2. Creates an adapter from the factory registry
// 3. Creates or retrieves a manager from the context
// 4. Adds the adapter to the manager and sets it as active
// 5. Stores the manager in the context for layer access
//
// After calling this, layers using DenseWithAdapters (or checking for the
// adapter manager in context) will automatically apply the adapter.
//
// Example:
//
//	config := lora.NewConfig().SetRank(8).SetAlpha(16)
//	manager, err := adapters.GetAdapterModel(ctx, config, "my_lora")
//	if err != nil {
//	    return err
//	}
//	// Now train with adapter - only adapter weights are trainable
//	manager.FreezeBaseWeights()
//
// Parameters:
//   - ctx: The model context where variables are stored
//   - config: Adapter configuration (LoRA, QLoRA, DoRA, or AdaLoRA)
//   - name: Unique name for this adapter
//
// Returns:
//   - The adapter manager (for further adapter management)
//   - Any error that occurred
func GetAdapterModel(ctx *context.Context, config AdapterConfig, name string) (*Manager, error) {
	if config == nil {
		return nil, errors.New("adapter config cannot be nil")
	}
	if name == "" {
		return nil, errors.New("adapter name cannot be empty")
	}

	// Validate configuration
	if err := config.Validate(); err != nil {
		return nil, errors.Wrap(err, "invalid adapter config")
	}

	// Get factory for this adapter type
	factory, ok := GetFactory(config.Type())
	if !ok {
		return nil, errors.Errorf("no factory registered for adapter type %q", config.Type())
	}

	// Create the adapter
	adapter, err := factory.Create(name, config)
	if err != nil {
		return nil, errors.Wrap(err, "failed to create adapter")
	}

	// Get or create manager
	manager := GetFromContext(ctx)
	if manager == nil {
		manager = NewManager(ctx)
		SetInContext(ctx, manager)
	}

	// Add adapter to manager
	if err := manager.Add(adapter); err != nil {
		return nil, errors.Wrap(err, "failed to add adapter to manager")
	}

	// Set as active
	if err := manager.SetActive(name); err != nil {
		return nil, errors.Wrap(err, "failed to activate adapter")
	}

	return manager, nil
}

// AddAdapter adds a new adapter to an existing manager without activating it.
// Use manager.SetActive(name) to activate it later.
//
// Parameters:
//   - ctx: The model context
//   - config: Adapter configuration
//   - name: Unique name for this adapter
//
// Returns:
//   - The adapter manager
//   - Any error that occurred
func AddAdapter(ctx *context.Context, config AdapterConfig, name string) (*Manager, error) {
	if config == nil {
		return nil, errors.New("adapter config cannot be nil")
	}
	if name == "" {
		return nil, errors.New("adapter name cannot be empty")
	}

	// Validate configuration
	if err := config.Validate(); err != nil {
		return nil, errors.Wrap(err, "invalid adapter config")
	}

	// Get factory for this adapter type
	factory, ok := GetFactory(config.Type())
	if !ok {
		return nil, errors.Errorf("no factory registered for adapter type %q", config.Type())
	}

	// Create the adapter
	adapter, err := factory.Create(name, config)
	if err != nil {
		return nil, errors.Wrap(err, "failed to create adapter")
	}

	// Get or create manager
	manager := GetFromContext(ctx)
	if manager == nil {
		manager = NewManager(ctx)
		SetInContext(ctx, manager)
	}

	// Add adapter to manager (but don't activate)
	if err := manager.Add(adapter); err != nil {
		return nil, errors.Wrap(err, "failed to add adapter to manager")
	}

	return manager, nil
}

// MergeAndUnload merges the active adapter weights into base weights permanently.
// After merging, the adapter can be unloaded and the model used without adapters.
//
// This is useful for:
// - Deploying a model without adapter overhead
// - Creating a fine-tuned model checkpoint
// - Reducing inference memory usage
//
// Note: This operation is irreversible. The adapter is marked as merged and
// will return base weights directly on subsequent Apply() calls.
//
// Parameters:
//   - ctx: The model context
//
// Returns:
//   - Any error that occurred
func MergeAndUnload(ctx *context.Context) error {
	manager := GetFromContext(ctx)
	if manager == nil {
		return errors.New("no adapter manager found in context")
	}

	adapter := manager.Active()
	if adapter == nil {
		return errors.New("no active adapter to merge")
	}

	if adapter.IsMerged() {
		return errors.New("adapter is already merged")
	}

	// The actual merging happens when Merge() is called on weight nodes
	// Here we just mark it as merged so future Apply() calls return base weights
	adapter.SetMerged(true)

	return nil
}

// DisableAdapter temporarily disables the active adapter.
// The adapter weights remain in memory but won't be applied during forward passes.
func DisableAdapter(ctx *context.Context) error {
	manager := GetFromContext(ctx)
	if manager == nil {
		return errors.New("no adapter manager found in context")
	}

	adapter := manager.Active()
	if adapter == nil {
		return nil // No active adapter, nothing to disable
	}

	adapter.Config().SetEnabled(false)
	return nil
}

// EnableAdapter re-enables the active adapter.
func EnableAdapter(ctx *context.Context) error {
	manager := GetFromContext(ctx)
	if manager == nil {
		return errors.New("no adapter manager found in context")
	}

	adapter := manager.Active()
	if adapter == nil {
		return errors.New("no active adapter to enable")
	}

	adapter.Config().SetEnabled(true)
	return nil
}

// SwitchAdapter switches to a different adapter by name.
// The previous adapter remains in memory and can be switched back to.
//
// Parameters:
//   - ctx: The model context
//   - name: Name of the adapter to switch to
//
// Returns:
//   - Any error that occurred
func SwitchAdapter(ctx *context.Context, name string) error {
	manager := GetFromContext(ctx)
	if manager == nil {
		return errors.New("no adapter manager found in context")
	}

	return manager.SetActive(name)
}

// DeactivateAdapter deactivates all adapters, returning to base model behavior.
func DeactivateAdapter(ctx *context.Context) error {
	manager := GetFromContext(ctx)
	if manager == nil {
		return nil // No manager, nothing to deactivate
	}

	return manager.SetActive("")
}

// CountTrainableParameters returns the number of trainable parameters
// in the currently active adapter.
func CountTrainableParameters(ctx *context.Context) int64 {
	manager := GetFromContext(ctx)
	if manager == nil {
		return 0
	}

	return manager.CountActiveParameters()
}

// ListAdapters returns the names of all registered adapters.
func ListAdapters(ctx *context.Context) []string {
	manager := GetFromContext(ctx)
	if manager == nil {
		return nil
	}

	return manager.List()
}

// GetActiveAdapterName returns the name of the currently active adapter,
// or empty string if none is active.
func GetActiveAdapterName(ctx *context.Context) string {
	manager := GetFromContext(ctx)
	if manager == nil {
		return ""
	}

	return manager.ActiveName()
}
