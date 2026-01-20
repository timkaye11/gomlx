// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package adapters

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/pkg/errors"
)

// AdapterCheckpoint holds the metadata and weights for saving/loading adapters.
type AdapterCheckpoint struct {
	// Name is the adapter instance name.
	Name string `json:"name"`

	// Type is the adapter type (lora, qlora, etc.).
	Type AdapterType `json:"type"`

	// Config holds the serialized adapter configuration.
	Config map[string]any `json:"config"`

	// Variables holds the adapter variable metadata.
	Variables []AdapterVariable `json:"variables"`
}

// AdapterVariable represents a serialized adapter variable.
type AdapterVariable struct {
	// Name is the full parameter name (scope/name).
	Name string `json:"name"`

	// Dimensions of the tensor shape.
	Dimensions []int `json:"dimensions"`

	// DType is the data type.
	DType dtypes.DType `json:"dtype"`

	// Offset is the byte offset in the binary file.
	Offset int `json:"offset"`

	// Length is the byte length.
	Length int `json:"length"`
}

const (
	// AdapterJSONSuffix is the suffix for adapter checkpoint JSON files.
	AdapterJSONSuffix = ".adapter.json"

	// AdapterBinSuffix is the suffix for adapter checkpoint binary files.
	AdapterBinSuffix = ".adapter.bin"
)

// SaveAdapter saves an adapter's weights to the specified path.
// The path should be a base name; .adapter.json and .adapter.bin suffixes will be added.
//
// Only adapter-specific variables are saved (those under the adapter's scope).
func SaveAdapter(ctx *context.Context, adapter Adapter, basePath string) error {
	if adapter == nil {
		return errors.New("adapter is nil")
	}

	checkpoint := &AdapterCheckpoint{
		Name:      adapter.Name(),
		Type:      adapter.Type(),
		Config:    serializeConfig(adapter.Config()),
		Variables: make([]AdapterVariable, 0),
	}

	// Find adapter variables by looking for variables scoped under the adapter name
	adapterScope := adapter.Name()
	var variableData []byte
	offset := 0

	for v := range ctx.IterVariables() {
		paramName := v.ParameterName()
		// Check if this variable belongs to the adapter
		if !isAdapterVariable(paramName, adapterScope) {
			continue
		}

		tensor := v.MustValue()
		shape := tensor.Shape()
		var data []byte
		err := tensor.ConstBytes(func(bytes []byte) {
			data = make([]byte, len(bytes))
			copy(data, bytes)
		})
		if err != nil {
			return errors.Wrapf(err, "failed to read variable %s", paramName)
		}

		checkpoint.Variables = append(checkpoint.Variables, AdapterVariable{
			Name:       paramName,
			Dimensions: shape.Dimensions,
			DType:      shape.DType,
			Offset:     offset,
			Length:     len(data),
		})

		variableData = append(variableData, data...)
		offset += len(data)
	}

	// Sort variables by name for consistent ordering
	sort.Slice(checkpoint.Variables, func(i, j int) bool {
		return checkpoint.Variables[i].Name < checkpoint.Variables[j].Name
	})

	// Recalculate offsets after sorting
	offset = 0
	sortedData := make([]byte, 0, len(variableData))
	for i := range checkpoint.Variables {
		v := ctx.GetVariableByScopeAndName(
			context.VariableScopeAndNameFromParameterName(checkpoint.Variables[i].Name))
		if v == nil {
			return errors.Errorf("variable %s not found during save", checkpoint.Variables[i].Name)
		}
		tensor := v.MustValue()
		err := tensor.ConstBytes(func(bytes []byte) {
			checkpoint.Variables[i].Offset = offset
			checkpoint.Variables[i].Length = len(bytes)
			sortedData = append(sortedData, bytes...)
			offset += len(bytes)
		})
		if err != nil {
			return errors.Wrapf(err, "failed to read variable %s", checkpoint.Variables[i].Name)
		}
	}

	// Write JSON metadata
	jsonPath := basePath + AdapterJSONSuffix
	jsonData, err := json.MarshalIndent(checkpoint, "", "  ")
	if err != nil {
		return errors.Wrap(err, "failed to marshal checkpoint JSON")
	}
	if err := os.WriteFile(jsonPath, jsonData, 0644); err != nil {
		return errors.Wrapf(err, "failed to write checkpoint JSON to %s", jsonPath)
	}

	// Write binary data
	binPath := basePath + AdapterBinSuffix
	if err := os.WriteFile(binPath, sortedData, 0644); err != nil {
		return errors.Wrapf(err, "failed to write checkpoint binary to %s", binPath)
	}

	return nil
}

// LoadAdapterWeights loads adapter weights from a checkpoint into the context.
// The path should be a base name; .adapter.json and .adapter.bin suffixes will be added.
//
// This function loads the weights but does not create the adapter instance.
// Use CreateAdapter with the loaded config to create the adapter.
func LoadAdapterWeights(ctx *context.Context, basePath string) (*AdapterCheckpoint, error) {
	// Read JSON metadata
	jsonPath := basePath + AdapterJSONSuffix
	jsonData, err := os.ReadFile(jsonPath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to read checkpoint JSON from %s", jsonPath)
	}

	var checkpoint AdapterCheckpoint
	if err := json.Unmarshal(jsonData, &checkpoint); err != nil {
		return nil, errors.Wrap(err, "failed to unmarshal checkpoint JSON")
	}

	// Read binary data
	binPath := basePath + AdapterBinSuffix
	binData, err := os.ReadFile(binPath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to read checkpoint binary from %s", binPath)
	}

	// Load variables into context
	for _, varInfo := range checkpoint.Variables {
		if varInfo.Offset+varInfo.Length > len(binData) {
			return nil, errors.Errorf("variable %s data extends beyond file bounds", varInfo.Name)
		}

		data := binData[varInfo.Offset : varInfo.Offset+varInfo.Length]
		shape := shapes.Make(varInfo.DType, varInfo.Dimensions...)
		tensor := tensors.FromShape(shape)
		err := tensor.MutableBytes(func(bytes []byte) {
			copy(bytes, data)
		})
		if err != nil {
			return nil, errors.Wrapf(err, "failed to write tensor data for %s", varInfo.Name)
		}

		// Set the variable in context
		scope, name := context.VariableScopeAndNameFromParameterName(varInfo.Name)
		ctxScoped := ctx.InAbsPath(scope)
		v := ctxScoped.GetVariableByScopeAndName(scope, name)
		if v != nil {
			v.MustSetValue(tensor)
		} else {
			ctxScoped.VariableWithValue(name, tensor)
		}
	}

	return &checkpoint, nil
}

// ListAdapterCheckpoints returns the base paths of adapter checkpoints in a directory.
func ListAdapterCheckpoints(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to read directory %s", dir)
	}

	var checkpoints []string
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if strings.HasSuffix(name, AdapterJSONSuffix) {
			baseName := strings.TrimSuffix(name, AdapterJSONSuffix)
			// Verify the binary file exists
			binPath := filepath.Join(dir, baseName+AdapterBinSuffix)
			if _, err := os.Stat(binPath); err == nil {
				checkpoints = append(checkpoints, filepath.Join(dir, baseName))
			}
		}
	}

	sort.Strings(checkpoints)
	return checkpoints, nil
}

// isAdapterVariable checks if a parameter name belongs to the given adapter scope.
func isAdapterVariable(paramName, adapterScope string) bool {
	// Adapter variables are typically scoped like: adapters/adapter_name/layer/lora_A
	// We check if the parameter name contains the adapter scope
	parts := strings.Split(paramName, "/")
	for i, part := range parts {
		if part == adapterScope {
			// Found the adapter scope, this is an adapter variable
			return true
		}
		// Check common adapter patterns
		if part == "adapters" && i+1 < len(parts) && parts[i+1] == adapterScope {
			return true
		}
	}
	// Also check for lora_ prefix patterns
	return strings.Contains(paramName, "lora_") ||
		strings.Contains(paramName, "dora_") ||
		strings.Contains(paramName, "ada_")
}

// serializeConfig converts an AdapterConfig to a map for JSON serialization.
func serializeConfig(config AdapterConfig) map[string]any {
	if config == nil {
		return nil
	}

	result := map[string]any{
		"type":           string(config.Type()),
		"enabled":        config.Enabled(),
		"target_modules": config.TargetModules(),
	}

	return result
}

// AdapterCheckpointInfo provides summary information about a saved adapter checkpoint.
type AdapterCheckpointInfo struct {
	Name          string
	Type          AdapterType
	NumVariables  int
	TotalBytes    int64
	VariableNames []string
}

// GetCheckpointInfo returns summary information about an adapter checkpoint.
func GetCheckpointInfo(basePath string) (*AdapterCheckpointInfo, error) {
	jsonPath := basePath + AdapterJSONSuffix
	jsonData, err := os.ReadFile(jsonPath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to read checkpoint JSON from %s", jsonPath)
	}

	var checkpoint AdapterCheckpoint
	if err := json.Unmarshal(jsonData, &checkpoint); err != nil {
		return nil, errors.Wrap(err, "failed to unmarshal checkpoint JSON")
	}

	info := &AdapterCheckpointInfo{
		Name:          checkpoint.Name,
		Type:          checkpoint.Type,
		NumVariables:  len(checkpoint.Variables),
		VariableNames: make([]string, len(checkpoint.Variables)),
	}

	for i, v := range checkpoint.Variables {
		info.TotalBytes += int64(v.Length)
		info.VariableNames[i] = v.Name
	}

	return info, nil
}
