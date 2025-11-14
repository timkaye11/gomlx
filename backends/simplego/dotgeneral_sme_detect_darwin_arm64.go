// Copyright 2025 The GoMLX Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !noasm && darwin && arm64

package simplego

import (
	"sync"
	"syscall"
)

var (
	smeDetected     bool
	smeDetectedOnce sync.Once
)

// detectSME checks for SME support on macOS (Apple M4 and later)
func detectSME() bool {
	smeDetectedOnce.Do(func() {
		// Check for SME support via sysctl
		// hw.optional.arm.FEAT_SME returns binary value: \x01 means available
		hasSME, err := syscall.Sysctl("hw.optional.arm.FEAT_SME")
		if err == nil && len(hasSME) > 0 && hasSME[0] == 1 {
			smeDetected = true
		}
	})
	return smeDetected
}

// hasSME indicates whether SME SIMD optimizations are available.
//
// SME (Scalable Matrix Extension) is ARM's latest SIMD technology, providing:
// - Variable-length vectors (128-2048 bits, determined at runtime)
// - Streaming mode optimized for matrix operations
// - Hardware support for FP32 multiply-accumulate operations
//
// Available on:
// - Apple M4 and later (2024+)
// - ARM Neoverse V2 and later server CPUs
//
// Detection: Uses macOS sysctl hw.optional.arm.FEAT_SME
var hasSME = detectSME()
