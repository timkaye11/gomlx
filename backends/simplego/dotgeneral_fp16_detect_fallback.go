// Copyright 2025 The GoMLX Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !arm64 || (arm64 && !darwin && !linux)

package simplego

// detectFP16NEON returns false on non-ARM64 platforms and unsupported ARM64 OSes.
func detectFP16NEON() bool {
	return false
}

// detectBF16NEON returns false on non-ARM64 platforms and unsupported ARM64 OSes.
func detectBF16NEON() bool {
	return false
}
