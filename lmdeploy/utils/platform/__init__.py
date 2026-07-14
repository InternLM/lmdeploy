# Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Platform utilities for LMDeploy."""

from .platform import (
    # Enums
    OSType,
    ArchType,
    GPUVendor,
    # Classes
    PlatformInfo,
    # Functions
    detect_platform,
    get_platform_info,
    clear_platform_cache,
    # Convenience functions
    is_linux,
    is_windows,
    is_darwin,
    has_nvidia_gpu,
    has_amd_gpu,
    has_gpu,
    get_gpu_vendor,
    get_cuda_version,
    get_rocm_version,
    # OS type detection
    get_os_type,
    get_arch_type,
)

from .toolchain import (
    # Enums
    CompilerType,
    # Classes
    CompilerInfo,
    ToolchainInfo,
    # Functions
    detect_toolchain,
    get_toolchain_info,
    clear_toolchain_cache,
    detect_c_compiler,
    detect_cxx_compiler,
    detect_cuda_compiler,
    detect_hip_compiler,
)

__all__ = [
    # Platform enums
    'OSType',
    'ArchType',
    'GPUVendor',
    # Platform classes
    'PlatformInfo',
    # Platform functions
    'detect_platform',
    'get_platform_info',
    'clear_platform_cache',
    'is_linux',
    'is_windows',
    'is_darwin',
    'has_nvidia_gpu',
    'has_amd_gpu',
    'has_gpu',
    'get_gpu_vendor',
    'get_cuda_version',
    'get_rocm_version',
    'get_os_type',
    'get_arch_type',
    # Toolchain enums
    'CompilerType',
    # Toolchain classes
    'CompilerInfo',
    'ToolchainInfo',
    # Toolchain functions
    'detect_toolchain',
    'get_toolchain_info',
    'clear_toolchain_cache',
    'detect_c_compiler',
    'detect_cxx_compiler',
    'detect_cuda_compiler',
    'detect_hip_compiler',
]
