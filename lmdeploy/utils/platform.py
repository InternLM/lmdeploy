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
"""Platform detection and configuration module.

This module provides unified platform detection for LMDeploy across
different operating systems, architectures, and hardware accelerators.

Supported platforms:
- Linux (x86_64, aarch64, arm)
- Windows (x86_64, i686)
- macOS (x86_64, arm64)
- Hardware accelerators: NVIDIA CUDA, AMD ROCm, Ascend, Cambricon, Apple MPS
"""

import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List


class OSType(Enum):
    """Operating system types."""
    LINUX = "linux"
    WINDOWS = "windows"
    DARWIN = "darwin"
    BSD = "bsd"
    UNKNOWN = "unknown"


class ArchType(Enum):
    """CPU architecture types."""
    X86_64 = "x86_64"
    AARCH64 = "aarch64"
    ARM = "arm"
    I686 = "i686"
    ARM64 = "arm64"  # Alias for aarch64 on Apple
    UNKNOWN = "unknown"


class GPUVendor(Enum):
    """GPU vendor types."""
    NVIDIA = "nvidia"
    AMD = "amd"
    ASCEND = "ascend"
    CAMBRICON = "cambricon"
    APPLE = "apple"
    INTEL = "intel"
    NONE = "none"  # No GPU / CPU only
    UNKNOWN = "unknown"


@dataclass
class PlatformInfo:
    """Platform information container."""

    # Operating system
    os_type: OSType = OSType.UNKNOWN
    os_name: str = ""
    os_version: str = ""

    # CPU architecture
    arch: ArchType = ArchType.UNKNOWN
    arch_name: str = ""

    # GPU information
    gpu_vendor: GPUVendor = GPUVendor.UNKNOWN
    gpu_name: str = ""
    gpu_count: int = 0
    gpu_memory: List[int] = field(default_factory=list)  # MB per GPU

    # SDK/Driver versions
    cuda_version: Optional[str] = None
    rocm_version: Optional[str] = None
    cudnn_version: Optional[str] = None

    # Compiler information
    cxx_compiler: str = ""
    cuda_compiler: str = ""

    # Build environment
    python_version: str = ""
    python_executable: str = ""

    # Platform-specific flags
    is_linux: bool = False
    is_windows: bool = False
    is_darwin: bool = False
    is_64bit: bool = True
    has_gpu: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'os_type': self.os_type.value,
            'os_name': self.os_name,
            'os_version': self.os_version,
            'arch': self.arch.value,
            'arch_name': self.arch_name,
            'gpu_vendor': self.gpu_vendor.value,
            'gpu_name': self.gpu_name,
            'gpu_count': self.gpu_count,
            'cuda_version': self.cuda_version,
            'rocm_version': self.rocm_version,
            'cudnn_version': self.cudnn_version,
            'cxx_compiler': self.cxx_compiler,
            'cuda_compiler': self.cuda_compiler,
            'python_version': self.python_version,
            'is_64bit': self.is_64bit,
            'has_gpu': self.has_gpu,
        }

    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [
            f"Platform: {self.os_name} {self.arch_name}",
            f"OS Type: {self.os_type.value}",
            f"Architecture: {self.arch.value}",
            f"GPU Vendor: {self.gpu_vendor.value}",
        ]

        if self.gpu_name:
            lines.append(f"GPU Name: {self.gpu_name}")
        if self.gpu_count > 0:
            lines.append(f"GPU Count: {self.gpu_count}")

        if self.cuda_version:
            lines.append(f"CUDA Version: {self.cuda_version}")
        if self.rocm_version:
            lines.append(f"ROCm Version: {self.rocm_version}")
        if self.cudnn_version:
            lines.append(f"cuDNN Version: {self.cudnn_version}")

        lines.append(f"Python: {self.python_version}")

        return "\n".join(lines)


def get_os_type() -> OSType:
    """Detect operating system type."""
    system = platform.system().lower()

    if system == "linux":
        return OSType.LINUX
    elif system == "windows" or system == "nt":
        return OSType.WINDOWS
    elif system == "darwin":
        return OSType.DARWIN
    elif "bsd" in system:
        return OSType.BSD
    else:
        return OSType.UNKNOWN


def get_arch_type() -> ArchType:
    """Detect CPU architecture type."""
    machine = platform.machine().lower()

    # Common architecture mappings
    arch_map = {
        "x86_64": ArchType.X86_64,
        "amd64": ArchType.X86_64,
        "aarch64": ArchType.AARCH64,
        "arm64": ArchType.ARM64,  # Apple Silicon
        "armv8l": ArchType.AARCH64,
        "armv7l": ArchType.ARM,
        "armv6l": ArchType.ARM,
        "i686": ArchType.I686,
        "i386": ArchType.I686,
        "i586": ArchType.I686,
    }

    return arch_map.get(machine, ArchType.UNKNOWN)


def _detect_nvidia_gpu() -> Optional[Dict[str, Any]]:
    """Detect NVIDIA GPU using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            gpus = []
            cuda_version = None

            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 2:
                    gpus.append({
                        'name': parts[0],
                        'memory_mb': int(parts[1]) if parts[1].isdigit() else 0,
                    })

            # Get CUDA version
            cuda_result = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap",
                 "--format=csv,noheader", "-u"],
                capture_output=True,
                text=True,
                timeout=5
            )

            return {
                'vendor': GPUVendor.NVIDIA,
                'gpus': gpus,
                'cuda_version': _get_cuda_version(),
            }

    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass

    return None


def _detect_amd_gpu() -> Optional[Dict[str, Any]]:
    """Detect AMD GPU using rocm-smi or rocminfo."""
    try:
        # Try rocm-smi first
        result = subprocess.run(
            ["rocm-smi", "--showproductname", "--json"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0 and result.stdout.strip():
            import json
            try:
                data = json.loads(result.stdout)
                gpus = []
                for gpu_id, info in data.items():
                    if isinstance(info, dict):
                        gpus.append({
                            'name': info.get('Device Name', 'AMD GPU'),
                            'memory_mb': info.get('VRAM', 0),
                        })

                rocm_version = _get_rocm_version()

                return {
                    'vendor': GPUVendor.AMD,
                    'gpus': gpus,
                    'rocm_version': rocm_version,
                }
            except json.JSONDecodeError:
                pass

        # Try rocminfo
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            return {
                'vendor': GPUVendor.AMD,
                'gpus': [{'name': 'AMD GPU', 'memory_mb': 0}],
                'rocm_version': _get_rocm_version(),
            }

    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass

    return None


def _detect_apple_gpu() -> Optional[Dict[str, Any]]:
    """Detect Apple MPS (Metal Performance Shaders)."""
    if platform.system() != "Darwin":
        return None

    try:
        # Check for Apple Silicon
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if "Apple" in result.stdout or result.returncode == 0:
            # Check if we can use MPS
            return {
                'vendor': GPUVendor.APPLE,
                'gpus': [{'name': 'Apple Silicon', 'memory_mb': 0}],
            }

    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass

    return None


def _detect_ascend_npu() -> Optional[Dict[str, Any]]:
    """Detect Huawei Ascend NPU."""
    try:
        # Try ascend-npu-dmi-tool
        result = subprocess.run(
            ["ascend-npu-dmi-tool", "-l"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0 and result.stdout.strip():
            return {
                'vendor': GPUVendor.ASCEND,
                'gpus': [{'name': 'Ascend NPU', 'memory_mb': 0}],
            }

        # Try CANN installation check
        cann_path = os.environ.get('ASCEND_HOME_PATH')
        if cann_path and os.path.exists(cann_path):
            return {
                'vendor': GPUVendor.ASCEND,
                'gpus': [{'name': 'Ascend NPU', 'memory_mb': 0}],
            }

    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass

    return None


def _detect_cambricon() -> Optional[Dict[str, Any]]:
    """Detect Cambricon MLU (Machine Learning Unit)."""
    try:
        # Try CNToolkit info
        result = subprocess.run(
            ["cnmon", "info"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0 and result.stdout.strip():
            return {
                'vendor': GPUVendor.CAMBRICON,
                'gpus': [{'name': 'Cambricon MLU', 'memory_mb': 0}],
            }

        # Try environment variable check
        cambricon_home = os.environ.get('CNNL_HOME')
        if cambricon_home and os.path.exists(cambricon_home):
            return {
                'vendor': GPUVendor.CAMBRICON,
                'gpus': [{'name': 'Cambricon MLU', 'memory_mb': 0}],
            }

    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass

    return None


def _get_cuda_version() -> Optional[str]:
    """Get CUDA version."""
    # Try nvcc first
    try:
        cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
        nvcc_path = os.path.join(cuda_home or '', 'bin', 'nvcc')

        result = subprocess.run(
            [nvcc_path, '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            match = re.search(r'release (\d+\.\d+)', result.stdout)
            if match:
                return match.group(1)
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass

    # Try nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            # Driver version doesn't directly give CUDA version
            # but we can estimate from driver version
            return None  # Would need mapping table
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass

    return None


def _get_rocm_version() -> Optional[str]:
    """Get ROCm version."""
    try:
        # Try rocminfo
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            # Extract ROCm version from output
            match = re.search(r'Hip Version:\s*(\d+\.\d+)', result.stdout)
            if match:
                return match.group(1)

            # Try alternative pattern
            match = re.search(r'ROCm Version:\s*(\d+\.\d+)', result.stdout)
            if match:
                return match.group(1)

    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass

    # Try environment variable
    rocm_path = os.environ.get('ROCM_PATH')
    if rocm_path:
        # Could parse version from path
        pass

    return None


def _get_cudnn_version() -> Optional[str]:
    """Get cuDNN version."""
    try:
        # Try nvcc
        cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
        nvcc_path = os.path.join(cuda_home or '', 'bin', 'nvcc')

        result = subprocess.run(
            [nvcc_path, "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            # cuDNN version is not directly available from nvcc
            # Check for cudnn_version.h
            include_path = os.path.join(cuda_home or '', 'include', 'cudnn_version.h')
            if os.path.exists(include_path):
                with open(include_path, 'r') as f:
                    content = f.read()
                    match = re.search(r'CUDNN_VERSION\s+(\d+)', content)
                    if match:
                        major = int(match.group(1)) // 100
                        minor = (int(match.group(1)) // 10) % 10
                        return f"{major}.{minor}"

    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass

    return None


def _get_cxx_compiler() -> str:
    """Get C++ compiler name."""
    if platform.system() == "Windows":
        # Check MSVC
        try:
            result = subprocess.run(
                ["cl"],
                capture_output=True,
                timeout=5
            )
            if result.returncode != 0 and "Copyright" in str(result.stderr):
                return "msvc"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    # Check common compilers
    compilers = ["g++", "clang++", "icpc", "c++"]
    for compiler in compilers:
        try:
            result = subprocess.run(
                [compiler, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return compiler
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue

    return "unknown"


def _get_cuda_compiler() -> str:
    """Get CUDA compiler (nvcc) path."""
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    nvcc_path = os.path.join(cuda_home or '', 'bin', 'nvcc')

    if os.path.exists(nvcc_path):
        return nvcc_path

    # Try to find nvcc in PATH
    try:
        result = subprocess.run(
            ["which", "nvcc"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return ""


def detect_gpu() -> Dict[str, Any]:
    """Detect GPU vendor and information.

    Returns:
        Dictionary with 'vendor' (GPUVendor) and 'gpus' (list of GPU info).
    """
    # Check order matters - more common GPUs first
    detectors = [
        _detect_nvidia_gpu,
        _detect_amd_gpu,
        _detect_apple_gpu,
        _detect_ascend_npu,
        _detect_cambricon,
    ]

    for detector in detectors:
        result = detector()
        if result:
            return result

    return {
        'vendor': GPUVendor.NONE,
        'gpus': [],
    }


def detect_platform() -> PlatformInfo:
    """Detect complete platform information.

    Returns:
        PlatformInfo object containing all platform details.
    """
    info = PlatformInfo()

    # OS detection
    info.os_type = get_os_type()
    info.os_name = platform.system()
    info.os_version = platform.release()
    info.is_linux = info.os_type == OSType.LINUX
    info.is_windows = info.os_type == OSType.WINDOWS
    info.is_darwin = info.os_type == OSType.DARWIN

    # Architecture detection
    info.arch = get_arch_type()
    info.arch_name = platform.machine()
    info.is_64bit = platform.machine() in ["x86_64", "aarch64", "arm64", "AMD64"]

    # GPU detection
    gpu_info = detect_gpu()
    info.gpu_vendor = gpu_info['vendor']
    info.has_gpu = info.gpu_vendor != GPUVendor.NONE and info.gpu_vendor != GPUVendor.UNKNOWN

    if gpu_info.get('gpus'):
        info.gpu_count = len(gpu_info['gpus'])
        if gpu_info['gpus']:
            info.gpu_name = gpu_info['gpus'][0]['name']
            info.gpu_memory = [g.get('memory_mb', 0) for g in gpu_info['gpus']]

    # SDK versions
    if info.gpu_vendor == GPUVendor.NVIDIA:
        info.cuda_version = gpu_info.get('cuda_version') or _get_cuda_version()
        info.cudnn_version = _get_cudnn_version()
    elif info.gpu_vendor == GPUVendor.AMD:
        info.rocm_version = gpu_info.get('rocm_version') or _get_rocm_version()

    # Compiler information
    info.cxx_compiler = _get_cxx_compiler()
    info.cuda_compiler = _get_cuda_compiler()

    # Python information
    info.python_version = platform.python_version()
    info.python_executable = sys.executable

    return info


# Global cached platform info
_cached_platform_info: Optional[PlatformInfo] = None


def get_platform_info(cache: bool = True) -> PlatformInfo:
    """Get platform information with optional caching.

    Args:
        cache: Whether to cache the result for subsequent calls.

    Returns:
        PlatformInfo object.
    """
    global _cached_platform_info

    if cache and _cached_platform_info is not None:
        return _cached_platform_info

    info = detect_platform()

    if cache:
        _cached_platform_info = info

    return info


def clear_platform_cache():
    """Clear the cached platform information."""
    global _cached_platform_info
    _cached_platform_info = None


# Convenience functions
def is_linux() -> bool:
    """Check if running on Linux."""
    return get_platform_info().is_linux


def is_windows() -> bool:
    """Check if running on Windows."""
    return get_platform_info().is_windows


def is_darwin() -> bool:
    """Check if running on macOS."""
    return get_platform_info().is_darwin


def has_nvidia_gpu() -> bool:
    """Check if NVIDIA GPU is available."""
    return get_platform_info().gpu_vendor == GPUVendor.NVIDIA


def has_amd_gpu() -> bool:
    """Check if AMD GPU is available."""
    return get_platform_info().gpu_vendor == GPUVendor.AMD


def has_gpu() -> bool:
    """Check if any GPU is available."""
    return get_platform_info().has_gpu


def get_gpu_vendor() -> GPUVendor:
    """Get the GPU vendor type."""
    return get_platform_info().gpu_vendor


def get_cuda_version() -> Optional[str]:
    """Get CUDA version if available."""
    return get_platform_info().cuda_version


def get_rocm_version() -> Optional[str]:
    """Get ROCm version if available."""
    return get_platform_info().rocm_version
