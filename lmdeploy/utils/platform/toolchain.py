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
"""Toolchain detection and configuration module.

This module provides toolchain detection for different platforms and accelerators:
- GCC/Clang for native compilation
- NVCC for CUDA
- HIP for AMD ROCm
- CANN for Ascend
- CNTK for Cambricon
"""

import os
import re
import subprocess
import shutil
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any


class CompilerType(Enum):
    """Compiler types."""
    GCC = "gcc"
    CLANG = "clang"
    MSVC = "msvc"
    ICC = "icc"
    NVCC = "nvcc"
    HIPCC = "hipcc"
    CANN = "cann"
    CNCC = "cncc"
    UNKNOWN = "unknown"


@dataclass
class CompilerInfo:
    """Compiler information."""
    type: CompilerType = CompilerType.UNKNOWN
    name: str = ""
    version: str = ""
    path: str = ""
    family: str = ""  # gcc, clang, msvc, etc.
    supports_cpp17: bool = False
    supports_cpp20: bool = False


@dataclass
class ToolchainInfo:
    """Complete toolchain information."""
    # C/C++ compiler
    c_compiler: CompilerInfo = field(default_factory=CompilerInfo)
    cxx_compiler: CompilerInfo = field(default_factory=CompilerInfo)

    # GPU compilers
    cuda_compiler: Optional[CompilerInfo] = None
    hip_compiler: Optional[CompilerInfo] = None

    # SDK paths
    cuda_home: str = ""
    rocm_home: str = ""
    ascend_home: str = ""
    cambricon_home: str = ""

    # Libraries
    cuda_libs: List[str] = field(default_factory=list)
    rocm_libs: List[str] = field(default_factory=list)

    # Build flags
    cxx_flags: List[str] = field(default_factory=list)
    c_flags: List[str] = field(default_factory=list)
    ld_flags: List[str] = field(default_factory=list)

    # Additional paths
    include_paths: List[str] = field(default_factory=list)
    library_paths: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'c_compiler': {
                'type': self.c_compiler.type.value,
                'name': self.c_compiler.name,
                'version': self.c_compiler.version,
                'path': self.c_compiler.path,
            },
            'cxx_compiler': {
                'type': self.cxx_compiler.type.value,
                'name': self.cxx_compiler.name,
                'version': self.cxx_compiler.version,
                'path': self.cxx_compiler.path,
            },
            'cuda_home': self.cuda_home,
            'rocm_home': self.rocm_home,
            'ascend_home': self.ascend_home,
            'cambricon_home': self.cambricon_home,
        }

        if self.cuda_compiler:
            result['cuda_compiler'] = {
                'type': self.cuda_compiler.type.value,
                'version': self.cuda_compiler.version,
                'path': self.cuda_compiler.path,
            }

        return result

    def get_cmake_defines(self) -> List[str]:
        """Get CMake definitions for this toolchain."""
        defines = []

        if self.cuda_home:
            defines.append(f"CUDA_TOOLKIT_ROOT_DIR={self.cuda_home}")

        if self.cxx_compiler.path:
            defines.append(f"CMAKE_CXX_COMPILER={self.cxx_compiler.path}")

        if self.c_compiler.path:
            defines.append(f"CMAKE_C_COMPILER={self.c_compiler.path}")

        for path in self.library_paths:
            defines.append(f"-DCMAKE_LIBRARY_PATH={path}")

        for path in self.include_paths:
            defines.append(f"-DCMAKE_INCLUDE_PATH={path}")

        return defines


def _get_gcc_info(compiler_path: str) -> Optional[CompilerInfo]:
    """Get GCC compiler information."""
    try:
        result = subprocess.run(
            [compiler_path, "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            output = result.stdout + result.stderr

            # Extract version
            version_match = re.search(r'(\d+)\.(\d+)\.(\d+)', output)
            version = version_match.group(0) if version_match else ""

            # Determine family
            if "clang" in output.lower():
                family = "clang"
            elif "g++" in compiler_path or "gcc" in compiler_path:
                family = "gcc"
            elif "icpc" in compiler_path or "icc" in compiler_path:
                family = "icc"
            else:
                family = "unknown"

            # Check C++ standard support
            supports_cpp17 = False
            supports_cpp20 = False

            # Try to compile test programs
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
                f.write("int main() { return 0; }")
                test_file = f.name

            try:
                # Check C++17
                result = subprocess.run(
                    [compiler_path, "-std=c++17", "-c", test_file, "-o", "/dev/null"],
                    capture_output=True,
                    timeout=10
                )
                supports_cpp17 = result.returncode == 0

                # Check C++20
                result = subprocess.run(
                    [compiler_path, "-std=c++20", "-c", test_file, "-o", "/dev/null"],
                    capture_output=True,
                    timeout=10
                )
                supports_cpp20 = result.returncode == 0
            except Exception:
                pass
            finally:
                try:
                    os.unlink(test_file)
                except Exception:
                    pass

            return CompilerInfo(
                type=CompilerType.CLANG if family == "clang" else CompilerType.GCC,
                name=os.path.basename(compiler_path),
                version=version,
                path=compiler_path,
                family=family,
                supports_cpp17=supports_cpp17,
                supports_cpp20=supports_cpp20,
            )

    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass

    return None


def _get_msvc_info() -> Optional[CompilerInfo]:
    """Get MSVC compiler information."""
    try:
        # Try to get MSVC version from environment
        vs_install_dir = os.environ.get('VSINSTALLDIR')
        if vs_install_dir:
            # Parse Visual Studio version
            version_match = re.search(r'Visual Studio (\d+)', vs_install_dir)
            if version_match:
                return CompilerInfo(
                    type=CompilerType.MSVC,
                    name="msvc",
                    version=version_match.group(1),
                    path="cl.exe",
                    family="msvc",
                )

        # Try cl.exe
        result = subprocess.run(
            ["cl"],
            capture_output=True,
            timeout=5
        )

        if "Copyright" in str(result.stderr) or result.returncode != 0:
            return CompilerInfo(
                type=CompilerType.MSVC,
                name="msvc",
                version="unknown",
                path="cl.exe",
                family="msvc",
            )

    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass

    return None


def _get_nvcc_info() -> Optional[CompilerInfo]:
    """Get NVCC compiler information."""
    # Find nvcc
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    nvcc_path = os.path.join(cuda_home or '', 'bin', 'nvcc')

    if not os.path.exists(nvcc_path):
        # Try to find in PATH
        nvcc_path = shutil.which('nvcc')

    if not nvcc_path:
        return None

    try:
        result = subprocess.run(
            [nvcc_path, "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            # Extract version
            version_match = re.search(r'release (\d+\.\d+)', result.stdout)
            version = version_match.group(1) if version_match else "unknown"

            return CompilerInfo(
                type=CompilerType.NVCC,
                name="nvcc",
                version=version,
                path=nvcc_path,
                family="cuda",
            )

    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass

    return None


def _get_hipcc_info() -> Optional[CompilerInfo]:
    """Get HIP compiler information."""
    hipcc_path = shutil.which('hipcc')

    if not hipcc_path:
        rocm_home = os.environ.get('ROCM_PATH')
        if rocm_home:
            hipcc_path = os.path.join(rocm_home, 'bin', 'hipcc')
            if not os.path.exists(hipcc_path):
                hipcc_path = None

    if not hipcc_path:
        return None

    try:
        result = subprocess.run(
            [hipcc_path, "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            # Extract version
            version_match = re.search(r'HIP version: (\d+\.\d+)', result.stdout)
            version = version_match.group(1) if version_match else "unknown"

            return CompilerInfo(
                type=CompilerType.HIPCC,
                name="hipcc",
                version=version,
                path=hipcc_path,
                family="hip",
            )

    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass

    return None


def detect_c_compiler() -> CompilerInfo:
    """Detect C compiler."""
    # Check common compilers
    compilers = ["gcc", "cc", "clang", "icc"]

    for compiler in compilers:
        compiler_path = shutil.which(compiler)
        if compiler_path:
            info = _get_gcc_info(compiler_path)
            if info:
                return info

    # Check Windows
    if os.name == "nt":
        info = _get_msvc_info()
        if info:
            return info

    return CompilerInfo()


def detect_cxx_compiler() -> CompilerInfo:
    """Detect C++ compiler."""
    # Check common compilers
    compilers = ["g++", "c++", "clang++", "icpc"]

    for compiler in compilers:
        compiler_path = shutil.which(compiler)
        if compiler_path:
            info = _get_gcc_info(compiler_path)
            if info:
                return info

    # Check Windows
    if os.name == "nt":
        info = _get_msvc_info()
        if info:
            return info

    return CompilerInfo()


def detect_cuda_compiler() -> Optional[CompilerInfo]:
    """Detect CUDA compiler (nvcc)."""
    return _get_nvcc_info()


def detect_hip_compiler() -> Optional[CompilerInfo]:
    """Detect HIP compiler (hipcc)."""
    return _get_hipcc_info()


def _get_cuda_paths() -> Dict[str, List[str]]:
    """Get CUDA include and library paths."""
    paths = {
        'include': [],
        'library': [],
    }

    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if not cuda_home:
        return paths

    # Common paths
    include_path = os.path.join(cuda_home, 'include')
    lib_path = os.path.join(cuda_home, 'lib64')
    lib_path_alt = os.path.join(cuda_home, 'lib')

    if os.path.exists(include_path):
        paths['include'].append(include_path)

    if os.path.exists(lib_path):
        paths['library'].append(lib_path)

    if os.path.exists(lib_path_alt):
        paths['library'].append(lib_path_alt)

    return paths


def _get_rocm_paths() -> Dict[str, List[str]]:
    """Get ROCm include and library paths."""
    paths = {
        'include': [],
        'library': [],
    }

    rocm_home = os.environ.get('ROCM_PATH')
    if not rocm_home:
        return paths

    # Common paths
    include_path = os.path.join(rocm_home, 'include')
    lib_path = os.path.join(rocm_home, 'lib')
    lib64_path = os.path.join(rocm_home, 'lib64')

    if os.path.exists(include_path):
        paths['include'].append(include_path)

    if os.path.exists(lib_path):
        paths['library'].append(lib_path)

    if os.path.exists(lib64_path):
        paths['library'].append(lib64_path)

    return paths


def detect_toolchain() -> ToolchainInfo:
    """Detect complete toolchain information.

    Returns:
        ToolchainInfo object containing all toolchain details.
    """
    info = ToolchainInfo()

    # Detect compilers
    info.c_compiler = detect_c_compiler()
    info.cxx_compiler = detect_cxx_compiler()
    info.cuda_compiler = detect_cuda_compiler()
    info.hip_compiler = detect_hip_compiler()

    # Get SDK paths
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    info.cuda_home = cuda_home or ""

    rocm_home = os.environ.get('ROCM_PATH')
    info.rocm_home = rocm_home or ""

    ascend_home = os.environ.get('ASCEND_HOME_PATH')
    info.ascend_home = ascend_home or ""

    cambricon_home = os.environ.get('CNNL_HOME')
    info.cambricon_home = cambricon_home or ""

    # Get include and library paths
    cuda_paths = _get_cuda_paths()
    info.include_paths.extend(cuda_paths.get('include', []))
    info.library_paths.extend(cuda_paths.get('library', []))

    rocm_paths = _get_rocm_paths()
    info.include_paths.extend(rocm_paths.get('include', []))
    info.library_paths.extend(rocm_paths.get('library', []))

    # Build standard flags
    if info.cxx_compiler.supports_cpp17:
        info.cxx_flags.append("-std=c++17")
    elif info.cxx_compiler.supports_cpp20:
        info.cxx_flags.append("-std=c++20")

    # Add optimization flags
    info.cxx_flags.extend(["-O3", "-fPIC"])

    # Add CUDA flags if available
    if info.cuda_compiler and cuda_home:
        info.cxx_flags.append(f"-I{os.path.join(cuda_home, 'include')}")
        info.ld_flags.append(f"-L{os.path.join(cuda_home, 'lib64')}")

    return info


# Global cached toolchain info
_cached_toolchain_info: Optional[ToolchainInfo] = None


def get_toolchain_info(cache: bool = True) -> ToolchainInfo:
    """Get toolchain information with optional caching.

    Args:
        cache: Whether to cache the result for subsequent calls.

    Returns:
        ToolchainInfo object.
    """
    global _cached_toolchain_info

    if cache and _cached_toolchain_info is not None:
        return _cached_toolchain_info

    info = detect_toolchain()

    if cache:
        _cached_toolchain_info = info

    return info


def clear_toolchain_cache():
    """Clear the cached toolchain information."""
    global _cached_toolchain_info
    _cached_toolchain_info = None
