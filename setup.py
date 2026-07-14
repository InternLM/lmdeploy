import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from setuptools import find_packages, setup

pwd = os.path.dirname(__file__)
version_file = 'lmdeploy/version.py'


def get_target_device() -> str:
    return os.getenv('LMDEPLOY_TARGET_DEVICE', 'cuda')


def readme() -> str:
    with open(os.path.join(pwd, 'README.md'), encoding='utf-8') as f:
        return f.read()


def get_version() -> str:
    file_path = os.path.join(pwd, version_file)
    pattern = re.compile(r"\s*__version__\s*=\s*'([0-9A-Za-z.-]+)'")
    with open(file_path) as f:
        for line in f:
            match = pattern.match(line)
            if match:
                return match.group(1)
        raise RuntimeError(f'No version found in {file_path}')


def get_turbomind_deps() -> List[str]:
    if os.name == 'nt':
        return []

    cuda_compiler = os.getenv('CUDACXX', os.getenv('CMAKE_CUDA_COMPILER', 'nvcc'))
    try:
        nvcc_output = subprocess.check_output(
            [cuda_compiler, '--version'],
            stderr=subprocess.DEVNULL,
            text=True
        )
        match = re.search(r'release\s+(\d+)', nvcc_output)
        if not match:
            return []
        cuda_version = int(match.group(1))

        if cuda_version >= 13:
            return [
                f'nvidia-nccl-cu{cuda_version}',
                'nvidia-cuda-runtime',
                'nvidia-cublas',
                'nvidia-curand',
            ]
        else:
            return [
                f'nvidia-nccl-cu{cuda_version}',
                f'nvidia-cuda-runtime-cu{cuda_version}',
                f'nvidia-cublas-cu{cuda_version}',
                f'nvidia-curand-cu{cuda_version}',
            ]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def parse_requirements(fname: str, with_version: bool = True) -> List[str]:
    require_fpath = os.path.join(pwd, fname)
    if not os.path.exists(require_fpath):
        return []

    requirements: List[str] = []
    with open(require_fpath, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if line.startswith('-r '):
                target = line.split(' ', 1)[1]
                if not os.path.isabs(target):
                    target = os.path.join(os.path.dirname(require_fpath), target)
                requirements.extend(parse_requirements(target, with_version))
                continue

            requirements.append(line)

    return requirements


if get_target_device() == 'cuda' and os.getenv('DISABLE_TURBOMIND', '').lower() not in ('yes', 'true', 'on', 't', '1'):
    import cmake_build_extension

    ext_modules = [
        cmake_build_extension.CMakeExtension(
            name='_turbomind',
            install_prefix='lmdeploy/lib',
            cmake_depends_on=['pybind11'],
            source_dir=str(Path(__file__).parent.absolute()),
            cmake_generator=None if os.name == 'nt' else 'Ninja',
            cmake_build_type=os.getenv('CMAKE_BUILD_TYPE', 'Release'),
            cmake_configure_options=[
                f'-DPython3_ROOT_DIR={Path(sys.prefix)}',
                f'-DPYTHON_EXECUTABLE={Path(sys.executable)}',
                '-DCALL_FROM_SETUP_PY:BOOL=ON',
                '-DBUILD_SHARED_LIBS:BOOL=OFF',
                # Select the bindings implementation
                '-DBUILD_PY_FFI=ON',
                '-DBUILD_MULTI_GPU=' + ('OFF' if os.name == 'nt' else 'ON'),
                '-DUSE_NVTX=' + ('OFF' if os.name == 'nt' else 'ON'),
            ],
        ),
    ]
    extra_deps = get_turbomind_deps()
    cmdclass = dict(build_ext=cmake_build_extension.BuildExtension, )
else:
    ext_modules = []
    cmdclass = {}
    extra_deps = []

if __name__ == '__main__':
    setup(
        name='lmdeploy',
        version=get_version(),
        description='A toolset for compressing, deploying and serving LLM',
        long_description=readme(),
        long_description_content_type='text/markdown',
        author='OpenMMLab',
        author_email='openmmlab@gmail.com',
        url='https://github.com/ghshhf/lmdeploy',
        project_urls={
            'Documentation': 'https://lmdeploy.readthedocs.io',
            'Source': 'https://github.com/ghshhf/lmdeploy',
            'Tracker': 'https://github.com/ghshhf/lmdeploy/issues',
        },
        packages=find_packages(exclude=[]),
        include_package_data=True,
        setup_requires=parse_requirements('requirements/build.txt'),
        tests_require=parse_requirements('requirements/test.txt'),
        install_requires=parse_requirements(f'requirements/runtime_{get_target_device()}.txt') + extra_deps,
        extras_require={
            'all': parse_requirements(f'requirements_{get_target_device()}.txt'),
            'lite': parse_requirements('requirements/lite.txt'),
            'serve': parse_requirements('requirements/serve.txt'),
            'docs': parse_requirements('requirements/docs.txt'),
            'dev': parse_requirements('requirements/test.txt'),
        },
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Programming Language :: Python :: 3.12',
            'Programming Language :: Python :: 3.13',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: POSIX :: Linux',
            'Operating System :: Microsoft :: Windows',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        keywords='llm, inference, deployment, quantization, serving',
        entry_points={'console_scripts': ['lmdeploy = lmdeploy.cli:run']},
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        python_requires='>=3.10',
    )
