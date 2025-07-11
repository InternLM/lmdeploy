import os
import re
import sys
from pathlib import Path

from setuptools import find_packages, setup

pwd = os.path.dirname(__file__)
version_file = 'lmdeploy/version.py'


def get_target_device():
    return os.getenv('LMDEPLOY_TARGET_DEVICE', 'cuda')


def readme():
    with open(os.path.join(pwd, 'README.md'), encoding='utf-8') as f:
        content = f.read()
    return content


def get_version():
    file_path = os.path.join(pwd, version_file)
    pattern = re.compile(r"\s*__version__\s*=\s*'(\d+\.\d+\.\d+)'")
    with open(file_path, 'r') as f:
        for line in f:
            m = pattern.match(line)
            if m:
                return m.group(1)
        else:
            assert False, f'No version found {file_path}'


def get_turbomind_deps():
    if os.name == 'nt':
        return []

    CUDAVER = os.getenv('CUDAVER', '11')
    return [
        f'nvidia-nccl-cu{CUDAVER}',
        f'nvidia-cuda-runtime-cu{CUDAVER}',
        f'nvidia-cublas-cu{CUDAVER}',
        f'nvidia-curand-cu{CUDAVER}',
    ]


def parse_requirements(fname='requirements.txt', with_version=True):
    """Parse the package dependencies listed in a file but strips specific
    versioning information.

    Args:
        fname (str): path to the file
        with_version (bool, default=False): if True include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            elif '@git+' in line:
                info['package'] = line
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip, rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if os.path.exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())

    return packages


if get_target_device() == 'cuda' and not os.getenv('DISABLE_TURBOMIND', '').lower() in ('yes', 'true', 'on', 't', '1'):
    import cmake_build_extension

    ext_modules = [
        cmake_build_extension.CMakeExtension(
            name='_turbomind',
            install_prefix='lmdeploy/lib',
            cmake_depends_on=['pybind11'],
            source_dir=str(Path(__file__).parent.absolute()),
            cmake_generator=None if os.name == 'nt' else 'Ninja',
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
    lmdeploy_package_data = ['lmdeploy/bin/llama_gemm']
    setup(
        name='lmdeploy',
        version=get_version(),
        description='A toolset for compressing, deploying and serving LLM',
        long_description=readme(),
        long_description_content_type='text/markdown',
        author='OpenMMLab',
        author_email='openmmlab@gmail.com',
        packages=find_packages(exclude=()),
        package_data={
            'lmdeploy': lmdeploy_package_data,
        },
        include_package_data=True,
        setup_requires=parse_requirements('requirements/build.txt'),
        tests_require=parse_requirements('requirements/test.txt'),
        install_requires=parse_requirements(f'requirements/runtime_{get_target_device()}.txt') + extra_deps,
        extras_require={
            'all': parse_requirements(f'requirements_{get_target_device()}.txt'),
            'lite': parse_requirements('requirements/lite.txt'),
            'serve': parse_requirements('requirements/serve.txt'),
        },
        classifiers=[
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Programming Language :: Python :: 3.12',
            'Programming Language :: Python :: 3.13',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
        ],
        entry_points={'console_scripts': ['lmdeploy = lmdeploy.cli:run']},
        ext_modules=ext_modules,
        cmdclass=cmdclass,
    )
