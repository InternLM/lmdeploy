# Copyright (c) OpenMMLab. All rights reserved.
import os
import shutil
import subprocess

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPyWithRustBinary(build_py):
    binary_name = 'lmdeploy-router-bin'

    def run(self):
        super().run()
        if os.environ.get('LMDEPLOY_ROUTER_BUILD_NO_RUST') == '1':
            return
        package_name = self.distribution.get_name().replace('-', '_')
        data_dir = f'{package_name}-{self.distribution.get_version()}.data'
        bin_dir = os.path.join(self.build_lib, data_dir, 'scripts')
        os.makedirs(bin_dir, exist_ok=True)
        binary = os.environ.get(
            'LMDEPLOY_ROUTER_BIN', 'target/release/lmdeploy-router'
        )
        if not os.path.isfile(binary):
            subprocess.run(
                [
                    'cargo',
                    'build',
                    '--release',
                    '--locked',
                    '--bin',
                    'lmdeploy-router',
                ],
                check=True,
            )
        shutil.copy(binary, os.path.join(bin_dir, self.binary_name))
        os.chmod(os.path.join(bin_dir, self.binary_name), 0o755)


no_rust = os.environ.get('LMDEPLOY_ROUTER_BUILD_NO_RUST') == '1'

rust_extensions = []
if not no_rust:
    from setuptools_rust import Binding, RustExtension

    rust_extensions.append(
        RustExtension(
            target='lmdeploy_router_rs',
            path='Cargo.toml',
            binding=Binding.PyO3,
        )
    )


setup(
    cmdclass={
        'build_py': BuildPyWithRustBinary,
    },
    rust_extensions=rust_extensions,
    zip_safe=False,
)
