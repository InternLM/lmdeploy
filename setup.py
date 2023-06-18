import os

from setuptools import find_packages, setup

pwd = os.path.dirname(__file__)
version_file = 'llmdeploy/version.py'


def readme():
    with open(os.path.join(pwd, 'README.md'), encoding='utf-8') as f:
        content = f.read()
    return content


def get_version():
    with open(os.path.join(pwd, version_file), 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


if __name__ == '__main__':
    setup(name='llmdeploy',
          version=get_version(),
          description='triton inference service of llama',
          long_description=readme(),
          long_description_content_type='text/markdown',
          author='OpenMMLab',
          author_email='openmmlab@gmail.com',
          packages=find_packages(
              exclude=('llmdeploy/serve/fastertransformer/triton_models', )),
          classifiers=[
              'Programming Language :: Python :: 3.8',
              'Programming Language :: Python :: 3.9',
              'Programming Language :: Python :: 3.10',
              'Intended Audience :: Developers',
              'Intended Audience :: Education',
              'Intended Audience :: Science/Research',
          ])
