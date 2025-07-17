#!/usr/bin/env bash
set -eux

export PYTHON_VERSION=$PYTHON_VERSION
export PLAT_NAME=$PLAT_NAME
export USERID=${USERID}
export GROUPID=${GROUPID}
export NCCL_INCLUDE_DIR=/usr/local/cuda/include
export NCCL_LIB_DIR=/usr/local/cuda/lib64

source /opt/conda/bin/activate
conda activate $PYTHON_VERSION

cd lmdeploy
pip install build change-wheel-version
python -m build --wheel -o /tmpbuild/
for file in $(find /tmpbuild/ -name "*.whl")
do
    platform_tag="$(basename $file | cut -d- -f3-4)-${PLAT_NAME}"
    change_wheel_version /tmpbuild/*.whl --delete-old-wheel --platform-tag ${platform_tag}
done
chown ${USERID}:${GROUPID} /tmpbuild/*
mv /tmpbuild/* /lmdeploy_build/
