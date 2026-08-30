# Copyright (c) OpenMMLab. All rights reserved.
"""Resolve ``oci://`` model references to a local path.

A model published as a CNCF ModelPack (https://github.com/modelpack/model-spec)
artifact lives in an ordinary container registry, so it reuses the registry,
credentials, mirroring and air-gap tooling a deployment already has for
container images.

Acquisition is delegated to a running ``llmman serve``
(https://github.com/llmmanorg/llmman), which already implements the ModelPack
media types, registry auth, resumable blob download and a content-addressed
store. The daemon does the pull (POST /api/pull, streamed so a multi-gigabyte
fetch is not silent) but deliberately exposes no local path, so
``llmman resolve --no-pull`` reports where the bytes landed.

An explicit ``oci://`` scheme is required rather than sniffing a bare
``registry/name:tag``: that shape is indistinguishable from a HuggingFace repo
id (``org/model``), so guessing would silently hijack existing deployments.
"""

import logging

from lmdeploy import llmman

# Plain logging rather than lmdeploy.utils.get_logger: utils imports this
# module from get_model(), and this one is kept free of heavy imports.
logger = logging.getLogger("lmdeploy")

SCHEME = "oci://"


def is_oci_ref(model_path) -> bool:
    """Whether ``model_path`` carries the ``oci://`` scheme."""
    if not model_path:
        return False
    return str(model_path).lower().startswith(SCHEME)


def strip_scheme(model_path) -> str:
    """Drop the ``oci://`` prefix, leaving the bare registry reference."""
    text = str(model_path)
    if is_oci_ref(text):
        return text[len(SCHEME) :]
    return text


def get_oci_model(model_path: str) -> str:
    """Pull an ``oci://`` reference through llmman and return the local path."""
    reference = strip_scheme(model_path)
    if not reference.strip():
        raise ValueError(f"empty OCI model reference: {model_path!r}")

    def _progress(status, completed, total):
        if total:
            logger.info(f"llmman: {status} ({completed}/{total} bytes)")
        else:
            logger.info(f"llmman: {status}")

    return llmman.pull_and_resolve(reference.strip(), progress=_progress)
