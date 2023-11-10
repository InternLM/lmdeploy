# Copyright (c) OpenMMLab. All rights reserved.
from huggingface_hub import snapshot_download
from typing import Optional, Union
from pathlib import Path


def download_hf_repo(
    repo_id: str,
    *,
    revision: Optional[str] = None,
    cache_dir: Union[str, Path, None] = None,
    proxies: Optional[dict] = None,
    resume_download: bool = True,
    force_download: bool = False,
    token: Optional[Union[bool, str]] = None,
    local_files_only: bool = False,
) -> str:
    """Download huggingface repo"""

    download_kwargs = {
        "revision": revision,
        "cache_dir": cache_dir,
        "proxies": proxies,
        "resume_download": resume_download,
        "force_download": force_download,
        "token": token,
        "local_files_only": local_files_only
    }

    downloaded_folder = snapshot_download(repo_id, **download_kwargs)
    return downloaded_folder
