# Copyright (c) OpenMMLab. All rights reserved.
"""``oci://`` model paths resolve to a local directory.

The scheme is explicit on purpose: a bare ``registry/name:tag`` is the same
shape as a HuggingFace repo id, so sniffing would hijack existing deployments.
These pin that, the output contract llmman promises, and that every other path
shape is untouched.
"""

import json
import os
import tempfile
from unittest import mock

import pytest

from lmdeploy import llmman
from lmdeploy.oci import get_oci_model, is_oci_ref, strip_scheme


class TestScheme:
    def test_recognizes_the_oci_scheme(self):
        assert is_oci_ref("oci://ghcr.io/org/model:tag")
        assert is_oci_ref("OCI://ghcr.io/org/model:tag")

    @pytest.mark.parametrize(
        "value",
        [
            "internlm/internlm2-chat-7b",
            "ghcr.io/org/model:tag",
            "/local/path/to/model",
            "s3://bucket/key",
            "http://0.0.0.0:23333",
            "",
            None,
        ],
    )
    def test_leaves_every_other_shape_alone(self, value):
        # A bare HF repo id must never be claimed.
        assert not is_oci_ref(value)

    def test_strips_the_scheme_only_when_present(self):
        assert strip_scheme("oci://ghcr.io/org/model:tag") == "ghcr.io/org/model:tag"
        assert strip_scheme("OCI://ghcr.io/org/model:tag") == "ghcr.io/org/model:tag"
        assert strip_scheme("internlm/internlm2-chat-7b") == "internlm/internlm2-chat-7b"


class TestResolveContract:
    """`llmman resolve --no-pull` reports where the daemon's pull landed."""

    def test_parses_the_documented_contract(self):
        with tempfile.TemporaryDirectory() as path:
            line = json.dumps({"reference": "ghcr.io/org/model:tag", "path": path, "format": "safetensors"})
            assert llmman.parse_resolve_output(line, "ref") == path

    def test_tolerates_trailing_newline_and_leaked_diagnostics(self):
        with tempfile.TemporaryDirectory() as path:
            out = "pulling blobs...\n" + json.dumps({"path": path}) + "\n"
            assert llmman.parse_resolve_output(out, "ref") == path

    def test_ignores_unknown_fields_so_the_contract_can_grow(self):
        with tempfile.TemporaryDirectory() as path:
            line = json.dumps({"path": path, "format": "gguf", "mmproj": "/x", "future": 1})
            assert llmman.parse_resolve_output(line, "ref") == path

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "   \n\n",
            "not json",
            '["a", "list"]',
            '{"no_path": 1}',
            '{"path": ""}',
            '{"path": 3}',
            '{"path": "/nonexistent/xyzzy"}',
        ],
    )
    def test_rejects_malformed_output(self, bad):
        with pytest.raises(RuntimeError):
            llmman.parse_resolve_output(bad, "ref")


class TestEndpoint:
    @pytest.mark.parametrize(
        "host,want",
        [
            ("", "http://127.0.0.1:17434"),
            ("1.2.3.4:9999", "http://1.2.3.4:9999"),
            ("1.2.3.4", "http://1.2.3.4:17434"),
            ("http://1.2.3.4:9999/ignored", "http://1.2.3.4:9999"),
            # A wildcard bind is meaningful to the server but not to a client.
            ("0.0.0.0:9999", "http://127.0.0.1:9999"),
            ("[::]:9999", "http://[::1]:9999"),
        ],
    )
    def test_parses_every_llmman_host_form(self, host, want):
        with mock.patch.dict(os.environ, {llmman.HOST_ENV: host}):
            assert llmman.endpoint() == want

    def test_binary_default_and_override(self):
        with mock.patch.dict(os.environ, {llmman.BIN_ENV: ""}):
            assert llmman.llmman_bin() == "llmman"
        with mock.patch.dict(os.environ, {llmman.BIN_ENV: "/opt/llmman"}):
            assert llmman.llmman_bin() == "/opt/llmman"


class TestGetOciModel:
    def test_rejects_an_empty_reference_without_touching_the_daemon(self):
        for ref in ("oci://", "oci://   "):
            with pytest.raises(ValueError):
                get_oci_model(ref)

    def test_strips_the_scheme_before_handing_off_to_llmman(self):
        with mock.patch("lmdeploy.oci.llmman.pull_and_resolve", return_value="/resolved") as acquire:
            assert get_oci_model("oci://ghcr.io/org/model:tag") == "/resolved"
        assert acquire.call_args[0][0] == "ghcr.io/org/model:tag"
        assert acquire.call_args[1]["progress"] is not None

    def test_reports_a_missing_binary(self):
        with mock.patch.dict(os.environ, {llmman.BIN_ENV: "/definitely/not/here"}):
            with pytest.raises(RuntimeError, match="not found"):
                llmman.resolve("ref")


class TestGetModelDispatch:
    """get_model routes oci:// away from the hub downloaders."""

    def test_oci_path_does_not_reach_snapshot_download(self):
        from lmdeploy.utils import get_model

        with mock.patch("lmdeploy.oci.get_oci_model", return_value="/resolved") as resolver:
            assert get_model("oci://ghcr.io/org/model:tag") == "/resolved"
        resolver.assert_called_once_with("oci://ghcr.io/org/model:tag")

    def test_hf_repo_id_still_uses_snapshot_download(self):
        from lmdeploy.utils import get_model

        with mock.patch("huggingface_hub.snapshot_download", return_value="/hf") as snap:
            assert get_model("internlm/internlm2-chat-7b") == "/hf"
        snap.assert_called_once()
