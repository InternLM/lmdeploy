"""Parametrize autotest cases by parallel layout with short pytest marks.

Layout marks mirror the parallel segment in ``get_case_str_by_config``
(``sorted(layout.items())``, no ``_``), e.g. ``tp1``, ``dp16ep16``,
``cp2tp8``, ``dp4ep8tp2``.

Distributed *runner* (ray / proxy) is a separate concern: keep
``@pytest.mark.distributed`` on the test function when the case needs
multi-node startup, even if the layout mark is the same (e.g. both local
and ray tp16 use layout mark ``tp16``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import pytest

from utils.config_utils import get_case_str_by_config, get_func_config_list

LOCAL_TP_LAYOUTS: tuple[dict[str, int], ...] = (
    {'tp': 1},
    {'tp': 2},
    {'tp': 4},
    {'tp': 8},
    {'tp': 16},
)

BASE_TP_LAYOUTS: tuple[dict[str, int], ...] = (
    {'tp': 1},
    {'tp': 2},
)

DISTRIBUTED_DP_EP_EQUAL_LAYOUTS: tuple[dict[str, int], ...] = (
    {'dp': 8, 'ep': 8},
    {'dp': 16, 'ep': 16},
    {'dp': 32, 'ep': 32},
)

# Backward-compatible alias.
DISTRIBUTED_DPEP_LAYOUTS = DISTRIBUTED_DP_EP_EQUAL_LAYOUTS

DISTRIBUTED_DP_EP_LAYOUTS: tuple[dict[str, int], ...] = (
    {'dp': 4, 'ep': 8},
)

DISTRIBUTED_TP_DP_EP_LAYOUTS: tuple[dict[str, int], ...] = (
    {'tp': 2, 'dp': 4, 'ep': 8},
)

DISTRIBUTED_CP_TP_LAYOUTS: tuple[dict[str, int], ...] = (
    {'cp': 2, 'tp': 8},
)

ALL_KNOWN_LAYOUTS: tuple[dict[str, int], ...] = (
    LOCAL_TP_LAYOUTS + BASE_TP_LAYOUTS + DISTRIBUTED_DP_EP_EQUAL_LAYOUTS +
    DISTRIBUTED_DP_EP_LAYOUTS + DISTRIBUTED_TP_DP_EP_LAYOUTS +
    DISTRIBUTED_CP_TP_LAYOUTS
)


def layout_mark_name(layout: dict[str, int]) -> str:
    """``{'tp': 1}`` → ``tp1``; ``{'dp': 16, 'ep': 16}`` → ``dp16ep16``.

    Same key order as ``get_case_str_by_config`` parallel segment
    (``sorted(layout.items())``), without underscore separators.
    """
    return ''.join(f'{key}{value}' for key, value in sorted(layout.items()))


def layout_mark(layout: dict[str, int]):
    """Return ``pytest.mark.<layout_mark_name(layout)>``."""
    return getattr(pytest.mark, layout_mark_name(layout))


def all_layout_mark_names() -> frozenset[str]:
    return frozenset(layout_mark_name(layout) for layout in ALL_KNOWN_LAYOUTS)


def build_multi_backend_layout_params(
    specs: Sequence[tuple[str, Sequence[dict[str, int]]]],
    *,
    model_type: str = 'chat_model',
    func_type: str = 'func',
    extra: dict | None = None,
    layout_extra_marks: Callable[[str, dict[str, int]], Iterable] | None = None,
    param_marks: Iterable = (),
    skip_empty_layouts: bool = True,
) -> list:
    """Build layout params for multiple backends; each row carries a backend mark."""
    rows: list = []
    for backend, layouts in specs:
        backend_mark = getattr(pytest.mark, backend)
        for layout in layouts:
            configs = get_func_config_list(
                backend, layout, model_type, func_type, extra,
            )
            if skip_empty_layouts and not configs:
                continue
            marks = [
                layout_mark(layout),
                backend_mark,
                *param_marks,
            ]
            if layout_extra_marks is not None:
                marks.extend(layout_extra_marks(backend, layout))
            for run_config in configs:
                rows.append(
                    pytest.param(
                        run_config,
                        marks=marks,
                        id=get_case_str_by_config(run_config),
                    ))
    return rows


def build_layout_params(
    backend: str,
    layouts: Sequence[dict[str, int]],
    *,
    model_type: str = 'chat_model',
    func_type: str = 'func',
    extra: dict | None = None,
    layout_extra_marks: Callable[[dict[str, int]], Iterable] | None = None,
    param_marks: Iterable = (),
    skip_empty_layouts: bool = True,
) -> list:
    """Build ``pytest.param`` rows; each param carries a layout mark."""
    rows: list = []
    for layout in layouts:
        configs = get_func_config_list(backend, layout, model_type, func_type, extra)
        if skip_empty_layouts and not configs:
            continue
        marks = [layout_mark(layout), *param_marks]
        if layout_extra_marks is not None:
            marks.extend(layout_extra_marks(layout))
        for run_config in configs:
            rows.append(
                pytest.param(
                    run_config,
                    marks=marks,
                    id=get_case_str_by_config(run_config),
                ))
    return rows


def build_eval_stage_params(
    backend: str,
    layouts: Sequence[dict[str, int]],
    *,
    test_types: Sequence[str] = ('infer', 'eval'),
    model_type: str = 'chat_model',
    func_type: str = 'evaluate',
    extra: dict | None = None,
    layout_extra_marks: Callable[[dict[str, int]], Iterable] | None = None,
    param_marks: Iterable = (),
    skip_empty_layouts: bool = True,
) -> list:
    """Build ``(test_type, run_config)`` params with layout / infer|eval / backend marks."""
    rows: list = []
    backend_mark = getattr(pytest.mark, backend)
    for test_type in test_types:
        stage_mark = pytest.mark.infer if test_type == 'infer' else pytest.mark.eval
        for layout in layouts:
            configs = get_func_config_list(
                backend, layout, model_type, func_type, extra=extra,
            )
            if skip_empty_layouts and not configs:
                continue
            marks = [
                layout_mark(layout),
                stage_mark,
                backend_mark,
                pytest.mark.flaky(reruns=0),
                *param_marks,
            ]
            if layout_extra_marks is not None:
                marks.extend(layout_extra_marks(layout))
            for run_config in configs:
                case_id = get_case_str_by_config(run_config)
                rows.append(
                    pytest.param(
                        test_type,
                        run_config,
                        marks=marks,
                        id=f'{test_type}-{case_id}',
                    ))
    return rows


def build_eval_longtext_params(
    backend: str,
    layout: dict[str, int],
    *,
    session_len: int,
    eval_config_name: str,
    eval_subpath: str,
    test_types: Sequence[str] = ('infer', 'eval'),
    use_proxy: bool = False,
) -> list:
    """Longtext evaluate rows with extra path/config metadata on each param."""
    rows: list = []
    backend_mark = getattr(pytest.mark, backend)
    extra = {'session_len': session_len}
    configs = get_func_config_list(
        backend, layout, func_type='longtext_evaluate', extra=extra,
    )
    for test_type in test_types:
        stage_mark = pytest.mark.infer if test_type == 'infer' else pytest.mark.eval
        marks = [
            layout_mark(layout),
            stage_mark,
            backend_mark,
            pytest.mark.flaky(reruns=0),
        ]
        if use_proxy:
            marks.append(pytest.mark.distributed)
        for run_config in configs:
            case_id = get_case_str_by_config(run_config)
            rows.append(
                pytest.param(
                    test_type,
                    run_config,
                    eval_config_name,
                    eval_subpath,
                    marks=marks,
                    id=f'{test_type}-{eval_config_name}-{case_id}',
                ))
    return rows
