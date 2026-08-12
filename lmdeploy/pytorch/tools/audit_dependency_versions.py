# Copyright (c) OpenMMLab. All rights reserved.
"""Inventory dependency constraints and source-level version checks.

This is a read-only maintenance tool. It does not decide whether a dependency
is compatible; it points maintainers to the requirement, environment, feature,
model, and API-policy locations that should be reviewed together.

Examples:

    python lmdeploy/pytorch/tools/audit_dependency_versions.py
    python lmdeploy/pytorch/tools/audit_dependency_versions.py triton
    python lmdeploy/pytorch/tools/audit_dependency_versions.py transformers
"""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

_VERSION_LITERAL = re.compile(r'^v?\d+(?:\.\d+)+(?:[-+._a-zA-Z0-9]*)?$')
_VERSION_SPEC = re.compile(r'(?:===|==|~=|!=|<=|>=|<|>)\s*v?\d+(?:\.\d+)+(?:\.\*)?')
_VERSION_PARSERS = frozenset({'parse', 'parse_version', 'Version', 'SpecifierSet'})


@dataclass(frozen=True)
class RequirementReference:
    """One install constraint declared by a requirements file."""

    package: str
    requirement: str
    path: Path
    line: int
    marker_applies: bool


@dataclass(frozen=True)
class SourceReference:
    """One source-level version policy or API threshold."""

    package: str
    value: str
    symbol: str | None
    role: str
    path: Path
    line: int


@dataclass(frozen=True)
class _VersionCandidate:
    value: str
    symbol: str | None
    line: int


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _iter_requirement_lines(path: Path, visited: set[Path] | None = None):
    if visited is None:
        visited = set()
    path = path.resolve()
    if path in visited:
        return
    visited.add(path)

    for line_number, raw_line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        line = raw_line.split('#', 1)[0].strip()
        if not line:
            continue
        if line.startswith(('-r ', '--requirement ')):
            include = line.split(maxsplit=1)[1]
            yield from _iter_requirement_lines(path.parent / include, visited)
            continue
        if line.startswith('-e '):
            continue
        yield path, line_number, line


def collect_requirement_references(root: Path) -> list[RequirementReference]:
    """Collect the CUDA runtime requirements without importing setup.py."""
    references = []
    requirements_path = root / 'requirements/runtime_cuda.txt'
    for path, line_number, line in _iter_requirement_lines(requirements_path):
        try:
            requirement = Requirement(line)
        except InvalidRequirement:
            continue
        references.append(
            RequirementReference(
                package=canonicalize_name(requirement.name),
                requirement=str(requirement),
                path=path,
                line=line_number,
                marker_applies=requirement.marker is None or requirement.marker.evaluate(),
            ))
    return references


def _call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _target_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, (ast.Tuple, ast.List)):
        names = [name for item in node.elts if (name := _target_name(item))]
        return ','.join(names) or None
    return None


def _looks_like_version(value: str) -> bool:
    return bool(_VERSION_LITERAL.fullmatch(value.strip()) or _VERSION_SPEC.search(value))


def _string_values(node: ast.AST) -> Iterable[ast.Constant]:
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str) and _looks_like_version(child.value):
            yield child


def _import_aliases(tree: ast.AST) -> dict[str, str]:
    aliases = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Import):
            continue
        for imported in node.names:
            aliases[imported.asname or imported.name.split('.')[0]] = canonicalize_name(imported.name.split('.')[0])
    return aliases


def _versioned_packages(tree: ast.AST, aliases: dict[str, str]) -> set[str]:
    packages = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or node.attr != '__version__':
            continue
        if isinstance(node.value, ast.Name):
            packages.add(aliases.get(node.value.id, canonicalize_name(node.value.id)))
    return packages


def _collect_candidates(tree: ast.AST) -> list[_VersionCandidate]:
    candidates = []
    assigned_values = set()
    for node in ast.walk(tree):
        symbol = None
        value_node = None
        if isinstance(node, ast.Assign):
            symbol = ','.join(filter(None, (_target_name(target) for target in node.targets))) or None
            is_parser_call = isinstance(node.value, ast.Call) and _call_name(node.value.func) in _VERSION_PARSERS
            if symbol and ('version' in symbol.lower() or is_parser_call):
                value_node = node.value
        elif isinstance(node, ast.AnnAssign):
            symbol = _target_name(node.target)
            is_parser_call = isinstance(node.value, ast.Call) and _call_name(node.value.func) in _VERSION_PARSERS
            if symbol and ('version' in symbol.lower() or is_parser_call):
                value_node = node.value

        if value_node is None:
            continue
        for value in _string_values(value_node):
            candidates.append(_VersionCandidate(value=value.value, symbol=symbol, line=value.lineno))
            assigned_values.add(id(value))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node.func) not in _VERSION_PARSERS:
            continue
        for value in _string_values(node):
            if id(value) not in assigned_values:
                candidates.append(_VersionCandidate(value=value.value, symbol=None, line=value.lineno))

    return list(dict.fromkeys(candidates))


def _candidate_packages(candidate: _VersionCandidate, source: str, versioned_packages: set[str]) -> set[str]:
    if len(versioned_packages) == 1:
        return versioned_packages

    context = source.splitlines()[candidate.line - 1].lower()
    if candidate.symbol:
        context += ' ' + candidate.symbol.lower()
    return {package for package in versioned_packages if package.replace('-', '_') in context}


def _classify_source(path: Path, symbol: str | None) -> str:
    path_text = path.as_posix()
    symbol_text = (symbol or '').lower()
    if '/check_env/' in path_text:
        return 'environment warning'
    if '/models/' in path_text:
        return 'model compatibility'
    if '/backends/' in path_text and any(token in symbol_text for token in ('min', 'max', 'spec', 'required')):
        return 'feature compatibility'
    if '/kernels/' in path_text:
        return 'API threshold'
    return 'version branch'


def collect_source_references(root: Path) -> list[SourceReference]:
    """Find explicit version policies and branches in PyTorch source."""
    references = []
    source_root = root / 'lmdeploy/pytorch'
    for path in sorted(source_root.rglob('*.py')):
        if path == Path(__file__).resolve():
            continue
        source = path.read_text(encoding='utf-8')
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            continue
        versioned_packages = _versioned_packages(tree, _import_aliases(tree))
        if not versioned_packages:
            continue
        for candidate in _collect_candidates(tree):
            for package in _candidate_packages(candidate, source, versioned_packages):
                references.append(
                    SourceReference(
                        package=package,
                        value=candidate.value,
                        symbol=candidate.symbol,
                        role=_classify_source(path, candidate.symbol),
                        path=path,
                        line=candidate.line,
                    ))
    return references


def _installed_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _requirement_status(reference: RequirementReference, installed_version: str | None) -> str:
    if not reference.marker_applies:
        return 'marker does not apply on this platform'
    if installed_version is None:
        return 'package not installed'
    requirement = Requirement(reference.requirement)
    if requirement.specifier.contains(installed_version, prereleases=True):
        return 'installed version matches'
    return 'REVIEW: CUDA runtime requirement excludes the installed version'


def _review_action(role: str) -> str:
    actions = {
        'environment warning': 'update after engine-wide compatibility review',
        'feature compatibility': 'update after this feature passes correctness and performance validation',
        'model compatibility': 'review only for the affected model family',
        'API threshold': 'usually retain; inspect only if the dependency API branch changed',
        'version branch': 'inspect the owning call path before changing',
    }
    return actions[role]


def print_package_report(package: str,
                         requirements: list[RequirementReference],
                         source_references: list[SourceReference],
                         root: Path,
                         output: TextIO = sys.stdout):
    """Print an actionable review checklist for one dependency."""
    package = canonicalize_name(package)
    installed_version = _installed_version(package)
    package_requirements = [item for item in requirements if item.package == package]
    package_sources = [item for item in source_references if item.package == package]

    print(f'\n{package}', file=output)
    print(f'  installed: {installed_version or "not installed"}', file=output)

    print('  install constraints:', file=output)
    if not package_requirements:
        print('    (none found)', file=output)
    for item in package_requirements:
        location = f'{_relative(item.path, root)}:{item.line}'
        status = _requirement_status(item, installed_version)
        print(f'    - {item.requirement} [{status}]', file=output)
        print(f'      {location}', file=output)

    print('  source version references:', file=output)
    if not package_sources:
        print('    (none found)', file=output)
    grouped_sources = {}
    for item in package_sources:
        grouped_sources.setdefault((item.path, item.role), []).append(item)
    for (path, role), items in sorted(grouped_sources.items(), key=lambda pair: (str(pair[0][0]), pair[0][1])):
        sorted_items = sorted(items, key=lambda item: item.line)
        values = ', '.join(f'{item.symbol}={item.value}' if item.symbol else item.value for item in sorted_items)
        lines = ','.join(str(item.line) for item in sorted_items)
        print(f'    - {role}: {values}', file=output)
        print(f'      {_relative(path, root)}:{lines}', file=output)
        print(f'      action: {_review_action(role)}', file=output)

    locations = sorted({f'{_relative(item.path, root)}:{item.line}' for item in package_requirements}
                       | {f'{_relative(item.path, root)}:{item.line}' for item in package_sources})
    print('  review checklist:', file=output)
    if not locations:
        print('    - No requirement or explicit source version reference was found.', file=output)
    else:
        print(f'    - Review all {len(locations)} locations listed above when changing {package}.', file=output)
        print('    - Do not update API thresholds merely to match the installation version.', file=output)
        print('    - Run the owner-specific correctness and performance tests before widening a tested range.',
              file=output)


def _parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description='Inventory dependency constraints and source-level version checks.',
        epilog=(
            'examples:\n'
            '  python lmdeploy/pytorch/tools/audit_dependency_versions.py\n'
            '  python lmdeploy/pytorch/tools/audit_dependency_versions.py triton\n'
            '  python lmdeploy/pytorch/tools/audit_dependency_versions.py transformers'),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'packages',
        nargs='*',
        help='dependencies to audit (default: CUDA dependencies with PyTorch source version references)',
    )
    parser.add_argument('--root', type=Path, default=_repository_root(), help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = _parse_args(argv)
    root = args.root.resolve()
    requirements = collect_requirement_references(root)
    source_references = collect_source_references(root)
    requirement_packages = {item.package for item in requirements}
    source_packages = {item.package for item in source_references}
    packages = args.packages or sorted(requirement_packages & source_packages)
    for package in dict.fromkeys(canonicalize_name(package) for package in packages):
        print_package_report(package, requirements, source_references, root)


if __name__ == '__main__':
    main()
