# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from intern-decision-inference (a8fe089), assets.py and image_processing.py.
"""Decision image preparation and TypeSafe-style inline asset references."""
from __future__ import annotations

import base64
import binascii
import json
import math
import re
from collections import Counter
from email import policy
from email.parser import BytesParser
from io import BytesIO
from typing import Literal
from uuid import uuid4

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from requests import RequestException

from lmdeploy.vl.media.connection import load_from_url
from lmdeploy.vl.media.image import ImageMediaIO

# Maximum Systemone image file size in bytes, after base64 decoding or multipart extraction.
MAX_FILE_BYTES = 16 * 1024**2
# Systemone byte limit for HTTP bodies, processed PNG totals, and expanded image-plus-text content.
MAX_TOTAL_BYTES = 32 * 1024**2
# Maximum image occurrences per request, counting repeated references separately.
MAX_OCCURRENCES = 8
# Resize alignment in pixels: patch_size (16) times merge_size (2), one visual token per grid cell.
FACTOR = 32
# Minimum resize target per image in pixels, corresponding to 64 visual tokens.
MIN_PIXELS = 64 * FACTOR**2
# Maximum resize budget per image in pixels, corresponding to 1024 visual tokens.
MAX_PIXELS_PER_IMAGE = 1024 * FACTOR**2
# Total resize budget in pixels, corresponding to 4096 visual tokens shared across image occurrences.
MAX_PIXELS_PER_REQUEST = 4096 * FACTOR**2

class DecisionImageIO(ImageMediaIO):
    """Decisions image policy from a8fe089, also applied to downloaded bytes."""

    max_bytes = 10 * 1024**2

    def load_base64(self, media_type, data):
        if not media_type.lower().startswith('image/'):
            raise ValueError('Image data URI must use an image MIME type.')
        if len(data) > 4 * ((self.max_bytes + 2) // 3):
            raise ValueError('Image exceeds the 10 MiB decoded-byte limit.')
        try:
            raw = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError, TypeError) as exc:
            raise ValueError('Image must contain valid base64 data.') from exc
        return self.load_bytes(raw)

    def load_bytes(self, data):
        if len(data) > self.max_bytes:
            raise ValueError('Image exceeds the 10 MiB decoded-byte limit.')
        if not data:
            raise ValueError('Image payload must not be empty.')
        try:
            with Image.open(BytesIO(data)) as image:
                if image.format not in {'PNG', 'JPEG', 'WEBP', 'GIF'} or getattr(image, 'n_frames', 1) != 1:
                    raise ValueError('Use a static PNG, JPEG, WEBP or GIF image.')
                width, height = image.size
                if width * height > 20_000_000 or max(width, height) > 20_000:
                    raise ValueError('Image exceeds 20 million pixels or 20000 pixels on one side.')
                if max(width, height) / min(width, height) > 200:
                    raise ValueError('Image aspect ratio must not exceed 200:1.')
                image.load()
                return image.convert('RGB')
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError('Image payload is not a valid image.') from exc

    def load_file(self, file_path):
        raise ValueError('Decisions images cannot reference local files.')


def prepare_decision_images(image_inputs, *, allowed_media_domains=None):
    """Decode once per URL, preserving source field locations and occurrence budgets."""
    images = {}
    loader = DecisionImageIO()
    try:
        for url, location in image_inputs:
            if url in images:
                continue
            try:
                if url.lower().startswith('data:'):
                    header, _, data = url.partition(',')
                    images[url] = loader.load_base64(header[5:].split(';')[0], data)
                else:
                    images[url] = load_from_url(url, loader, allowed_media_domains=allowed_media_domains)
            except (ValueError, RequestException) as exc:
                if isinstance(exc, RequestException):
                    exc = ValueError('Could not download image.')
                raise ValidationError.from_exception_data('DecisionInputImage', [dict(
                    type='value_error', loc=location, ctx={'error': exc})]) from exc
        for url, image in images.items():
            images[url] = resized = resize_image(image, len(image_inputs))
            if resized is not image:
                image.close()
    except BaseException:
        for image in images.values():
            image.close()
        raise
    return images


class AssetError(Exception):
    def __init__(self, message, status=400):
        self.status, self.message = status, message
        super().__init__(message)


class ImageAsset(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    type: Literal['image'] = 'image'
    format: Literal['png', 'jpg', 'jpeg'] | None = None
    data_base64: str | None = Field(default=None, repr=False)
    part: str | None = None


def decode_asset(asset):
    if asset.get('part') is not None:
        raise AssetError('assets.part requires multipart/form-data with a matching image part.')
    encoded, fmt = asset.get('data_base64'), asset.get('format')
    if fmt not in ('png', 'jpg', 'jpeg') or not isinstance(encoded, str) or not encoded:
        raise AssetError('Each base64 asset requires format png/jpg/jpeg and data_base64.')
    if len(encoded) > 4 * ((MAX_FILE_BYTES + 2) // 3):
        raise AssetError('Image file exceeds 16 MiB.', 413)
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise AssetError('data_base64 must be padded standard base64 without whitespace or data: prefix.') from None
    if base64.b64encode(raw).decode('ascii') != encoded:
        raise AssetError('data_base64 must use canonical padded standard base64.')
    if len(raw) > MAX_FILE_BYTES:
        raise AssetError('Image file exceeds 16 MiB.', 413)
    try:
        with Image.open(BytesIO(raw)) as image:
            if image.format != ('PNG' if fmt == 'png' else 'JPEG') or getattr(image, 'n_frames', 1) != 1:
                raise AssetError('Asset format must match a static PNG or JPEG file.')
            w, h = image.size
            if max(w, h) > 1568 or w * h > 2458624:
                raise AssetError('Image dimensions exceed 1568 pixels per side or 2458624 total pixels.', 413)
            if min(w, h) < 280 or w * h < 78400:
                raise AssetError('Image requires at least 280 pixels per side and 78400 total pixels.')
            image.verify()
        with Image.open(BytesIO(raw)) as decoded:
            decoded.load()
            rgb = decoded.convert('RGB')
            rgb.info.clear()  # Do not retain metadata or apply EXIF orientation.
            return rgb
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, SyntaxError):
        raise AssetError('Asset is not a complete, valid PNG or JPEG image.') from None


def reference_pattern(refs):
    # At the same offset, prefer the longest reference, then insertion order.
    # TypeSafe specifies non-overlap but not same-offset tie breaking.
    return re.compile('|'.join(re.escape(r) for r in sorted(refs, key=len, reverse=True))) if refs else None


def map_strings(value, transform):
    if isinstance(value, str):
        return transform(value)
    if isinstance(value, list):
        return [map_strings(v, transform) for v in value]
    if isinstance(value, dict):
        return {k: map_strings(v, transform) for k, v in value.items()}
    return value


def map_evidence(row, transform):
    questions = {}
    for key, q in row['questions'].items():
        questions[key] = dict(q)
        if 'instructions' in q:
            questions[key]['instructions'] = map_strings(q['instructions'], transform)
        if 'criteria' in q:
            questions[key]['criteria'] = map_strings(q['criteria'], transform)
    return {**row, 'state': map_strings(row['state'], transform), 'questions': questions}


def prepare_assets(row) -> dict[str, Image.Image]:
    """Validate before resizing and return request-owned, fully loaded RGB images.

    References may share an image for read-only preprocessing. No open file handles
    are retained; successful results live as long as the parser and its messages.
    """
    assets = row.get('assets') or {}
    if not assets:
        return {}
    lengths = []
    for ref in assets:
        if not ref.strip():
            raise AssetError('Asset references must not be empty or blank.')
        try:
            lengths.append(len(ref.encode('utf-8')))
        except UnicodeError:
            raise AssetError('Asset references must contain valid UTF-8 text.') from None
    if max(lengths) > 1024 or sum(lengths) > 65536:
        raise AssetError('Asset references exceed 1024 bytes each or 65536 bytes combined.', 413)
    pattern, counts = reference_pattern(assets), Counter()
    def count(value):
        if value in assets:
            counts[value] += 1
        else:
            counts.update(m.group() for m in pattern.finditer(value))
        return value
    map_evidence(row, count)
    if set(counts) != set(assets):
        raise AssetError('Every asset must be referenced in state, instructions or criteria string values; '
                         'object keys do not count.')
    if sum(counts.values()) > MAX_OCCURRENCES:
        raise AssetError('Intern-Decision supports at most 8 expanded image occurrences, '
                         'including repeated references.')
    total = expanded = 0
    text_row = {k: row[k] for k in ('state', 'questions')}
    expanded += len(json.dumps(text_row, ensure_ascii=False).encode('utf-8'))
    images = {}
    try:
        for ref, asset in assets.items():
            images[ref] = image = decode_asset(asset)
            # Preserve pre-resize byte limits, including every repeated reference.
            with BytesIO() as buffer:
                image.save(buffer, format='PNG')
                size = buffer.tell()
            total += size
            expanded += size * counts[ref]
            if total > MAX_TOTAL_BYTES or expanded > MAX_TOTAL_BYTES:
                raise AssetError('Processed images or expanded request exceed 32 MiB.', 413)
        for ref, image in images.items():
            images[ref] = resized = resize_image(image, sum(counts.values()))
            if resized is not image:
                image.close()
    except BaseException:
        for image in images.values():
            image.close()
        raise
    return images


def prepare_evidence(row):
    """Replace only evidence values with temporary sentinels before serialization.

    Sentinels never reach the tokenizer. This prevents accidental replacement in
    JSON keys, prompt scaffolding, or serialized escape sequences.
    """
    assets = row.get('assets') or {}
    if not assets:
        return row, {}
    pattern = reference_pattern(assets)
    original = json.dumps({k: row[k] for k in ('state', 'questions')}, ensure_ascii=False)
    while True:
        prefix = 'ASSET_' + uuid4().hex + '_'
        if prefix not in original:
            break
    tokens = {ref: prefix + str(i) + '_END' for i, ref in enumerate(assets)}
    def replace(value):
        return tokens[value] if value in assets else pattern.sub(lambda m: tokens[m.group()], value)
    return map_evidence(row, replace), {token: ref for ref, token in tokens.items()}


def image_content(text, sentinels, images_by_ref):
    """Expand sentinels into image blocks without re-encoding or decoding assets."""
    if not sentinels:
        return text
    pattern = re.compile('(' + '|'.join(re.escape(s) for s in sentinels) + ')')
    content = []
    for fragment in pattern.split(text):
        if fragment in sentinels:
            content.append({'type': 'image_data', 'image_data': {'data': images_by_ref[sentinels[fragment]]}})
        elif fragment:
            content.append({'type': 'text', 'text': fragment})
    return content


def unpack_multipart(raw, content_type):
    """Parse an already size-bounded MIME body in the validation executor.

    No temporary files, URLs, or client-supplied paths are opened. MIME transfer
    encodings and nested multiparts are deliberately rejected.
    """
    if '\r' in content_type or '\n' in content_type:
        raise AssetError('Invalid multipart Content-Type.')
    msg = BytesParser(policy=policy.default).parsebytes(
        ('Content-Type: ' + content_type + '\r\nMIME-Version: 1.0\r\n\r\n').encode('ascii') + raw)
    if msg.get_content_type() != 'multipart/form-data' or not msg.is_multipart():
        raise AssetError('Expected multipart/form-data with a boundary.')
    if any(part.defects for part in msg.walk()):
        raise AssetError('Malformed or incomplete multipart request.')
    parts = list(msg.iter_parts())
    if not parts or len(parts) > MAX_OCCURRENCES + 1:
        raise AssetError('Multipart requires request first and at most 8 image parts.')
    files, body = {}, None
    for i, part in enumerate(parts):
        name = part.get_param('name', header='content-disposition')
        if (part.is_multipart() or part.get_content_disposition() != 'form-data'
                or not isinstance(name, str) or not name.strip()
                or part.get('Content-Transfer-Encoding') is not None
                or len(part.get_all('Content-Disposition', [])) != 1
                or len(part.get_all('Content-Type', [])) > 1):
            raise AssetError('Invalid multipart part name, disposition or encoding.')
        payload = part.get_payload(decode=True)
        if i == 0:
            if name != 'request' or part.get_content_type() != 'application/json':
                raise AssetError('First multipart part must be request with application/json.')
            try:
                body = json.loads(payload)
            except (ValueError, UnicodeError, RecursionError):
                raise AssetError('Malformed JSON request part.') from None
        else:
            if name == 'request' or name in files:
                raise AssetError('Multipart names must be unique; request is reserved.')
            if part.get('Content-Type') is not None and part.get_content_type() not in (
                    'image/png', 'image/jpeg', 'image/jpg', 'application/octet-stream'):
                raise AssetError('Unsupported image part Content-Type.')
            if len(payload) > MAX_FILE_BYTES:
                raise AssetError('Image file exceeds 16 MiB.', 413)
            files[name] = payload
    if not isinstance(body, dict) or not isinstance(body.get('assets', {}), dict):
        raise AssetError('Multipart request must be an object with an assets map.')
    used = set()
    for asset in body.get('assets', {}).values():
        if not isinstance(asset, dict) or 'part' not in asset:
            continue
        name = asset['part']
        if not isinstance(name, str) or name not in files or name in used:
            raise AssetError('Each image part must be referenced by exactly one asset.')
        used.add(name)
        data = files[name]
        fmt = asset.get('format')
        if fmt is None:
            fmt = 'png' if data.startswith(b'\x89PNG\r\n\x1a\n') else 'jpg'
        asset.pop('part')
        asset['format'] = fmt
        asset['data_base64'] = base64.b64encode(data).decode('ascii')
    if used != set(files):
        raise AssetError('Every image part must be referenced by exactly one asset.')
    return body


def resized_size(width, height, image_count):
    if image_count < 1 or image_count > 8:
        raise ValueError('Expected 1 to 8 images')
    budget = min(MAX_PIXELS_PER_IMAGE, (MAX_PIXELS_PER_REQUEST // image_count // FACTOR**2) * FACTOR**2)
    w = max(FACTOR, round(width / FACTOR) * FACTOR)
    h = max(FACTOR, round(height / FACTOR) * FACTOR)
    if w * h > budget:
        scale = math.sqrt(width * height / budget)
        w = max(FACTOR, math.floor(width / scale / FACTOR) * FACTOR)
        h = max(FACTOR, math.floor(height / scale / FACTOR) * FACTOR)
    elif w * h < MIN_PIXELS:
        scale = math.sqrt(MIN_PIXELS / (width * height))
        w = math.ceil(width * scale / FACTOR) * FACTOR
        h = math.ceil(height * scale / FACTOR) * FACTOR
    return w, h


def resize_image(image, image_count):
    size = resized_size(*image.size, image_count)
    return image if image.size == size else image.resize(size, Image.Resampling.BICUBIC)
