# Copyright (c) OpenMMLab. All rights reserved.
"""Score one native Intern-Decision request through /v1/jev.

python examples/decision/decision.py --server http://127.0.0.1:23333 [--image image.png]
"""
import argparse
import base64
import json
import mimetypes

REQUEST = {
    'state': {
        'signal': 'red'
    },
    'questions': {
        'action': {
            'type': 'choice',
            'instructions': 'Select an action.',
            'criteria': {
                'stop': 'Stop at the red light.',
                'go': 'Drive through the red light.'
            }
        },
        'risk': {
            'type': 'score',
            'instructions': 'How risky is driving through the light?',
            'criteria': {
                '0': 'Safe',
                '1': 'Some risk',
                '2': 'Dangerous'
            }
        },
        'red': {
            'type': 'noul',
            'instructions': 'Is the signal red?'
        },
    },
}


def upload(path):
    """Encode a client-local image as a complete data URL."""
    media_type = mimetypes.guess_type(path)[0]
    if media_type is None or not media_type.startswith('image/'):
        raise ValueError(f'Cannot determine an image MIME type for {path}')
    with open(path, 'rb') as f:
        data = base64.b64encode(f.read()).decode()
    return f'data:{media_type};base64,{data}'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--server', required=True, help='URL of `lmdeploy serve api_server ... --task intern-decision`')
    parser.add_argument('--image', action='append', default=[])
    args = parser.parse_args()

    import requests
    body = {**REQUEST, 'images': [upload(path) for path in args.image]}
    response = requests.post(f'{args.server}/v1/jev', json=body, timeout=300)
    response.raise_for_status()
    result = response.json()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
