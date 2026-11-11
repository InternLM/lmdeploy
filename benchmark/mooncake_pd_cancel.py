# Copyright (c) OpenMMLab. All rights reserved.
"""Check HTTP cancellation and grant ownership against deployed PD
endpoints."""
import argparse
import asyncio
import json

import httpx


async def run(args):
    async with httpx.AsyncClient(trust_env=False, timeout=60) as client:
        payload = dict(model=args.model, input_ids=[100, 101, 102, 103] * 32, messages=[],
                       max_tokens=512, temperature=0, ignore_eos=True, stream=True, return_token_ids=True)

        async def cancel_after_output():
            async with client.stream('POST', args.router_url + '/v1/chat/completions', json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith('data: ') and line != 'data: [DONE]':
                        data = json.loads(line[6:])
                        assert 'error' not in data, data
                        if any(c.get('output_ids') for c in data['choices']):
                            return
            raise AssertionError('No token was produced before the stream ended.')

        for _ in range(args.repeat):
            early = [asyncio.create_task(client.post(args.router_url + '/v1/chat/completions', json=payload))
                     for _ in range(8)]
            await asyncio.sleep(0.02)
            for task in early:
                task.cancel()
            outcomes = await asyncio.gather(*early, return_exceptions=True)
            for outcome in outcomes:
                assert isinstance(outcome, asyncio.CancelledError), outcome
            await asyncio.gather(*(cancel_after_output() for _ in range(8)))

        async def snapshot(url):
            response = await client.get(url + '/distserve/pd_status')
            response.raise_for_status()
            return response.json()

        for _ in range(100):
            snapshots = await asyncio.gather(*(snapshot(url) for url in args.p_url + args.d_url))
            if all(s['exports'] == s['receives'] == s['pinned_blocks'] == 0 for s in snapshots):
                break
            await asyncio.sleep(0.1)
        else:
            raise AssertionError(f'Cancellation retained allocations: {snapshots}')
        assert not any(s['quarantined'] for s in snapshots)

        p_url = args.p_url[0]
        export = await client.post(p_url + '/v1/chat/completions', json=dict(
            payload, stream=False, max_tokens=1, pd_prefill=True, preserve_cache=True, with_cache=True))
        export.raise_for_status()
        handoff = export.json()['pd_handoff']
        control = dict(transfer_id=handoff['transfer_id'], engine_epoch=handoff['engine_epoch'],
                       receiver_epoch='validation-receiver')
        response = await client.post(p_url + '/distserve/pd_control', json=dict(control, action='claim'))
        response.raise_for_status()
        try:
            response = await client.post(p_url + '/distserve/pd_control', json=dict(control, action='revoke'))
            response.raise_for_status()
            held = await snapshot(p_url)
            assert held['exports'] == held['grants'] == 1 and held['pinned_blocks'] > 0, held
            stale = await client.post(p_url + '/distserve/pd_control',
                                      json=dict(control, action='release', engine_epoch='stale'))
            assert stale.status_code == 409, stale.text
        finally:
            response = await client.post(p_url + '/distserve/pd_control', json=dict(control, action='release'))
            response.raise_for_status()
        released = await snapshot(p_url)
        assert released['exports'] == released['pinned_blocks'] == 0, released
        print(json.dumps(dict(cancelled=args.repeat * 16, drained=snapshots,
                              revoked_grant=held, after_release=released)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('router-url', 'model'):
        parser.add_argument('--' + name, required=True)
    for name in ('p-url', 'd-url'):
        parser.add_argument('--' + name, required=True, action='append')
    parser.add_argument('--repeat', type=int, default=4)
    asyncio.run(run(parser.parse_args()))


if __name__ == '__main__':
    main()
