# Copyright (c) OpenMMLab. All rights reserved.
"""Eight-rank GLM-5.3 MoE A/B using real DeepEP and shared-expert kernels.

Uses the official config with synthetic weights, not a model-accuracy workload. Each timing trial interleaves six CUDA
graphs over the same inputs and weights.
"""
import argparse
import json
import os
import statistics
from collections import Counter
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from lmdeploy.pytorch import envs
from lmdeploy.pytorch.backends.cuda.token_dispatcher import DeepEPBuffer
from lmdeploy.pytorch.config import DistConfig, QuantizationConfig
from lmdeploy.pytorch.distributed import DistContext, get_dist_manager
from lmdeploy.pytorch.model_inputs import BuildModelContext, StepContextManager, set_step_ctx_manager
from lmdeploy.pytorch.models.glm5_next import Glm5NextMoE
from lmdeploy.pytorch.models.patch import build_model_context

ARMS = {'baseline': (False, False), 'prepare': (False, True),
        'overlap': (True, False), 'both': (True, True),
        'full': (False, False), 'full-overlap': (True, False)}


def initialize_weights(model):
    """Keep finite, nontrivial expert outputs without loading the whole LLM."""
    for name, parameter in model.named_parameters():
        if parameter.dtype == torch.float8_e4m3fn:
            parameter.copy_(torch.randn(parameter.shape, device=parameter.device,
                                        dtype=torch.bfloat16).to(parameter.dtype))
        elif 'weight_scale_inv' in name:
            k = 2048 if '.down.' in name or '.down_proj.' in name else 4096
            parameter.fill_(k ** -.5)
        elif 'correction_bias' in name:
            parameter.uniform_(-.05, .05)
        else:
            parameter.normal_(std=4096 ** -.5)


def run(args):
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', timeout=timedelta(seconds=240))
    assert dist.get_world_size() == 8
    cfg = DistConfig(dp=8, ep=8, tp=1, dp_rank=rank)
    context = DistContext.build(rank=rank, dist_config=cfg)
    raw = json.loads((args.model / 'config.json').read_text())
    config = SimpleNamespace(**raw['text_config'])
    config.quantization_config = raw['quantization_config']
    build = BuildModelContext(quant_config=QuantizationConfig.from_config(config),
                              max_batch_size=32, num_spec_tokens=5)
    manager = StepContextManager(build)
    set_step_ctx_manager(manager)
    valid = torch.tensor(1, device='cuda', dtype=torch.int32)
    step = SimpleNamespace(global_is_decoding=lambda: True,
                           moe_valid_token_count=valid, is_dummy=False)
    results = {'kind': 'synthetic_weights_real_dp8_ep8_moe', 'gpu': torch.cuda.get_device_name(),
               'trials': args.trials, 'replays': args.replays,
               'arm_flags': {name: dict(shared_overlap=flags[0], fused_prepare=flags[1],
                                        full_prepare=name in ('full', 'full-overlap'))
                             for name, flags in ARMS.items()}, 'workloads': {}}
    with get_dist_manager().context(context), build_model_context(build), manager.context(step), torch.inference_mode():
        torch.manual_seed(65306 + rank)
        model = Glm5NextMoE(config, layer_idx=0, dtype=torch.bfloat16, device=torch.device('cuda', rank))
        initialize_weights(model)
        if rank == 0:
            print('MODEL_READY', type(model.shared_experts.gate_up_proj.impl).__name__, flush=True)
        for rows in args.rows:
            valid.fill_(rows)
            x = torch.randn(1, rows, 4096, device='cuda', dtype=torch.bfloat16)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graphs, outputs = {}, {}
            for arm, (overlap, prepare) in ARMS.items():
                model._enable_shared_expert_overlap = overlap
                model._enable_fused_moe_prepare = arm in ('full', 'full-overlap')
                # The backend constructs its LL facade on every forward.
                # Change the constructor's source setting, not a stale facade.
                envs.glm53_fuse_route_prepare = prepare
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        model(x)
                torch.cuda.synchronize()
                dist.barrier()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    outputs[arm] = model(x)
                graphs[arm] = graph
                graph.replay()
                torch.cuda.synchronize()
                dist.barrier()
            correctness = {}
            reference = outputs['baseline'].clone()
            for arm in ARMS:
                torch.testing.assert_close(outputs[arm], reference, rtol=0, atol=0)
                correctness[arm] = 'bitwise_equal_all_eight_ranks'
            times = {arm: [] for arm in ARMS}
            for trial in range(args.trials):
                order = list(ARMS)
                order = order[trial % len(order):] + order[:trial % len(order)]
                for arm in order:
                    dist.barrier()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.replays):
                        graphs[arm].replay()
                    end.record()
                    end.synchronize()
                    elapsed = torch.tensor(start.elapsed_time(end) * 1000 / args.replays, device='cuda')
                    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                    times[arm].append(elapsed.item())
            base = statistics.median(times['baseline'])
            summary = {arm: dict(median_us=statistics.median(values), trials_us=values,
                                  speedup=base / statistics.median(values)) for arm, values in times.items()}
            results['workloads'][str(rows)] = dict(arms=summary, correctness=correctness)
            if rank == 0:
                print('ROWS', rows, {arm: round(item['median_us'], 3) for arm, item in summary.items()}, flush=True)
                args.output.write_text(json.dumps(results, indent=2) + '\n')
            if args.trace and rows == args.rows[-1]:
                if rank == 0:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as profiler:
                        for arm in ('baseline', 'full-overlap'):
                            with torch.profiler.record_function(arm):
                                for _ in range(4):
                                    graphs[arm].replay()
                            torch.cuda.synchronize()
                    profiler.export_chrome_trace(str(args.trace))
                    trace = json.loads(args.trace.read_text())
                    counts = Counter(item['name'] for item in trace['traceEvents'] if item.get('cat') == 'kernel')
                    # Check the actual captured kernels, not only Python flags.
                    assert counts['_glm53_moe_prepare_kernel'] == 4, counts
                    assert counts['_mask_invalid_routes_kernel'] == 4, counts
                    results['trace_route_kernel_counts'] = {
                        name: counts[name] for name in ('_glm53_moe_prepare_kernel', '_mask_invalid_routes_kernel')}
                    args.output.write_text(json.dumps(results, indent=2) + '\n')
                else:
                    for arm in ('baseline', 'full-overlap'):
                        for _ in range(4):
                            graphs[arm].replay()
                        torch.cuda.synchronize()
                dist.barrier()
            del graphs, outputs
        torch.cuda.synchronize()
        dist.barrier()
        DeepEPBuffer.destroy()
    context.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trace', type=Path)
    parser.add_argument('--rows', type=int, nargs='+', default=[1, 2, 4, 6, 12, 16, 24, 96, 192])
    parser.add_argument('--trials', type=int, default=11)
    parser.add_argument('--replays', type=int, default=20)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(arguments.output)
    run(arguments)
