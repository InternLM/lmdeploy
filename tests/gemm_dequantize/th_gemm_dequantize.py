# flake8: noqa
import unittest

import torch


def random_tensor(shape, dtype, device, mean=0, std=1):
    return torch.empty(shape, dtype=dtype, device=device).normal_(mean, std)


class TestGemmDequantize(unittest.TestCase):

    def setUp(self) -> None:
        torch.classes.load_library('lib/libth_transformer.so')
        torch.classes.load_library('lib/libgemm_dq_unit_ops.so')
        self.unpack_packed_int4s = torch.ops.fastertransformer.unpack_int4_packed_tensor_to_int8
        self.pack_int4s = torch.ops.fastertransformer.pack_int8_tensor_to_packed_int4
        self.fused_gemm_dq = torch.ops.gemm_dq_unit_ops.fused_gemm_dq
        self.fused_gemm_dq_bias_act = torch.ops.gemm_dq_unit_ops.fused_gemm_dq_bias_act
        self.bench = torch.ops.gemm_dq_unit_ops.benchmark_against_cublas_fp
        self.preprocess_weights_for_mixed_gemm = torch.ops.fastertransformer.preprocess_weights_for_mixed_gemm

        self.symmetric_quantizer = torch.ops.fastertransformer._symmetric_quantize_last_axis_of_batched_matrix

        torch.manual_seed(734876213)

    def dequantize_test_helper(self, weight_type, quant_type):
        assert quant_type == torch.int8 or quant_type == torch.quint4x2

        lower_bound = -128 if quant_type == torch.int8 else -8
        upper_bound = 127 if quant_type == torch.int8 else 7

        m, n, k = 64, 128, 64
        weights = torch.randint(lower_bound,
                                upper_bound, [k, n],
                                dtype=torch.int8,
                                device='cpu')

        packed_weight = self.pack_int4s(
            weights) if quant_type == torch.quint4x2 else weights
        cuda_weights = self.preprocess_weights_for_mixed_gemm(
            packed_weight, quant_type).to('cuda')
        weights = weights.to('cuda')

        act = torch.eye(m, dtype=weight_type, device='cuda')
        scales = torch.ones([n], dtype=weight_type, device='cuda')

        actual = self.fused_gemm_dq(act, cuda_weights, scales)
        torch.testing.assert_close(actual,
                                   weights,
                                   atol=0,
                                   rtol=0,
                                   check_dtype=False)

    def test_fp16_int8_dequantize(self):
        self.dequantize_test_helper(torch.float16, torch.int8)

    def test_bf16_int8_dequantize(self):
        self.dequantize_test_helper(torch.bfloat16, torch.int8)

    def test_fp16_int4_dequantize(self):
        self.dequantize_test_helper(torch.float16, torch.quint4x2)

    def test_bf16_int4_dequantize(self):
        self.dequantize_test_helper(torch.bfloat16, torch.quint4x2)

    def apply_act(self, inp, act_str):
        if act_str == 'identity':
            return inp
        elif act_str == 'silu':
            return torch.nn.SiLU()(inp)
        elif act_str == 'relu':
            return torch.nn.ReLU()(inp)
        elif act_str == 'gelu':
            return torch.nn.GELU(approximate='tanh')(inp)
        else:
            assert False, 'Unsupported activation'

    def gemm_dequant_test_helper(self,
                                 compute_type,
                                 weight_dtype,
                                 gemm_ms,
                                 gemm_ns,
                                 gemm_ks,
                                 rtol,
                                 atol,
                                 act_str='only_gemm',
                                 benchmark=False):
        assert weight_dtype == torch.int8 or weight_dtype == torch.quint4x2, 'Weight must be quantized'

        for gemm_k in gemm_ks:
            for gemm_n in gemm_ns:
                torch_weights_cpu = random_tensor((gemm_k, gemm_n),
                                                  dtype=compute_type,
                                                  device='cpu',
                                                  mean=0,
                                                  std=0.002)
                ref_torch_weights, processed_torch_weights, torch_weight_scales = self.symmetric_quantizer(
                    torch_weights_cpu, weight_dtype)
                ref_torch_weights = self.unpack_packed_int4s(
                    ref_torch_weights
                ) if weight_dtype == torch.quint4x2 else ref_torch_weights
                ref_torch_weights = ref_torch_weights.to('cuda')
                processed_torch_weights = processed_torch_weights.to('cuda')
                torch_weight_scales = torch_weight_scales.to('cuda')
                torch_biases = random_tensor((gemm_n),
                                             dtype=compute_type,
                                             device='cuda',
                                             mean=0,
                                             std=0.1)

                for num_rows in gemm_ms:
                    torch_activations = torch.randn(size=(num_rows, gemm_k),
                                                    dtype=compute_type,
                                                    device='cuda')

                    scales_unsqueezed = torch_weight_scales.unsqueeze(0)
                    casted_weights = ref_torch_weights.to(
                        torch_activations.dtype)
                    dequantized_weights = torch.multiply(
                        casted_weights, scales_unsqueezed)
                    if benchmark:
                        assert act_str == 'only_gemm', 'Benchmarks against cublas must use just GEMM.'
                        torch.cuda.profiler.start()
                        times, results = self.bench(torch_activations,
                                                    processed_torch_weights,
                                                    torch_weight_scales,
                                                    dequantized_weights, 200)
                        torch.cuda.profiler.stop()
                        times = times[0]
                        cublas_time = times[0].item()
                        ft_time = times[1].item()
                        ft_speedup = cublas_time / ft_time
                        print('{},{},{},{},{},{}'.format(
                            num_rows, gemm_n, gemm_k, cublas_time, ft_time,
                            ft_speedup))
                        reference_result = results[0]
                        ft_result = results[1]
                    else:
                        if act_str == 'only_gemm':
                            reference_result = torch.matmul(
                                torch_activations, dequantized_weights)
                            ft_result = self.fused_gemm_dq(
                                torch_activations, processed_torch_weights,
                                torch_weight_scales)
                        else:
                            reference_result = torch.matmul(
                                torch_activations, dequantized_weights)
                            reference_result += torch_biases.unsqueeze(0)
                            reference_result = self.apply_act(
                                reference_result, act_str)

                            ft_result = self.fused_gemm_dq_bias_act(
                                torch_activations, processed_torch_weights,
                                torch_weight_scales, torch_biases, act_str)

                    msg = 'FC1 Failed on m={}, n={}, k={}'.format(
                        num_rows, gemm_n, gemm_k)
                    torch.testing.assert_close(ft_result,
                                               reference_result,
                                               rtol=rtol,
                                               atol=atol,
                                               msg=msg,
                                               check_dtype=False)

    def test_fp16_int8_gemm(self):
        self.gemm_dequant_test_helper(
            torch.float16,
            torch.int8,
            gemm_ms=[256, 177, 195, 125, 66, 33, 8, 2, 1],
            gemm_ns=[1024, 2048, 4096],
            gemm_ks=[4096, 8192, 16384],
            rtol=0.001,
            atol=0.002)

    def test_fp16_int4_gemm(self):
        self.gemm_dequant_test_helper(
            torch.float16,
            torch.quint4x2,
            gemm_ms=[256, 177, 195, 125, 66, 33, 8, 2, 1],
            gemm_ns=[1024, 2048, 4096],
            gemm_ks=[4096, 8192, 16384],
            rtol=0.001,
            atol=0.002)

    def test_bf16_int8_gemm(self):
        self.gemm_dequant_test_helper(
            torch.bfloat16,
            torch.int8,
            gemm_ms=[256, 177, 195, 125, 66, 33, 8, 2, 1],
            gemm_ns=[1024, 2048, 4096],
            gemm_ks=[4096, 8192, 16384],
            rtol=0.01,
            atol=0.01)

    def test_bf16_int4_gemm(self):
        self.gemm_dequant_test_helper(
            torch.bfloat16,
            torch.quint4x2,
            gemm_ms=[256, 177, 195, 125, 66, 33, 8, 2, 1],
            gemm_ns=[1024, 2048, 4096],
            gemm_ks=[4096, 8192, 16384],
            rtol=0.01,
            atol=0.01)

    def test_fp16_int8_gemm_bias(self):
        self.gemm_dequant_test_helper(torch.float16,
                                      torch.int8,
                                      gemm_ms=[256],
                                      gemm_ns=[1024],
                                      gemm_ks=[8192],
                                      rtol=0.001,
                                      atol=0.002,
                                      act_str='identity')

    def test_fp16_int8_gemm_bias_relu(self):
        self.gemm_dequant_test_helper(torch.float16,
                                      torch.int8,
                                      gemm_ms=[256],
                                      gemm_ns=[1024],
                                      gemm_ks=[8192],
                                      rtol=0.001,
                                      atol=0.002,
                                      act_str='relu')

    def test_fp16_int8_gemm_bias_gelu(self):
        self.gemm_dequant_test_helper(torch.float16,
                                      torch.int8,
                                      gemm_ms=[256],
                                      gemm_ns=[1024],
                                      gemm_ks=[8192],
                                      rtol=0.001,
                                      atol=0.002,
                                      act_str='gelu')

    def test_fp16_int8_gemm_bias_silu(self):
        self.gemm_dequant_test_helper(torch.float16,
                                      torch.int8,
                                      gemm_ms=[256],
                                      gemm_ns=[1024],
                                      gemm_ks=[8192],
                                      rtol=0.001,
                                      atol=0.002,
                                      act_str='silu')

    def bench_helper(self, act_type, quant_type, rtol, atol):
        # Warm, using bfloat here since it seems to reliably use cublas.
        x = random_tensor([20480, 20480], torch.bfloat16, device='cuda')
        warm_iters = 30
        for iter in range(warm_iters):
            res = x @ x

        m_shapes = torch.arange(0, 12)
        m_shapes = 2**m_shapes

        self.gemm_dequant_test_helper(act_type,
                                      quant_type,
                                      gemm_ms=[128],
                                      gemm_ns=[1536],
                                      gemm_ks=[12288],
                                      rtol=rtol,
                                      atol=atol,
                                      benchmark=True)

    @unittest.skip("This is a benchmark so don't run by default")
    def test_fp16_int8_cublas(self):
        self.bench_helper(torch.float16, torch.int8, 1e-3, 0.002)

    @unittest.skip("This is a benchmark so don't run by default")
    def test_bf16_int8_cublas(self):
        self.bench_helper(torch.bfloat16, torch.int8, 1e-2, 1e-2)

    @unittest.skip("This is a benchmark so don't run by default")
    def test_fp16_int4_cublas(self):
        self.bench_helper(torch.float16, torch.quint4x2, 1e-3, 0.002)

    @unittest.skip("This is a benchmark so don't run by default")
    def test_bf16_int4_cublas(self):
        self.bench_helper(torch.bfloat16, torch.quint4x2, 1e-2, 1e-2)


if __name__ == '__main__':
    unittest.main()
