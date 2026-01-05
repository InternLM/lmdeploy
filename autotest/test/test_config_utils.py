import os

from utils.config_utils import (_is_kvint_model, get_case_str_by_config, get_cli_common_param, get_config,
                                get_func_config_list, is_model_in_list, parse_config_by_case)


def test_config():
    os.environ['DEVICE'] = 'test'
    config = get_config()
    assert 'model_path' in config.keys()
    assert 'resource_path' in config.keys()
    assert 'log_path' in config.keys()
    assert 'server_log_path' in config.keys()
    assert 'eval_path' in config.keys()
    assert 'mllm_eval_path' in config.keys()
    assert 'benchmark_path' in config.keys()
    assert 'dataset_path' in config.keys()
    assert 'prefix_dataset_path' in config.keys()
    assert 'env_tag' in config.keys()
    assert 'config' in config.keys()
    assert 'tp' in config.get('config')

    assert is_model_in_list(config, parallel_config={'tp': 1}, model='test/test_tp1')
    assert is_model_in_list(config, parallel_config={'tp': 2}, model='test/test_tp1') is False
    assert is_model_in_list(config, parallel_config={'ep': 1},
                            model='test/test_tp1') is False, is_model_in_list(config,
                                                                              parallel_config={'ep': 1},
                                                                              model='test/test_tp1')
    assert is_model_in_list(config, parallel_config={'tp': 2}, model='test/test_tp2-inner-4bits')
    assert is_model_in_list(config, parallel_config={'tp': 2}, model='test/test_tp2-inner-w8a8')
    assert is_model_in_list(config, parallel_config={'tp': 8}, model='test/test_tp8-inner-gptq')
    assert is_model_in_list(config, parallel_config={'tp': 8}, model='test/test_cp2tp8') is False
    assert is_model_in_list(config, parallel_config={'tp': 8, 'cp': 2}, model='test/test_cp2tp8')
    assert is_model_in_list(config, parallel_config={'cp': 2, 'tp': 8}, model='test/test_cp2tp8')
    assert is_model_in_list(config, parallel_config={'cp': 4, 'tp': 8}, model='test/test_cp2tp8') is False
    assert is_model_in_list(config, parallel_config={'dp': 8, 'ep': 8}, model='test/test_dpep8')
    assert is_model_in_list(config, parallel_config={'dp': 4, 'ep': 8}, model='test/test_dpep8') is False
    assert is_model_in_list(config, parallel_config={'ep': 4, 'dp': 8}, model='test/test_dpep8') is False

    assert _is_kvint_model(config, 'turbomind', 'test/test_tp1-inner-4bits', 8) is False
    assert _is_kvint_model(config, 'turbomind', 'test/test_tp1-inner-4bits', 4)
    assert _is_kvint_model(config, 'turbomind', 'any', 0)
    assert _is_kvint_model(config, 'pytorch', 'test/test_tp1-inner-gptq', 8) is False
    assert _is_kvint_model(config, 'pytorch', 'test/test_tp1-inner-gptq', 4)
    assert _is_kvint_model(config, 'pytorch', 'test/test_vl_tp1-inner-gptq', 8) is False
    assert _is_kvint_model(config, 'pytorch', 'test/test_cp2tp8-inner-w8a8', 4) is False


def test_get_case_str_by_config():
    run_config = {
        'model': 'test/test_dpep16',
        'backend': 'turbomind',
        'communicator': 'nccl',
        'quant_policy': 8,
        'parallel_config': {
            'dp': 16,
            'ep': 16
        }
    }
    case_str = get_case_str_by_config(run_config)
    assert case_str == 'turbomind_test-dpep16_nccl_dp16_ep16_8', case_str
    run_config_parsed = parse_config_by_case(case_str)
    assert run_config_parsed['model'] == 'test-dpep16'
    assert run_config_parsed['backend'] == 'turbomind'
    assert run_config_parsed['communicator'] == 'nccl'
    assert run_config_parsed['quant_policy'] == 8
    assert run_config_parsed['parallel_config']['dp'] == 16
    assert run_config_parsed['parallel_config']['ep'] == 16


def test_cli_common_param():
    run_config = {
        'model': 'test/test_dpep16-inner-4bits',
        'backend': 'turbomind',
        'communicator': 'nccl',
        'quant_policy': 8,
        'parallel_config': {
            'dp': 16,
            'ep': 16
        },
        'dtype': 'bfloat16',
        'device': 'ascend',
        'enable_prefix_caching': None,
        'max_batch_size': 2048,
        'session_len': 8192,
        'cache_max_entry_count': 0.75
    }

    cli_params = get_cli_common_param(run_config)
    assert cli_params == '--backend turbomind --communicator nccl --device ascend --dtype bfloat16 --quant-policy 8 --model-format awq --dp 16 --ep 16 --enable-prefix-caching --max-batch-size 2048 --session-len 8192 --cache-max-entry-count 0.75', cli_params  # noqa
    run_config = {
        'model': 'test/test_dpep16-inner-4bits',
        'backend': 'pytorch',
        'communicator': 'hccl',
        'quant_policy': 0,
        'parallel_config': {
            'tp': 8
        }
    }

    cli_params = get_cli_common_param(run_config)
    assert cli_params == '--backend pytorch --communicator hccl --tp 8', cli_params


def test_return_info_turbomind():
    os.environ['TEST_ENV'] = 'test'
    backend = 'turbomind'
    func_chat_tp1 = get_func_config_list(backend, parallel_config={'tp': 1}, model_type='chat_model', func_type='func')
    assert len(func_chat_tp1) == 12, len(func_chat_tp1)
    func_chat_tp2 = get_func_config_list(backend, parallel_config={'tp': 2}, model_type='chat_model', func_type='func')
    assert len(func_chat_tp2) == 32, len(func_chat_tp2)
    func_chat_tp8 = get_func_config_list(backend, parallel_config={'tp': 8}, model_type='chat_model', func_type='func')
    assert len(func_chat_tp8) == 24, len(func_chat_tp8)
    func_chat_cptp = get_func_config_list(backend,
                                          parallel_config={
                                              'cp': 2,
                                              'tp': 8
                                          },
                                          model_type='chat_model',
                                          func_type='func')
    assert len(func_chat_cptp) == 8, len(func_chat_cptp)
    func_chat_dpep8 = get_func_config_list(backend,
                                           parallel_config={
                                               'dp': 8,
                                               'ep': 8
                                           },
                                           model_type='chat_model',
                                           func_type='func')
    assert len(func_chat_dpep8) == 0, len(func_chat_dpep8)
    func_chat_dpep16 = get_func_config_list(backend,
                                            parallel_config={
                                                'dp': 16,
                                                'ep': 16
                                            },
                                            model_type='chat_model',
                                            func_type='func')
    assert len(func_chat_dpep16) == 0, len(func_chat_dpep16)
    func_base_tp1 = get_func_config_list(backend, parallel_config={'tp': 1}, model_type='base_model', func_type='func')
    assert len(func_base_tp1) == 6, len(func_base_tp1)
    func_base_tp2 = get_func_config_list(backend, parallel_config={'tp': 2}, model_type='base_model', func_type='func')
    assert len(func_base_tp2) == 4, len(func_base_tp2)

    evaluate_tp1 = get_func_config_list(backend,
                                        parallel_config={'tp': 1},
                                        model_type='chat_model',
                                        func_type='evaluate')
    assert len(evaluate_tp1) == 6, len(evaluate_tp1)
    benchmark_tp2 = get_func_config_list(backend,
                                         parallel_config={'tp': 2},
                                         model_type='chat_model',
                                         func_type='benchmark')
    assert len(benchmark_tp2) == 4, len(benchmark_tp2)
    longtext_tp8 = get_func_config_list(backend,
                                        parallel_config={'tp': 8},
                                        model_type='chat_model',
                                        func_type='longtext')
    assert len(longtext_tp8) == 12, len(longtext_tp8)
    evaluate_cptp = get_func_config_list(backend,
                                         parallel_config={
                                             'cp': 2,
                                             'tp': 8
                                         },
                                         model_type='chat_model',
                                         func_type='evaluate')
    assert len(evaluate_cptp) == 4, len(evaluate_cptp)
    benchmark_dpep8 = get_func_config_list(backend,
                                           parallel_config={
                                               'dp': 8,
                                               'ep': 8
                                           },
                                           model_type='chat_model',
                                           func_type='benchmark')
    assert len(benchmark_dpep8) == 0, len(benchmark_dpep8)

    mllm_benchmark_tp1 = get_func_config_list(backend,
                                              parallel_config={'tp': 1},
                                              model_type='chat_model',
                                              func_type='mllm_benchmark')
    assert len(mllm_benchmark_tp1) == 6, len(mllm_benchmark_tp1)
    mllm_longtext_tp2 = get_func_config_list(backend,
                                             parallel_config={'tp': 2},
                                             model_type='chat_model',
                                             func_type='mllm_longtext')
    assert len(mllm_longtext_tp2) == 0, len(mllm_longtext_tp2)
    mllm_evaluate_tp8 = get_func_config_list(backend,
                                             parallel_config={'tp': 8},
                                             model_type='chat_model',
                                             func_type='mllm_evaluate')
    assert len(mllm_evaluate_tp8) == 12, len(mllm_evaluate_tp8)
    mllm_evaluate_dpep16 = get_func_config_list(backend,
                                                parallel_config={
                                                    'dp': 16,
                                                    'ep': 16
                                                },
                                                model_type='chat_model',
                                                func_type='evaluate')
    assert len(mllm_evaluate_dpep16) == 0, len(mllm_evaluate_dpep16)
    mllm_benchmark_cptp = get_func_config_list(backend,
                                               parallel_config={
                                                   'cp': 2,
                                                   'tp': 8
                                               },
                                               model_type='chat_model',
                                               func_type='benchmark')
    assert len(mllm_benchmark_cptp) == 4, len(mllm_benchmark_cptp)


def test_return_info_pytorch():
    os.environ['TEST_ENV'] = 'test'
    backend = 'pytorch'
    func_chat_tp1 = get_func_config_list(backend, parallel_config={'tp': 1}, model_type='chat_model', func_type='func')
    assert len(func_chat_tp1) == 12, len(func_chat_tp1)
    func_chat_tp2 = get_func_config_list(backend, parallel_config={'tp': 2}, model_type='chat_model', func_type='func')
    assert len(func_chat_tp2) == 19, len(func_chat_tp2)
    func_chat_tp8 = get_func_config_list(backend, parallel_config={'tp': 8}, model_type='chat_model', func_type='func')
    assert len(func_chat_tp8) == 6, len(func_chat_tp8)
    func_chat_cptp = get_func_config_list(backend,
                                          parallel_config={
                                              'cp': 2,
                                              'tp': 8
                                          },
                                          model_type='chat_model',
                                          func_type='func')
    assert len(func_chat_cptp) == 4, len(func_chat_cptp)
    func_chat_dpep8 = get_func_config_list(backend,
                                           parallel_config={
                                               'dp': 8,
                                               'ep': 8
                                           },
                                           model_type='chat_model',
                                           func_type='func')
    assert len(func_chat_dpep8) == 5, len(func_chat_dpep8)
    func_chat_dpep16 = get_func_config_list(backend,
                                            parallel_config={
                                                'dp': 16,
                                                'ep': 16
                                            },
                                            model_type='chat_model',
                                            func_type='func')
    assert len(func_chat_dpep16) == 6, len(func_chat_dpep16)
    func_base_tp1 = get_func_config_list(backend, parallel_config={'tp': 1}, model_type='base_model', func_type='func')
    assert len(func_base_tp1) == 7, len(func_base_tp1)
    func_base_tp2 = get_func_config_list(backend, parallel_config={'tp': 2}, model_type='base_model', func_type='func')
    assert len(func_base_tp2) == 4, len(func_base_tp2)

    evaluate_tp1 = get_func_config_list(backend,
                                        parallel_config={'tp': 1},
                                        model_type='chat_model',
                                        func_type='evaluate')
    assert len(evaluate_tp1) == 7, len(evaluate_tp1)
    benchmark_tp2 = get_func_config_list(backend,
                                         parallel_config={'tp': 2},
                                         model_type='chat_model',
                                         func_type='benchmark')
    assert len(benchmark_tp2) == 3, len(benchmark_tp2)
    longtext_tp8 = get_func_config_list(backend,
                                        parallel_config={'tp': 8},
                                        model_type='chat_model',
                                        func_type='longtext')
    assert len(longtext_tp8) == 3, len(longtext_tp8)
    evaluate_cptp = get_func_config_list(backend,
                                         parallel_config={
                                             'cp': 2,
                                             'tp': 8
                                         },
                                         model_type='chat_model',
                                         func_type='evaluate')
    assert len(evaluate_cptp) == 2, len(evaluate_cptp)
    benchmark_dpep8 = get_func_config_list(backend,
                                           parallel_config={
                                               'dp': 8,
                                               'ep': 8
                                           },
                                           model_type='chat_model',
                                           func_type='benchmark')
    assert len(benchmark_dpep8) == 2, len(benchmark_dpep8)

    mllm_benchmark_tp1 = get_func_config_list(backend,
                                              parallel_config={'tp': 1},
                                              model_type='chat_model',
                                              func_type='mllm_benchmark')
    assert len(mllm_benchmark_tp1) == 5, len(mllm_benchmark_tp1)
    mllm_longtext_tp2 = get_func_config_list(backend,
                                             parallel_config={'tp': 2},
                                             model_type='chat_model',
                                             func_type='mllm_longtext')
    assert len(mllm_longtext_tp2) == 0, len(mllm_longtext_tp2)
    mllm_evaluate_tp8 = get_func_config_list(backend,
                                             parallel_config={'tp': 8},
                                             model_type='chat_model',
                                             func_type='mllm_evaluate')
    assert len(mllm_evaluate_tp8) == 3, len(mllm_evaluate_tp8)
    mllm_evaluate_dpep16 = get_func_config_list(backend,
                                                parallel_config={
                                                    'dp': 16,
                                                    'ep': 16
                                                },
                                                model_type='chat_model',
                                                func_type='evaluate')
    assert len(mllm_evaluate_dpep16) == 3, len(mllm_evaluate_dpep16)
    mllm_benchmark_cptp = get_func_config_list(backend,
                                               parallel_config={
                                                   'cp': 2,
                                                   'tp': 8
                                               },
                                               model_type='chat_model',
                                               func_type='benchmark')
    assert len(mllm_benchmark_cptp) == 2, len(mllm_benchmark_cptp)


def test_run_config():
    os.environ['TEST_ENV'] = 'test'
    backend = 'turbomind'
    run_config1 = get_func_config_list(backend, parallel_config={'tp': 1}, model_type='chat_model', func_type='func')[0]
    assert run_config1['model'] == 'test/test_tp1'
    assert run_config1['backend'] == 'turbomind'
    assert run_config1['communicator'] == 'nccl'
    assert run_config1['quant_policy'] == 0
    assert run_config1['parallel_config'] == {'tp': 1}
    os.environ['TEST_ENV'] = 'testascend'
    backend = 'pytorch'
    run_config2 = get_func_config_list(backend, parallel_config={'tp': 1}, model_type='chat_model', func_type='func')[0]
    assert run_config2['model'] == 'test/test_tp1'
    assert run_config2['backend'] == 'pytorch'
    assert run_config2['communicator'] == 'hccl'
    assert run_config2['quant_policy'] == 0
    assert run_config2['parallel_config'] == {'tp': 1}
    run_config3 = get_func_config_list(backend,
                                       parallel_config={'tp': 1},
                                       model_type='chat_model',
                                       func_type='func',
                                       extra={
                                           'speculative_algorithm': 'eagle',
                                           'session_len': 1024
                                       })[0]
    assert run_config3['model'] == 'test/test_tp1'
    assert run_config3['backend'] == 'pytorch'
    assert run_config3['communicator'] == 'hccl'
    assert run_config3['quant_policy'] == 0
    assert run_config3['parallel_config'] == {'tp': 1}
    assert run_config3['speculative_algorithm'] == 'eagle'
    assert run_config3['session_len'] == 1024


test_cli_common_param()
test_run_config()
test_get_case_str_by_config()
test_return_info_pytorch()
test_config()
test_return_info_turbomind()
