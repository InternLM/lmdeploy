import pytest
import utils.constant as constant
from utils.config_utils import (
    get_case_str_by_config,
    get_eval_preset_config,
    get_model_path_from_config,
    get_workerid,
)
from utils.evaluate_utils import eval_test
from utils.pytest_layout_utils import BASE_TP_LAYOUTS, build_eval_stage_params
from utils.run_restful_chat import start_openai_service, terminate_restful_api

_BASE_EVAL = 'base'


def run_base_eval_test(config, run_config, worker_id, test_type='infer'):
    """Run base-model OpenCompass eval via TurboMindAPIModel + api_server.

    Points OpenCompass at the api_server directly (not the proxy): proxy does
    not forward ``/get_ppl`` or ``/v1/encode`` needed by TurboMindAPIModel.
    """
    preset_config = get_eval_preset_config(config, run_config, _BASE_EVAL)
    eval_path = config.get('eval_path')
    case_name = get_case_str_by_config(run_config)
    model_path = get_model_path_from_config(config, run_config.get('model'))
    extra_config = {'max-num-workers': 256}

    if test_type == 'infer':
        port = constant.DEFAULT_PORT + get_workerid(worker_id)
        pid, content = start_openai_service(config, run_config, worker_id)
        try:
            assert pid > 0, f'Failed to start RESTful API server: {content}'
            eval_test(model_path,
                      eval_path,
                      case_name,
                      port=port,
                      test_type=test_type,
                      extra_config=extra_config,
                      eval_config_name=_BASE_EVAL,
                      **preset_config)
        finally:
            if pid > 0:
                terminate_restful_api(worker_id)
    else:
        eval_test(model_path,
                  eval_path,
                  case_name,
                  port=constant.DEFAULT_PORT + get_workerid(worker_id),
                  test_type=test_type,
                  extra_config=extra_config,
                  eval_config_name=_BASE_EVAL,
                  **preset_config)


_BASE_EVAL_PARAMS = (
    build_eval_stage_params(
        'turbomind',
        BASE_TP_LAYOUTS,
        model_type='base_model',
        func_type='evaluate',
    )
    + build_eval_stage_params(
        'pytorch',
        BASE_TP_LAYOUTS,
        model_type='base_model',
        func_type='evaluate',
    )
)


@pytest.mark.parametrize('test_type, run_config', _BASE_EVAL_PARAMS)
def test_api_base_evaluate_local(config, run_config, worker_id, test_type):
    run_base_eval_test(config, run_config, worker_id, test_type)
