import os

import pytest
import yaml


@pytest.fixture(scope='session')
def config(request):
    config_path = os.path.join(request.config.rootdir, 'config.yaml')
    print(config_path)
    with open(config_path) as f:
        env_config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    return env_config
