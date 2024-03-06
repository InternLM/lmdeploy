# autotest case

We provide a autotest caseset to do regression.

## Preparation before testing

To improve the efficiency of test case execution, we have downloaded the hf model files to a specific path in advance for easy use in test cases. The path where the model files are stored is defined in the `autotest/config.yaml` file with parameter `model_path`.

Since the test cases involve converting the hf model using convert, the converted model storage path is defined in the `autotest/config.yaml` file parameter `dst_path`.

The `autotest/config.yaml` file also defines the supported model table and corresponding model categories, such as the `model_map` parameter, as well as the log storage path `log_path` used during test case execution.

If you want to create a test environment, you need to prepare the above content and modify the config.yaml file as needed.

## How to run testcases

Install required dependencies using the following command line:

```bash
python3 -m pip install -r requirements/test.txt
```

Run pytest command line with case filtering through -m flag or folder name. eg: `-m convert` Filter cases related to convert or `autotest/tools/convert` for the case in the folder. The corresponding results will be stored in the `allure-results` directory.

```bash
pytest autotest -m convert --clean-alluredir --alluredir=allure-results
pytest autotest/tools/convert --clean-alluredir --alluredir=allure-results

```

If you need to generate reports and display report features, you need to install allure according to the [install documentation of allure](https://allurereport.org/docs/gettingstarted-installation/#install-via-the-system-package-manager-for-linux). You can also install it directly using the following command:

```bash
wget https://github.com/allure-framework/allure2/releases/download/2.25.0/allure_2.24.1-1_all.deb

sudo apt-get install -y openjdk-8-jre-headless
sudo dpkg -i ./allure_2.24.1-1_all.deb
```

Then generate the test report and view the corresponding HTML page by using the following command. The generated report will be stored in `allure-reports`.

```bash
allure generate -c -o allure-reports
allure open ./allure-reports
```

## Test case functionality coverage

The testcases are including following models:

tools model - related to tutorials, the case is basic

interface model - interface function cases of pipeline、 restful api and triton server api

The relationship between functionalities and test cases is as follows:

| case model |             Function             |                    Test Case File                    |
| :--------: | :------------------------------: | :--------------------------------------------------: |
|   tools    |       quantization - w4a16       |    tools/quantization/test_quantization_w4a16.py     |
|   tools    |       quantization - w8a8        |     tools/quantization/test_quantization_w8a8.py     |
|   tools    |      quantization - kv int8      |    tools/quantization/test_quantization_kvint8.py    |
|   tools    | quantization - kv int8 and w4a16 | tools/quantization/test_quantization_kvint8_w4a16.py |
|   tools    |             convert              |            tools/convert/test_convert.py             |
|   tools    |    pipeline chat - turbomind     |    tools/pipeline/test_pipeline_chat_turbomind.py    |
|   tools    |     pipeline chat - pytorch      |     tools/pipeline/test_pipeline_chat_pytorch.py     |
|   tools    |   restful_api chat - turbomind   |    tools/pipeline/test_restful_chat_turbomind.py     |
|   tools    |    restful_api chat - pytorch    |     tools/pipeline/test_restful_chat_pytorch.py      |
|   tools    |     command chat - workspace     |      tools/chat/test_command_chat_workspace.py       |
|   tools    |   command chat - hf turbomind    |     tools/chat/test_command_chat_hf_turbomind.py     |
|   tools    |    command chat - hf pytorch     |      tools/chat/test_command_chat_hf_pytorch.py      |
| interface  |    command chat - hf pytorch     |      tools/chat/test_command_chat_hf_pytorch.py      |

The modules and models currently covered by the turbomind and pytorch backend is in `autotest/config.yaml` by using turbomind_model and pytorch_model.

## How to add a testcase

If you want add a new model into tool testcase, you should repare the model in your machine <a href="##Preparation before testing">Jump to prepare Section</a> then add it into `autotest/config.yaml`.

## How to add a chatcase template

We have provided some basic cases in the YAML file for dialogue testing.
For CLI command usage with `chat_prompt_case.yaml` file, use `prompt_case.yaml` file for pipeline chat、 restful api and gradio testing.

If you want to add a dialogue case, you need to modify the corresponding YAML file.

The structure and logic of the YAML file are as follows:

```yaml

# casename, please name the case function, eg: This case is used to test whether there is memory ability for previous round information during multi-round dialogue.
memory_test:
    - please introduce some attractions in Chengdu: # Round 1 prompt
        # output assert rule list, all rules need to be satisfied for the case to pass.
        - contain: # The output needs to contain any one of the following items
            - chengdu
        - contain:
            - 熊猫
            - panda
        - llama2: # For specific models that require different assert logic, the key is the model type and the value is a list of assert rules. This is a example for llama2 model. In this case, other assert rules will become invalid.
            - len_g:
                10
    - please introduce some delicious foods: # Round 2 prompt
        # output assert info list
        - contain:
            - chengdu
        - len_g: # The output's length should larger then 10
            10
    - XXX: # Round 3 prompt

```
