# autotest case

We provide a autotest caseset to do regression.

## How to run testcases

Install required dependencies using the following command line:

```bash
python3 -m pip install -r requirements/test.txt
```

Run pytest command line with case filtering through -m flag. eg: `-m internlm_chat_7b` Filter cases related to internlm_chat_7b. The corresponding results will be stored in the `allure-results` directory.

```bash
pytest autotest -m internlm_chat_7b --clean-alluredir --alluredir=allure-results
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

## Preparation before testing

To improve the efficiency of test case execution, we have downloaded the hf model files to a specific path in advance for easy use in test cases. The path where the model files are stored is defined in the `autotest/config.yaml` file with parameter `model_path`.

Since the test cases involve converting the hf model using convert, the converted model storage path is defined in the `autotest/config.yaml` file parameter `dst_path`.

The `autotest/config.yaml` file also defines the supported model table and corresponding model categories, such as the `model_map` parameter, as well as the log storage path `log_path` used during test case execution.

If you want to create a test environment, you need to prepare the above content and modify the config.yaml file as needed.

## Test case functionality coverage

The test cases cover the following functionalities:

![image](https://github.com/InternLM/lmdeploy/assets/145004780/85d6a2d3-cc4f-459c-8dc1-22c17b69954f)

The relationship between functionalities and test cases is as follows:

|        Function         |          Test Case File           |
| :---------------------: | :-------------------------------: |
|   w4a16 quantization    |    test_order1_quantization_w4    |
|    w8a8 quantization    |   test_order1_quantization_w8a8   |
|         convert         |        test_order2_convert        |
|      pipeline chat      |     test_order3_pipeline_chat     |
| pipeline chat - pytorch | test_order3_pipeline_chat_pytorch |
|    restful_api chat     |     test_order3_restful_chat      |
|   command chat - cli    |     test_order3_command_chat      |
|    command chat - hf    |    test_order3_command_chat_hf    |
| command chat - pytorch  | test_order3_command_chat_pytorch  |

The modules and models currently covered by the test cases are listed below:

|                                   Models                                   | w4a16 quantization | w8a8 quantization | kvint8 quantization | convert | pipeline chat | pipeline chat - pytorch | restful_api chat | command chat - cli | command chat - hf | command chat - pytorch |
| :------------------------------------------------------------------------: | :----------------: | :---------------: | :-----------------: | :-----: | :-----------: | :---------------------: | :--------------: | :----------------: | :---------------: | :--------------------: |
|   [internlm2_chat_7b](https://huggingface.co/internlm/internlm2-chat-7b)   |         No         |        No         |         No          |   Yes   |      Yes      |           Yes           |        No        |        Yes         |        Yes        |          Yes           |
|  [internlm2_chat_20b](https://huggingface.co/internlm/internlm2-chat-20b)  |        Yes         |        Yes        |         No          |   Yes   |      Yes      |           No            |       Yes        |        Yes         |        Yes        |          Yes           |
|    [internlm_chat_7b](https://huggingface.co/internlm/internlm-chat-7b)    |         No         |        No         |         No          |   Yes   |      Yes      |           Yes           |       Yes        |        Yes         |        Yes        |           No           |
|   [internlm_chat_20b](https://huggingface.co/internlm/internlm-chat-20b)   |        Yes         |        No         |         No          |   Yes   |      Yes      |           No            |        No        |        Yes         |        Yes        |           No           |
|   [llama2_chat_7b_w4](https://huggingface.co/lmdeploy/llama2-chat-7b-w4)   |         No         |        No         |         No          |   Yes   |      Yes      |           No            |        No        |        Yes         |        Yes        |           No           |
|          [Qwen_7B_Chat](https://huggingface.co/Qwen/Qwen-7B-Chat)          |        Yes         |        No         |         No          |   Yes   |      Yes      |           No            |        No        |        Yes         |        Yes        |           No           |
|         [Qwen_14B_Chat](https://huggingface.co/Qwen/Qwen-14B-Chat)         |        Yes         |        No         |         No          |   Yes   |      Yes      |           No            |        No        |        Yes         |        Yes        |           No           |
| [Baichuan2_7B_Chat](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat) |        Yes         |        No         |         No          |   Yes   |      Yes      |           No            |        No        |        Yes         |        Yes        |           No           |
|    [llama_2_7b_chat](https://huggingface.co/meta-llama/Llama-2-7b-chat)    |        Yes         |        No         |         No          |   Yes   |      Yes      |           No            |        No        |        Yes         |        Yes        |           No           |

## How to add a testcase

you need to confirm that the corresponding model is ready <a href="##Preparation before testing">Jump to prepare Section</a>, then you can copy the existing case in the corresponding function test file. Please modify case mark, case story, case name and parameters if need.

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
