import pytest


@pytest.mark.order(10)
@pytest.mark.lagent
@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize('model', ['internlm/internlm2_5-7b-chat'])
def test_repeat(config, model):
    from lagent.llms import INTERNLM2_META, LMDeployPipeline

    model = LMDeployPipeline(
        path='/'.join([config.get('model_path'), model]),
        meta_template=INTERNLM2_META,
        tp=1,
        top_k=40,
        top_p=0.8,
        temperature=1.2,
        stop_words=['<|im_end|>'],
        max_new_tokens=4096,
    )
    response_list = []
    for i in range(3):
        print(f'run_{i}：')
        response = model.chat([{
            'role':
            'user',
            'content':
            '已知$$z_{1}=1$$,$$z_{2}=\\text{i}$$,$$z_{3}=-1$$,$$z_{4}=-\\text{i}$$,顺次连结它们所表示的点,则所得图形围成的面积为（ ）\nA. $$\\dfrac{1}{4}$$\n B. $$\\dfrac{1}{2}$$\n C. $$1$$\n D. $$2$$\n\n'  # noqa: F401, E501
        }])
        print(response)
        response_list.append(response)
        assert len(response) > 10 and '$\\boxed' in response
    assert response_list[0] != response_list[1] and response_list[1] != response_list[2]
