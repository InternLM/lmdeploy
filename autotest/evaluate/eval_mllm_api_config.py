import copy as cp
import os
from functools import partial

from vlmeval.api import *  # noqa
from vlmeval.vlm import *  # noqa

PandaGPT_ROOT = None
MiniGPT4_ROOT = None
TransCore_ROOT = None
Yi_ROOT = None
OmniLMM_ROOT = None
Mini_Gemini_ROOT = None
VXVERSE_ROOT = None
VideoChat2_ROOT = None
VideoChatGPT_ROOT = None
PLLaVA_ROOT = None
RBDash_ROOT = None
VITA_ROOT = None
LLAVA_V1_7B_MODEL_PTH = 'Please set your local path to LLaVA-7B-v1.1 here, the model weight is obtained by merging LLaVA delta weight based on vicuna-7b-v1.1 in https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md with vicuna-7b-v1.1. '  # noqa

video_models = {
    'Video-LLaVA-7B':
    partial(VideoLLaVA, model_path='LanguageBind/Video-LLaVA-7B'),  # noqa
    'Video-LLaVA-7B-HF':
    partial(
        VideoLLaVA_HF,  # noqa
        model_path='LanguageBind/Video-LLaVA-7B-hf'  # noqa
    ),
    'VideoChat2-HD':
    partial(
        VideoChat2_HD,  # noqa
        model_path='OpenGVLab/VideoChat2_HD_stage4_Mistral_7B',
        root=VideoChat2_ROOT,
        config_file='./vlmeval/vlm/video_llm/configs/videochat2_hd.json',
    ),
    'Chat-UniVi-7B':
    partial(Chatunivi, model_path='Chat-UniVi/Chat-UniVi'),  # noqa
    'Chat-UniVi-7B-v1.5':
    partial(
        Chatunivi,  # noqa
        model_path='Chat-UniVi/Chat-UniVi-7B-v1.5'),
    'LLaMA-VID-7B':
    partial(
        LLaMAVID,  # noqa
        model_path='YanweiLi/llama-vid-7b-full-224-video-fps-1'),
    'Video-ChatGPT':
    partial(
        VideoChatGPT,  # noqa
        model_path='MBZUAI/Video-ChatGPT-7B',
        dir_root=VideoChatGPT_ROOT),
    'PLLaVA-7B':
    partial(PLLaVA, model_path='ermu2001/pllava-7b', dir_root=PLLaVA_ROOT),  # noqa
    'PLLaVA-13B':
    partial(
        PLLaVA,  # noqa
        model_path='ermu2001/pllava-13b',
        dir_root=PLLaVA_ROOT),
    'PLLaVA-34B':
    partial(
        PLLaVA,  # noqa
        model_path='ermu2001/pllava-34b',
        dir_root=PLLaVA_ROOT  # noqa
    ),
}

ungrouped = {
    'AKI':
    partial(AKI, name='AKI', ckpt_pth='Sony/AKI-4B-phi-3.5-mini'),  # noqa
    'TransCore_M':
    partial(TransCoreM, root=TransCore_ROOT),  # noqa
    'PandaGPT_13B':
    partial(PandaGPT, name='PandaGPT_13B', root=PandaGPT_ROOT),  # noqa
    'flamingov2':
    partial(
        OpenFlamingo,  # noqa
        name='v2',
        mpt_pth='anas-awadalla/mpt-7b',
        ckpt_pth='openflamingo/OpenFlamingo-9B-vitl-mpt7b',
    ),
    'VisualGLM_6b':
    partial(VisualGLM, model_path='THUDM/visualglm-6b'),  # noqa
    'mPLUG-Owl2':
    partial(mPLUG_Owl2, model_path='MAGAer13/mplug-owl2-llama2-7b'),  # noqa
    'mPLUG-Owl3':
    partial(mPLUG_Owl3, model_path='mPLUG/mPLUG-Owl3-7B-240728'),  # noqa
    'OmniLMM_12B':
    partial(
        OmniLMM12B,  # noqa
        model_path='openbmb/OmniLMM-12B',
        root=OmniLMM_ROOT  # noqa
    ),
    'MGM_7B':
    partial(
        Mini_Gemini,  # noqa
        model_path='YanweiLi/MGM-7B-HD',
        root=Mini_Gemini_ROOT  # noqa
    ),
    'Bunny-llama3-8B':
    partial(BunnyLLama3, model_path='BAAI/Bunny-v1_1-Llama-3-8B-V'),  # noqa
    'VXVERSE':
    partial(VXVERSE, model_name='XVERSE-V-13B', root=VXVERSE_ROOT),  # noqa
    '360VL-70B':
    partial(QH_360VL, model_path='qihoo360/360VL-70B'),  # noqa
    'Llama-3-MixSenseV1_1':
    partial(LLama3Mixsense, model_path='Zero-Vision/Llama-3-MixSenseV1_1'),  # noqa
    'Parrot':
    partial(Parrot, model_path='AIDC-AI/Parrot-7B'),  # noqa
    'OmChat':
    partial(OmChat, model_path='omlab/omchat-v2.0-13B-single-beta_hf'),  # noqa
    'RBDash_72b':
    partial(RBDash, model_path='RBDash-Team/RBDash-v1.5', root=RBDash_ROOT),  # noqa
    'Pixtral-12B':
    partial(Pixtral, model_path='mistralai/Pixtral-12B-2409'),  # noqa
    'Falcon2-VLM-11B':
    partial(Falcon2VLM, model_path='tiiuae/falcon-11B-vlm'),  # noqa
}

o1_key = os.environ.get('O1_API_KEY', None)
o1_base = os.environ.get('O1_API_BASE', None)
o1_apis = {
    'o1':
    partial(
        GPT4V,  # noqa
        model='o1-2024-12-17',
        key=o1_key,
        api_base=o1_base,
        temperature=0,
        img_detail='high',
        retry=3,
        timeout=1800,
        max_tokens=16384,
        verbose=False,
    ),
    'o3':
    partial(
        GPT4V,  # noqa
        model='o3-2025-04-16',
        key=o1_key,
        api_base=o1_base,
        temperature=0,
        img_detail='high',
        retry=3,
        timeout=1800,
        max_tokens=16384,
        verbose=False,
    ),
    'o4-mini':
    partial(
        GPT4V,  # noqa
        model='o4-mini-2025-04-16',
        key=o1_key,
        api_base=o1_base,
        temperature=0,
        img_detail='high',
        retry=3,
        timeout=1800,
        max_tokens=16384,
        verbose=False,
    ),
}

api_models = {
    # GPT
    'GPT4V':
    partial(
        GPT4V,  # noqa
        model='gpt-4-1106-vision-preview',
        temperature=0,
        img_size=512,
        img_detail='low',
        retry=10,
        verbose=False,
    ),
    'GPT4V_HIGH':
    partial(
        GPT4V,  # noqa
        model='gpt-4-1106-vision-preview',
        temperature=0,
        img_size=-1,
        img_detail='high',
        retry=10,
        verbose=False,
    ),
    'GPT4V_20240409':
    partial(
        GPT4V,  # noqa
        model='gpt-4-turbo-2024-04-09',
        temperature=0,
        img_size=512,
        img_detail='low',
        retry=10,
        verbose=False,
    ),
    'GPT4V_20240409_HIGH':
    partial(
        GPT4V,  # noqa
        model='gpt-4-turbo-2024-04-09',
        temperature=0,
        img_size=-1,
        img_detail='high',
        retry=10,
        verbose=False,
    ),
    'GPT4o':
    partial(
        GPT4V,  # noqa
        model='gpt-4o-2024-05-13',
        temperature=0,
        img_size=512,
        img_detail='low',
        retry=10,
        verbose=False,
    ),
    'GPT4o_HIGH':
    partial(
        GPT4V,  # noqa
        model='gpt-4o-2024-05-13',
        temperature=0,
        img_size=-1,
        img_detail='high',
        retry=10,
        verbose=False,
    ),
    'GPT4o_20240806':
    partial(
        GPT4V,  # noqa
        model='gpt-4o-2024-08-06',
        temperature=0,
        img_size=-1,
        img_detail='high',
        retry=10,
        verbose=False,
    ),
    'GPT4o_20241120':
    partial(
        GPT4V,  # noqa
        model='gpt-4o-2024-11-20',
        temperature=0,
        img_size=-1,
        img_detail='high',
        retry=10,
        verbose=False,
    ),
    'ChatGPT4o':
    partial(
        GPT4V,  # noqa
        model='chatgpt-4o-latest',
        temperature=0,
        img_size=-1,
        img_detail='high',
        retry=10,
        verbose=False,
    ),
    'GPT4o_MINI':
    partial(
        GPT4V,  # noqa
        model='gpt-4o-mini-2024-07-18',
        temperature=0,
        img_size=-1,
        img_detail='high',
        retry=10,
        verbose=False,
    ),
    'GPT4.5':
    partial(
        GPT4V,  # noqa
        model='gpt-4.5-preview-2025-02-27',
        temperature=0,
        timeout=600,
        img_size=-1,
        img_detail='high',
        retry=10,
        verbose=False,
    ),
    'gpt-4.1-2025-04-14':
    partial(
        GPT4V,  # noqa
        model='gpt-4.1-2025-04-14',
        temperature=0,
        img_size=-1,
        img_detail='high',
        retry=10,
        verbose=False,
    ),
    'gpt-4.1-mini-2025-04-14':
    partial(
        GPT4V,  # noqa
        model='gpt-4.1-mini-2025-04-14',
        temperature=0,
        img_size=-1,
        img_detail='high',
        retry=10,
        verbose=False,
    ),
    'gpt-4.1-nano-2025-04-14':
    partial(
        GPT4V,  # noqa
        model='gpt-4.1-nano-2025-04-14',
        temperature=0,
        img_size=-1,
        img_detail='high',
        retry=10,
        verbose=False,
    ),
    'gpt-5-2025-08-07':
    partial(
        GPT4V,  # noqa
        model='gpt-5-2025-08-07',
        img_detail='high',
        retry=3,
        verbose=False,
        max_tokens=2**14,
        timeout=300,
    ),
    'gpt-5-mini-2025-08-07':
    partial(
        GPT4V,  # noqa
        model='gpt-5-mini-2025-08-07',
        img_detail='high',
        retry=3,
        verbose=False,
        max_tokens=2**14,
        timeout=300,
    ),
    'gpt-5-nano-2025-08-07':
    partial(
        GPT4V,  # noqa
        model='gpt-5-nano-2025-08-07',
        img_detail='high',
        retry=3,
        verbose=False,
        max_tokens=2**14,
        timeout=300,
    ),
    # Gemini
    'GeminiPro1-0':
    partial(
        Gemini,  # noqa
        model='gemini-1.0-pro',
        temperature=0,
        retry=10),  # now GeminiPro1-0 is only supported by vertex backend
    'GeminiPro1-5':
    partial(
        Gemini,  # noqa
        model='gemini-1.5-pro',
        temperature=0,
        retry=10),
    'GeminiFlash1-5':
    partial(
        Gemini,  # noqa
        model='gemini-1.5-flash',
        temperature=0,
        retry=10),
    'GeminiPro1-5-002':
    partial(
        GPT4V,  # noqa
        model='gemini-1.5-pro-002',
        temperature=0,
        retry=10),  # Internal Use Only
    'GeminiFlash1-5-002':
    partial(
        GPT4V,  # noqa
        model='gemini-1.5-flash-002',
        temperature=0,
        retry=10),  # Internal Use Only
    'GeminiFlash2-0':
    partial(
        Gemini,  # noqa
        model='gemini-2.0-flash',
        temperature=0,
        retry=10),
    'GeminiFlashLite2-0':
    partial(
        Gemini,  # noqa
        model='gemini-2.0-flash-lite',
        temperature=0,
        retry=10),
    'GeminiFlash2-5':
    partial(
        Gemini,  # noqa
        model='gemini-2.5-flash',
        temperature=0,
        retry=10),
    'GeminiPro2-5':
    partial(
        Gemini,  # noqa
        model='gemini-2.5-pro',
        temperature=0,
        retry=10),

    # Qwen-VL
    'QwenVLPlus':
    partial(QwenVLAPI, model='qwen-vl-plus', temperature=0, retry=10),  # noqa
    'QwenVLMax':
    partial(QwenVLAPI, model='qwen-vl-max', temperature=0, retry=10),  # noqa
    'QwenVLMax-250408':
    partial(QwenVLAPI, model='qwen-vl-max-2025-04-08', temperature=0, retry=10),  # noqa

    # Reka
    'RekaEdge':
    partial(Reka, model='reka-edge-20240208'),  # noqa
    'RekaFlash':
    partial(Reka, model='reka-flash-20240226'),  # noqa
    'RekaCore':
    partial(Reka, model='reka-core-20240415'),  # noqa
    # Step1V
    'Step1V':
    partial(
        GPT4V,  # noqa
        model='step-1v-32k',
        api_base='https://api.stepfun.com/v1/chat/completions',
        temperature=0,
        retry=10,
        img_size=-1,
        img_detail='high',
    ),
    'Step1.5V-mini':
    partial(
        GPT4V,  # noqa
        model='step-1.5v-mini',
        api_base='https://api.stepfun.com/v1/chat/completions',
        temperature=0,
        retry=10,
        img_size=-1,
        img_detail='high',
    ),
    'Step1o':
    partial(
        GPT4V,  # noqa
        model='step-1o-vision-32k',
        api_base='https://api.stepfun.com/v1/chat/completions',
        temperature=0,
        retry=10,
        img_size=-1,
        img_detail='high',
    ),
    # Yi-Vision
    'Yi-Vision':
    partial(
        GPT4V,  # noqa
        model='yi-vision',
        api_base='https://api.lingyiwanwu.com/v1/chat/completions',
        temperature=0,
        retry=10,
    ),
    # Claude
    'Claude3V_Opus':
    partial(
        Claude3V,  # noqa model="claude-3-opus-20240229", temperature=0, retry=10, verbose=False
    ),
    'Claude3V_Sonnet':
    partial(
        Claude3V,  # noqa
        model='claude-3-sonnet-20240229',
        temperature=0,
        retry=10,
        verbose=False,
    ),
    'Claude3V_Haiku':
    partial(
        Claude3V,  # noqa
        model='claude-3-haiku-20240307',
        temperature=0,
        retry=10,
        verbose=False,
    ),
    'Claude3-5V_Sonnet':
    partial(
        Claude3V,  # noqa
        model='claude-3-5-sonnet-20240620',
        temperature=0,
        retry=10,
        verbose=False,
    ),
    'Claude3-5V_Sonnet_20241022':
    partial(
        Claude3V,  # noqa
        model='claude-3-5-sonnet-20241022',
        temperature=0,
        retry=10,
        verbose=False,
    ),
    'Claude3-7V_Sonnet':
    partial(
        Claude3V,  # noqa
        model='claude-3-7-sonnet-20250219',
        temperature=0,
        retry=10,
        verbose=False,
    ),
    'Claude4_Opus':
    partial(
        Claude3V,  # noqa
        model='claude-4-opus-20250514',
        temperature=0,
        retry=10,
        verbose=False,
        timeout=1800),
    'Claude4_Sonnet':
    partial(
        Claude3V,  # noqa
        model='claude-4-sonnet-20250514',
        temperature=0,
        retry=10,
        verbose=False,
        timeout=1800),
    # GLM4V
    'GLM4V':
    partial(GLMVisionAPI, model='glm4v-biz-eval', temperature=0, retry=10),  # noqa
    'GLM4V_PLUS':
    partial(GLMVisionAPI, model='glm-4v-plus', temperature=0, retry=10),  # noqa
    'GLM4V_PLUS_20250111':
    partial(GLMVisionAPI, model='glm-4v-plus-0111', temperature=0, retry=10),  # noqa
    # MiniMax abab
    'abab6.5s':
    partial(
        GPT4V,  # noqa
        model='abab6.5s-chat',
        api_base='https://api.minimax.chat/v1/chat/completions',
        temperature=0,
        retry=10,
    ),
    'abab7-preview':
    partial(
        GPT4V,  # noqa
        model='abab7-chat-preview',
        api_base='https://api.minimax.chat/v1/chat/completions',
        temperature=0,
        retry=10,
    ),
    # CongRong
    'CongRong-v1.5':
    partial(CWWrapper, model='cw-congrong-v1.5', temperature=0, retry=10),  # noqa
    'CongRong-v2.0':
    partial(CWWrapper, model='cw-congrong-v2.0', temperature=0, retry=10),  # noqa
    # SenseNova
    'SenseNova-V6-Pro':
    partial(SenseChatVisionAPI, model='SenseNova-V6-Pro', temperature=0, retry=10),  # noqa
    'SenseNova-V6-Reasoner':
    partial(SenseChatVisionAPI, model='SenseNova-V6-Reasoner', temperature=0, retry=10),  # noqa
    'SenseNova-V6-5-Pro':
    partial(SenseChatVisionAPI, model='SenseNova-V6-5-Pro', retry=10),  # noqa
    'HunYuan-Vision':
    partial(HunyuanVision, model='hunyuan-vision', temperature=0, retry=10),  # noqa
    'HunYuan-Standard-Vision':
    partial(HunyuanVision, model='hunyuan-standard-vision', temperature=0, retry=10),  # noqa
    'HunYuan-Large-Vision':
    partial(HunyuanVision, model='hunyuan-large-vision', temperature=0, retry=10),  # noqa
    'BailingMM-Lite-1203':
    partial(bailingMMAPI, model='BailingMM-Lite-1203', temperature=0, retry=10),  # noqa
    'BailingMM-Pro-0120':
    partial(bailingMMAPI, model='BailingMM-Pro-0120', temperature=0, retry=10),  # noqa
    # BlueLM-2.5
    'BlueLM-2.5-3B':
    partial(BlueLM_API, model='BlueLM-2.5-3B', temperature=0, retry=3),  # noqa
    # JiuTian-VL
    'JTVL':
    partial(JTVLChatAPI, model='jt-vl-chat', temperature=0, retry=10),  # noqa
    'JTVL-Mini':
    partial(JTVLChatAPI_Mini, model='jt-vl-chat-mini', temperature=0, retry=10),  # noqa
    'JTVL-2B':
    partial(JTVLChatAPI_2B, model='jt-vl-chat-2b', temperature=0, retry=10),  # noqa
    'Taiyi':
    partial(TaiyiAPI, model='taiyi', temperature=0, retry=10),  # noqa
    # TeleMM
    'TeleMM':
    partial(TeleMMAPI, model='TeleAI/TeleMM', temperature=0, retry=10),  # noqa
    'Qwen2.5-VL-32B-Instruct-SiliconFlow':
    partial(SiliconFlowAPI, model='Qwen/Qwen2.5-VL-32B-Instruct', temperature=0, retry=10),  # noqa
    # lmdeploy api
    'lmdeploy_port23333':
    partial(
        LMDeployAPI,  # noqa
        api_base='http://0.0.0.0:23333/v1/chat/completions',
        temperature=0,
        retry=10,
    ),
    'lmdeploy_port23334':
    partial(
        LMDeployAPI,  # noqa
        api_base='http://0.0.0.0:23333/v1/chat/completions',
        temperature=0,
        retry=10,
        timeout=100,
    ),
    'lmdeploy_port23335':
    partial(
        LMDeployAPI,  # noqa
        api_base='http://0.0.0.0:23333/v1/chat/completions',
        temperature=0,
        retry=10,
        timeout=300,
    ),
    'lmdeploy_port23336':
    partial(
        LMDeployAPI,  # noqa
        api_base='http://0.0.0.0:23333/v1/chat/completions',
        temperature=0,
        retry=10,
        timeout=300,
    ),
    'lmdeploy_port23337':
    partial(
        LMDeployAPI,  # noqa
        api_base='http://0.0.0.0:23333/v1/chat/completions',
        temperature=0,
        retry=10,
        timeout=300,
    ),
    'lmdeploy_port23338':
    partial(
        LMDeployAPI,  # noqa
        api_base='http://0.0.0.0:23333/v1/chat/completions',
        temperature=0,
        retry=10,
        timeout=300,
    ),
    'lmdeploy_port23339':
    partial(
        LMDeployAPI,  # noqa
        api_base='http://0.0.0.0:23333/v1/chat/completions',
        temperature=0,
        retry=10,
        timeout=300,
    ),
    'lmdeploy_port23340':
    partial(
        LMDeployAPI,  # noqa
        api_base='http://0.0.0.0:23333/v1/chat/completions',
        temperature=0,
        retry=10,
        timeout=300,
    ),
    'lmdeploy_internvl_78B_MPO':
    partial(
        LMDeployAPI,  # noqa
        api_base='http://0.0.0.0:23333/v1/chat/completions',
        temperature=0,
        retry=10,
        timeout=100,
    ),
    'lmdeploy_qvq_72B_preview':
    partial(
        LMDeployAPI,  # noqa
        api_base='http://0.0.0.0:23333/v1/chat/completions',
        temperature=0,
        retry=10,
        timeout=300,
    ),
    'Taichu-VLR-3B':
    partial(
        TaichuVLRAPI,  # noqa
        model='taichu_vlr_3b',
        url='https://platform.wair.ac.cn/maas/v1/chat/completions'),
    'Taichu-VLR-7B':
    partial(
        TaichuVLRAPI,  # noqa
        model='taichu_vlr_7b',
        url='https://platform.wair.ac.cn/maas/v1/chat/completions'),
    # doubao_vl
    'DoubaoVL':
    partial(
        DoubaoVL,  # noqa
        model='Doubao-1.5-vision-pro',
        temperature=0,
        retry=3,
        verbose=False  # noqa
    ),
    'Seed1.5-VL':
    partial(
        DoubaoVL,  # noqa
        model='doubao-1-5-thinking-vision-pro-250428',
        temperature=0,
        retry=3,
        verbose=False,
        max_tokens=16384,
    ),
    'Seed1.6':
    partial(
        DoubaoVL,  # noqa
        model='doubao-seed-1.6-250615',
        temperature=0,
        retry=3,
        verbose=False,
        max_tokens=16384,
    ),
    'Seed1.6-Flash':
    partial(
        DoubaoVL,  # noqa
        model='doubao-seed-1.6-flash-250615',
        temperature=0,
        retry=3,
        verbose=False,
        max_tokens=16384,
    ),
    'Seed1.6-Thinking':
    partial(
        DoubaoVL,  # noqa
        model='doubao-seed-1.6-thinking-250615',
        temperature=0,
        retry=3,
        verbose=False,
        max_tokens=16384,
    ),
    # Shopee MUG-U
    'MUG-U-7B':
    partial(
        MUGUAPI,  # noqa
        model='MUG-U',
        temperature=0,
        retry=10,
        verbose=False,
        timeout=300),
    # grok
    'grok-vision-beta':
    partial(
        GPT4V,  # noqa
        model='grok-vision-beta',
        api_base='https://api.x.ai/v1/chat/completions',
        temperature=0,
        retry=10,
    ),
    'grok-2-vision-1212':
    partial(
        GPT4V,  # noqa
        model='grok-2-vision',
        api_base='https://api.x.ai/v1/chat/completions',
        temperature=0,
        retry=10,
    ),
    'grok-4-0709':
    partial(
        GPT4V,  # noqa
        model='grok-4-0709',
        api_base='https://api.x.ai/v1/chat/completions',
        temperature=0,
        retry=3,
        timeout=1200,
        max_tokens=16384),
    # kimi
    'moonshot-v1-8k':
    partial(
        GPT4V,  # noqa
        model='moonshot-v1-8k-vision-preview',
        api_base='https://api.moonshot.cn/v1/chat/completions',
        temperature=0,
        retry=10,
    ),
    'moonshot-v1-32k':
    partial(
        GPT4V,  # noqa
        model='moonshot-v1-32k-vision-preview',
        api_base='https://api.moonshot.cn/v1/chat/completions',
        temperature=0,
        retry=10,
    ),
    'moonshot-v1-128k':
    partial(
        GPT4V,  # noqa
        model='moonshot-v1-128k-vision-preview',
        api_base='https://api.moonshot.cn/v1/chat/completions',
        temperature=0,
        retry=10,
    ),
    'ernie4.5-turbo':
    partial(
        GPT4V,  # noqa
        model='ernie-4.5-turbo-vl-32k',
        temperature=0,
        retry=3,
        max_tokens=12000,
    ),
    'ernie4.5-a3b':
    partial(
        GPT4V,  # noqa
        model='ernie-4.5-vl-28b-a3b',
        temperature=0,
        retry=3,
        max_tokens=8000,
    )
}

api_models['gpt-5'] = cp.deepcopy(api_models['gpt-5-2025-08-07'])
api_models['gpt-5-mini'] = cp.deepcopy(api_models['gpt-5-mini-2025-08-07'])
api_models['gpt-5-nano'] = cp.deepcopy(api_models['gpt-5-nano-2025-08-07'])

emu_series = {
    'emu2_chat': partial(Emu, model_path='BAAI/Emu2-Chat'),  # noqa
    'emu3_chat': partial(Emu3_chat, model_path='BAAI/Emu3-Chat'),  # noqa
    'emu3_gen': partial(Emu3_gen, model_path='BAAI/Emu3-Gen'),  # noqa
}

granite_vision_series = {
    'granite_vision_3.1_2b_preview': partial(
        GraniteVision3,  # noqa
        model_path='ibm-granite/granite-vision-3.1-2b-preview'),  # noqa
    'granite_vision_3.2_2b': partial(GraniteVision3, model_path='ibm-granite/granite-vision-3.2-2b'),  # noqa
    'granite_vision_3.3_2b': partial(GraniteVision3, model_path='ibm-granite/granite-vision-3.3-2b'),  # noqa
}

mmalaya_series = {
    'MMAlaya': partial(MMAlaya, model_path='DataCanvas/MMAlaya'),  # noqa
    'MMAlaya2': partial(MMAlaya2, model_path='DataCanvas/MMAlaya2'),  # noqa
}

minicpm_series = {
    'MiniCPM-V': partial(MiniCPM_V, model_path='openbmb/MiniCPM-V'),  # noqa
    'MiniCPM-V-2': partial(MiniCPM_V, model_path='openbmb/MiniCPM-V-2'),  # noqa
    'MiniCPM-Llama3-V-2_5': partial(
        MiniCPM_Llama3_V,  # noqa
        model_path='openbmb/MiniCPM-Llama3-V-2_5'  # noqa
    ),
    'MiniCPM-V-2_6': partial(MiniCPM_V_2_6, model_path='openbmb/MiniCPM-V-2_6'),  # noqa
    'MiniCPM-o-2_6': partial(MiniCPM_o_2_6, model_path='openbmb/MiniCPM-o-2_6'),  # noqa
    'MiniCPM-V-4': partial(MiniCPM_V_4, model_path='openbmb/MiniCPM-V-4'),  # noqa
    'MiniCPM-V-4_5': partial(MiniCPM_V_4_5, model_path='openbmb/MiniCPM-V-4_5'),  # noqa
}

xtuner_series = {
    'llava-internlm2-7b':
    partial(
        LLaVA_XTuner,  # noqa
        llm_path='internlm/internlm2-chat-7b',
        llava_path='xtuner/llava-internlm2-7b',
        visual_select_layer=-2,
        prompt_template='internlm2_chat',
    ),
    'llava-internlm2-20b':
    partial(
        LLaVA_XTuner,  # noqa
        llm_path='internlm/internlm2-chat-20b',
        llava_path='xtuner/llava-internlm2-20b',
        visual_select_layer=-2,
        prompt_template='internlm2_chat',
    ),
    'llava-internlm-7b':
    partial(
        LLaVA_XTuner,  # noqa
        llm_path='internlm/internlm-chat-7b',
        llava_path='xtuner/llava-internlm-7b',
        visual_select_layer=-2,
        prompt_template='internlm_chat',
    ),
    'llava-v1.5-7b-xtuner':
    partial(
        LLaVA_XTuner,  # noqa
        llm_path='lmsys/vicuna-7b-v1.5',
        llava_path='xtuner/llava-v1.5-7b-xtuner',
        visual_select_layer=-2,
        prompt_template='vicuna',
    ),
    'llava-v1.5-13b-xtuner':
    partial(
        LLaVA_XTuner,  # noqa
        llm_path='lmsys/vicuna-13b-v1.5',
        llava_path='xtuner/llava-v1.5-13b-xtuner',
        visual_select_layer=-2,
        prompt_template='vicuna',
    ),
    'llava-llama-3-8b':
    partial(
        LLaVA_XTuner,  # noqa
        llm_path='xtuner/llava-llama-3-8b-v1_1',
        llava_path='xtuner/llava-llama-3-8b-v1_1',
        visual_select_layer=-2,
        prompt_template='llama3_chat',
    ),
}

qwen_series = {
    'qwen_base': partial(QwenVL, model_path='Qwen/Qwen-VL'),  # noqa
    'qwen_chat': partial(QwenVLChat, model_path='Qwen/Qwen-VL-Chat'),  # noqa
    'monkey': partial(Monkey, model_path='echo840/Monkey'),  # noqa
    'monkey-chat': partial(MonkeyChat, model_path='echo840/Monkey-Chat'),  # noqa
    'minimonkey': partial(MiniMonkey, model_path='mx262/MiniMonkey'),  # noqa
}

thyme_series = {
    'Thyme-7B': partial(Thyme, model_path='Kwai-Keye/Thyme-RL')  # noqa
}

llava_series = {
    'llava_v1.5_7b':
    partial(LLaVA, model_path='liuhaotian/llava-v1.5-7b'),  # noqa
    'llava_v1.5_13b':
    partial(LLaVA, model_path='liuhaotian/llava-v1.5-13b'),  # noqa
    'llava_v1_7b':
    partial(LLaVA, model_path=LLAVA_V1_7B_MODEL_PTH),  # noqa
    'sharegpt4v_7b':
    partial(LLaVA, model_path='Lin-Chen/ShareGPT4V-7B'),  # noqa
    'sharegpt4v_13b':
    partial(LLaVA, model_path='Lin-Chen/ShareGPT4V-13B'),  # noqa
    'llava_next_vicuna_7b':
    partial(
        LLaVA_Next,  # noqa
        model_path='llava-hf/llava-v1.6-vicuna-7b-hf'  # noqa
    ),
    'llava_next_vicuna_13b':
    partial(
        LLaVA_Next,  # noqa
        model_path='llava-hf/llava-v1.6-vicuna-13b-hf'  # noqa
    ),
    'llava_next_mistral_7b':
    partial(
        LLaVA_Next,  # noqa
        model_path='llava-hf/llava-v1.6-mistral-7b-hf'  # noqa
    ),
    'llava_next_yi_34b':
    partial(LLaVA_Next, model_path='llava-hf/llava-v1.6-34b-hf'),  # noqa
    'llava_next_llama3':
    partial(
        LLaVA_Next,  # noqa
        model_path='llava-hf/llama3-llava-next-8b-hf'  # noqa
    ),
    'llava_next_72b':
    partial(LLaVA_Next, model_path='llava-hf/llava-next-72b-hf'),  # noqa
    'llava_next_110b':
    partial(LLaVA_Next, model_path='llava-hf/llava-next-110b-hf'),  # noqa
    'llava_next_qwen_32b':
    partial(
        LLaVA_Next2,  # noqa
        model_path='lmms-lab/llava-next-qwen-32b'  # noqa
    ),
    'llava_next_interleave_7b':
    partial(
        LLaVA_Next,  # noqa
        model_path='llava-hf/llava-interleave-qwen-7b-hf'  # noqa
    ),
    'llava_next_interleave_7b_dpo':
    partial(
        LLaVA_Next,  # noqa
        model_path='llava-hf/llava-interleave-qwen-7b-dpo-hf'  # noqa
    ),
    'llava-onevision-qwen2-0.5b-ov-hf':
    partial(
        LLaVA_OneVision_HF,  # noqa
        model_path='llava-hf/llava-onevision-qwen2-0.5b-ov-hf'  # noqa
    ),
    'llava-onevision-qwen2-0.5b-si-hf':
    partial(
        LLaVA_OneVision_HF,  # noqa
        model_path='llava-hf/llava-onevision-qwen2-0.5b-si-hf'  # noqa
    ),
    'llava-onevision-qwen2-7b-ov-hf':
    partial(
        LLaVA_OneVision_HF,  # noqa
        model_path='llava-hf/llava-onevision-qwen2-7b-ov-hf'  # noqa
    ),
    'llava-onevision-qwen2-7b-si-hf':
    partial(
        LLaVA_OneVision_HF,  # noqa
        model_path='llava-hf/llava-onevision-qwen2-7b-si-hf'  # noqa
    ),
    'llava_onevision_qwen2_0.5b_si':
    partial(
        LLaVA_OneVision,  # noqa
        model_path='lmms-lab/llava-onevision-qwen2-0.5b-si'  # noqa
    ),
    'llava_onevision_qwen2_7b_si':
    partial(
        LLaVA_OneVision,  # noqa
        model_path='lmms-lab/llava-onevision-qwen2-7b-si'  # noqa
    ),
    'llava_onevision_qwen2_72b_si':
    partial(
        LLaVA_OneVision,  # noqa
        model_path='lmms-lab/llava-onevision-qwen2-72b-si'  # noqa
    ),
    'llava_onevision_qwen2_0.5b_ov':
    partial(
        LLaVA_OneVision,  # noqa
        model_path='lmms-lab/llava-onevision-qwen2-0.5b-ov'  # noqa
    ),
    'llava_onevision_qwen2_7b_ov':
    partial(
        LLaVA_OneVision,  # noqa
        model_path='lmms-lab/llava-onevision-qwen2-7b-ov'  # noqa
    ),
    'llava_onevision_qwen2_72b_ov':
    partial(
        LLaVA_OneVision,  # noqa
        model_path='lmms-lab/llava-onevision-qwen2-72b-ov-sft'  # noqa
    ),
    'Aquila-VL-2B':
    partial(LLaVA_OneVision, model_path='BAAI/Aquila-VL-2B-llava-qwen'),  # noqa
    'llava_video_qwen2_7b':
    partial(
        LLaVA_OneVision,  # noqa
        model_path='lmms-lab/LLaVA-Video-7B-Qwen2'  # noqa
    ),
    'llava_video_qwen2_72b':
    partial(
        LLaVA_OneVision,  # noqa
        model_path='lmms-lab/LLaVA-Video-72B-Qwen2'  # noqa
    ),
}

varco_vision_series = {
    'varco-vision-hf': partial(
        LLaVA_OneVision_HF,  # noqa
        model_path='NCSOFT/VARCO-VISION-14B-HF'  # noqa
    ),
    'varco-vision-2-1.7b': partial(
        VarcoVision,  # noqa
        model_path='NCSOFT/VARCO-VISION-2.0-1.7B'  # noqa
    ),
    'varco-vision-2-14b': partial(
        VarcoVision,  # noqa
        model_path='NCSOFT/VARCO-VISION-2.0-14B'  # noqa
    ),
}

vita_series = {
    'vita': partial(VITA, model_path='VITA-MLLM/VITA', root=VITA_ROOT),  # noqa
    'vita_qwen2': partial(VITAQwen2, model_path='VITA-MLLM/VITA-1.5', root=VITA_ROOT),  # noqa
}

long_vita_series = {
    'Long-VITA-16K': partial(
        LongVITA,  # noqa
        model_path='VITA-MLLM/Long-VITA-16K_HF',
        max_num_frame=128  # noqa
    ),
    'Long-VITA-128K': partial(
        LongVITA,  # noqa
        model_path='VITA-MLLM/Long-VITA-128K_HF',
        max_num_frame=256  # noqa
    ),
    'Long-VITA-1M': partial(
        LongVITA,  # noqa
        model_path='VITA-MLLM/Long-VITA-1M_HF',
        max_num_frame=256  # noqa
    ),
}

interns1_mini = {
    'Intern-S1-mini':
    partial(
        InternS1Chat,  # noqa
        model_path='/mnt/shared-storage-user/mllm/lijinsong/models/Intern-S1-mini/'  # noqa
    ),
}

internvl = {
    'InternVL-Chat-V1-1': partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL-Chat-V1-1',
        version='V1.1'  # noqa
    ),
    'InternVL-Chat-V1-2': partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL-Chat-V1-2',
        version='V1.2'  # noqa
    ),
    'InternVL-Chat-V1-2-Plus': partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL-Chat-V1-2-Plus',
        version='V1.2'  # noqa
    ),
    'InternVL-Chat-V1-5': partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL-Chat-V1-5',  # noqa
        version='V1.5',
    )
}

mini_internvl = {
    'Mini-InternVL-Chat-2B-V1-5':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/Mini-InternVL-Chat-2B-V1-5',
        version='V1.5'  # noqa
    ),
    'Mini-InternVL-Chat-4B-V1-5':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/Mini-InternVL-Chat-4B-V1-5',
        version='V1.5'  # noqa
    ),
}

internvl2 = {
    'InternVL2-1B':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL2-1B',
        version='V2.0'  # noqa
    ),
    'InternVL2-2B':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL2-2B',
        version='V2.0'  # noqa
    ),
    'InternVL2-4B':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL2-4B',
        version='V2.0'  # noqa
    ),
    'InternVL2-8B':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL2-8B',
        version='V2.0'  # noqa
    ),
    'InternVL2-26B':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL2-26B',
        version='V2.0'  # noqa
    ),
    'InternVL2-40B':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL2-40B',
        version='V2.0'  # noqa
    ),
    'InternVL2-76B':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL2-Llama3-76B',
        version='V2.0'  # noqa
    ),
    'InternVL2-8B-MPO':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL2-8B-MPO',
        version='V2.0'  # noqa
    ),
    'InternVL2-8B-MPO-CoT':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL2-8B-MPO',
        version='V2.0',
        use_mpo_prompt=True,
    ),
}

internvl2_5 = {
    'InternVL2_5-1B':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL2_5-1B',
        version='V2.0'  # noqa
    ),
    'InternVL2_5-2B':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL2_5-2B',
        version='V2.0'  # noqa
    ),
    'QTuneVL1-2B':
    partial(
        InternVLChat,  # noqa
        model_path='hanchaow/QTuneVL1-2B',
        version='V2.0'  # noqa
    ),
    'InternVL2_5-4B':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL2_5-4B',
        version='V2.0'  # noqa
    ),
    'InternVL2_5-8B':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL2_5-8B',
        version='V2.0'  # noqa
    ),
    'InternVL2_5-26B':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL2_5-26B',
        version='V2.0'  # noqa
    ),
    'InternVL2_5-38B':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL2_5-38B',
        version='V2.0'  # noqa
    ),
    'InternVL2_5-78B':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL2_5-78B',
        version='V2.0'  # noqa
    ),
    # InternVL2.5 series with Best-of-N evaluation
    'InternVL2_5-8B-BoN-8':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL2_5-8B',
        version='V2.0',  # noqa
        best_of_n=8,
        reward_model_path='OpenGVLab/VisualPRM-8B',
    ),
}

internvl2_5_mpo = {
    'InternVL2_5-1B-MPO':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL2_5-1B-MPO',
        version='V2.0',
        use_mpo_prompt=True,
    ),
    'InternVL2_5-2B-MPO':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL2_5-2B-MPO',
        version='V2.0',
        use_mpo_prompt=True,
    ),
    'InternVL2_5-4B-MPO':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL2_5-4B-MPO',
        version='V2.0',
        use_mpo_prompt=True,
    ),
    'InternVL2_5-8B-MPO':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL2_5-8B-MPO',
        version='V2.0',
        use_mpo_prompt=True,
    ),
    'InternVL2_5-26B-MPO':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL2_5-26B-MPO',
        version='V2.0',
        use_mpo_prompt=True,
    ),
    'InternVL2_5-38B-MPO':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL2_5-38B-MPO',
        version='V2.0',
        use_mpo_prompt=True,
    ),
    'InternVL2_5-78B-MPO':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL2_5-78B-MPO',
        version='V2.0',
        use_mpo_prompt=True,
    ),
    'InternVL2_5-8B-GUI':
    partial(
        InternVLChat,  # noqa
        model_path='/fs-computility/mllm1/shared/zhaoxiangyu/models/internvl2_5_8b_internlm2_5_7b_dynamic_res_stage1',
        version='V2.0',
        max_new_tokens=512,
        screen_parse=False,
    ),
    'InternVL3-7B-GUI':
    partial(
        InternVLChat,  # noqa
        model_path='/fs-computility/mllm1/shared/zhaoxiangyu/GUI/checkpoints/internvl3_7b_dynamic_res_stage1_56/',
        version='V2.0',
        max_new_tokens=512,
        screen_parse=False,
    ),
}

internvl3 = {
    'InternVL3-1B': partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL3-1B',
        version='V2.0'  # noqa
    ),
    'InternVL3-2B': partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL3-2B',
        version='V2.0'  # noqa
    ),
    'InternVL3-8B': partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL3-8B',
        version='V2.0',  # noqa
    ),
    'InternVL3-9B': partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL3-9B',
        version='V2.0'  # noqa
    ),
    'InternVL3-14B': partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL3-14B',
        version='V2.0'  # noqa
    ),
    'InternVL3-38B': partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL3-38B',
        version='V2.0'  # noqa
    ),
    'InternVL3-78B': partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL3-78B',
        version='V2.0'  # noqa
    ),
}

internvl3_5 = {
    'InternVL3_5-1B':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL3_5-1B',
        version='V2.0'  # noqa
    ),
    'InternVL3_5-2B':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL3_5-2B',
        version='V2.0'  # noqa
    ),
    'InternVL3_5-4B':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL3_5-4B',
        version='V2.0'  # noqa
    ),
    'InternVL3_5-8B':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL3_5-8B',
        version='V2.0'  # noqa
    ),
    'InternVL3_5-14B':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL3_5-14B',
        version='V2.0'  # noqa
    ),
    'InternVL3_5-GPT-OSS-20B-A4B-Preview':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview',
        version='V2.0'  # noqa
    ),
    'InternVL3_5-30B-A3B':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL3_5-30B-A3B',
        version='V2.0'  # noqa
    ),
    'InternVL3_5-38B':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL3_5-38B',
        version='V2.0'  # noqa
    ),
    'InternVL3_5-241B-A28B':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL3_5-241B-A28B',
        version='V2.0'  # noqa
    ),
    'InternVL3_5-1B-Thinking':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL3_5-1B',
        use_lmdeploy=True,  # noqa
        max_new_tokens=2**16,
        cot_prompt_version='r1',
        do_sample=True,
        version='V2.0'),
    'InternVL3_5-2B-Thinking':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL3_5-2B',
        use_lmdeploy=True,  # noqa
        max_new_tokens=2**16,
        cot_prompt_version='r1',
        do_sample=True,
        version='V2.0'),
    'InternVL3_5-4B-Thinking':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL3_5-4B',
        use_lmdeploy=True,  # noqa
        max_new_tokens=2**16,
        cot_prompt_version='r1',
        do_sample=True,
        version='V2.0'),
    'InternVL3_5-8B-Thinking':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL3_5-8B',
        use_lmdeploy=True,  # noqa
        max_new_tokens=2**16,
        cot_prompt_version='r1',
        do_sample=True,
        version='V2.0'),
    'InternVL3_5-14B-Thinking':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL3_5-14B',
        use_lmdeploy=True,  # noqa
        max_new_tokens=2**16,
        cot_prompt_version='r1',
        do_sample=True,
        version='V2.0'),
    'InternVL3_5-GPT-OSS-20B-A4B-Preview-Thinking':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview',
        use_lmdeploy=True,  # noqa
        max_new_tokens=2**16,
        cot_prompt_version='r1',
        do_sample=True,
        version='V2.0'),
    'InternVL3_5-30B-A3B-Thinking':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL3_5-30B-A3B',
        use_lmdeploy=True,  # noqa
        max_new_tokens=2**16,
        cot_prompt_version='r1',
        do_sample=True,
        version='V2.0'),
    'InternVL3_5-38B-Thinking':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL3_5-38B',
        use_lmdeploy=True,  # noqa
        max_new_tokens=2**16,
        cot_prompt_version='r1',
        do_sample=True,
        version='V2.0'),
    'InternVL3_5-241B-A28B-Thinking':
    partial(
        InternVLChat,  # noqa
        model_path='OpenGVLab/InternVL3_5-241B-A28B',
        use_lmdeploy=True,  # noqa
        max_new_tokens=2**16,
        cot_prompt_version='r1',
        do_sample=True,
        version='V2.0'),
}

qwen3vl_series = {
    'Qwen3-VL-235B-A22B-Instruct':
    partial(
        Qwen3VLChat,  # noqa
        model_path='Qwen/Qwen3-VL-235B-A22B-Instruct',
        use_custom_prompt=False,
        use_vllm=True,
        temperature=0.7,
        max_new_tokens=16384,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20),
    'Qwen3-VL-235B-A22B-Thinking':
    partial(
        Qwen3VLChat,  # noqa
        model_path='Qwen/Qwen3-VL-235B-A22B-Thinking',
        use_custom_prompt=False,
        use_vllm=True,
        temperature=1.0,
        max_new_tokens=40960,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        top_p=0.95,
        top_k=20),
    'Qwen3-VL-30B-A3B-Instruct':
    partial(
        Qwen3VLChat,  # noqa
        model_path='Qwen/Qwen3-VL-30B-A3B-Instruct',
        use_custom_prompt=False,
        use_vllm=True,
        temperature=0.7,
        max_new_tokens=16384,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20),
    'Qwen3-VL-30B-A3B-Thinking':
    partial(
        Qwen3VLChat,  # noqa
        model_path='Qwen/Qwen3-VL-30B-A3B-Thinking',
        use_custom_prompt=False,
        use_vllm=True,
        temperature=1.0,
        max_new_tokens=40960,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        top_p=0.95,
        top_k=20),
    'Qwen3-VL-8B-Thinking':
    partial(
        Qwen3VLChat,  # noqa
        model_path='Qwen/Qwen3-VL-8B-Thinking',
        use_custom_prompt=False,
        use_vllm=True,
        temperature=1.0,
        max_new_tokens=40960,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        top_p=0.95,
        top_k=20),
    'Qwen3-VL-4B-Thinking':
    partial(
        Qwen3VLChat,  # noqa
        model_path='Qwen/Qwen3-VL-4B-Thinking',
        use_custom_prompt=False,
        use_vllm=True,
        temperature=1.0,
        max_new_tokens=40960,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        top_p=0.95,
        top_k=20),
    'Qwen3-VL-8B-Instruct':
    partial(
        Qwen3VLChat,  # noqa
        model_path='Qwen/Qwen3-VL-8B-Instruct',
        use_custom_prompt=False,
        use_vllm=True,
        temperature=0.7,
        max_new_tokens=16384,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20),
    'Qwen3-VL-4B-Instruct':
    partial(
        Qwen3VLChat,  # noqa
        model_path='Qwen/Qwen3-VL-4B-Instruct',
        use_custom_prompt=False,
        use_vllm=True,
        temperature=0.7,
        max_new_tokens=16384,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20),
    'Qwen3-VL-2B-Instruct':
    partial(
        Qwen3VLChat,  # noqa
        model_path='Qwen/Qwen3-VL-2B-Instruct',
        use_custom_prompt=False,
        use_vllm=True,
        temperature=0.7,
        max_new_tokens=16384,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20),
    'Qwen3-VL-32B-Instruct':
    partial(
        Qwen3VLChat,  # noqa
        model_path='Qwen/Qwen3-VL-32B-Instruct',
        use_custom_prompt=False,
        use_vllm=True,
        temperature=0.7,
        max_new_tokens=16384,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20),
    'Qwen3-VL-2B-Thinking':
    partial(
        Qwen3VLChat,  # noqa
        model_path='Qwen/Qwen3-VL-2B-Thinking',
        use_custom_prompt=False,
        use_vllm=True,
        temperature=1.0,
        max_new_tokens=40960,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        top_p=0.95,
        top_k=20),
    'Qwen3-VL-32B-Thinking':
    partial(
        Qwen3VLChat,  # noqa
        model_path='Qwen/Qwen3-VL-32B-Thinking',
        use_custom_prompt=False,
        use_vllm=False,
        temperature=1.0,
        max_new_tokens=40960,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        top_p=0.95,
        top_k=20),
    'Qwen3-Omni-30B-A3B-Instruct':
    partial(
        Qwen3VLChat,  # noqa
        model_path='Qwen/Qwen3-Omni-30B-A3B-Instruct',
        use_custom_prompt=False,
        use_vllm=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_new_tokens=16384,
    ),
    'Qwen3-Omni-30B-A3B-Thinking':
    partial(
        Qwen3VLChat,  # noqa
        model_path='Qwen/Qwen3-Omni-30B-A3B-Thinking',
        use_custom_prompt=False,
        use_vllm=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_new_tokens=16384,
    ),
    'Qwen3-Omni-30B-A3B-Captioner':
    partial(
        Qwen3VLChat,  # noqa
        model_path='Qwen/Qwen3-Omni-30B-A3B-Captioner',
        use_custom_prompt=False,
        use_vllm=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_new_tokens=16384,
    ),
}

sail_series = {
    'SAIL-VL-2B':
    partial(SailVL, model_path='BytedanceDouyinContent/SAIL-VL-2B'),  # noqa
    'SAIL-VL-1.5-2B':
    partial(SailVL, model_path='BytedanceDouyinContent/SAIL-VL-1d5-2B', use_msac=True),  # noqa
    'SAIL-VL-1.5-8B':
    partial(SailVL, model_path='BytedanceDouyinContent/SAIL-VL-1d5-8B', use_msac=True),  # noqa
    'SAIL-VL-1.6-8B':
    partial(SailVL, model_path='BytedanceDouyinContent/SAIL-VL-1d6-8B', use_msac=True),  # noqa
    'SAIL-VL-1.7-Thinking-2B-2507':
    partial(
        SailVL,  # noqa
        model_path='BytedanceDouyinContent/SAIL-VL-1d7-Thinking-2B-2507',
        use_msac=True,
        use_cot=True,
        max_new_tokens=4096),  # noqa
    'SAIL-VL-1.7-Thinking-8B-2507':
    partial(
        SailVL,  # noqa
        model_path='BytedanceDouyinContent/SAIL-VL-1d7-Thinking-8B-2507',
        use_msac=True,
        use_cot=True,
        max_new_tokens=4096),  # noqa
    'SAIL-VL2-2B':
    partial(SailVL, model_path='BytedanceDouyinContent/SAIL-VL2-2B', use_msac=True),  # noqa
    'SAIL-VL2-8B':
    partial(SailVL, model_path='BytedanceDouyinContent/SAIL-VL2-8B', use_msac=True),  # noqa
}

ristretto_series = {
    'Ristretto-3B': partial(Ristretto, model_path='LiAutoAD/Ristretto-3B'),  # noqa
}

yivl_series = {
    'Yi_VL_6B': partial(Yi_VL, model_path='01-ai/Yi-VL-6B', root=Yi_ROOT),  # noqa
    'Yi_VL_34B': partial(Yi_VL, model_path='01-ai/Yi-VL-34B', root=Yi_ROOT),  # noqa
}

xcomposer_series = {
    'XComposer': partial(XComposer, model_path='internlm/internlm-xcomposer-vl-7b'),  # noqa
    'sharecaptioner': partial(ShareCaptioner, model_path='Lin-Chen/ShareCaptioner'),  # noqa
    'XComposer2': partial(XComposer2, model_path='internlm/internlm-xcomposer2-vl-7b'),  # noqa
    'XComposer2_1.8b': partial(
        XComposer2,  # noqa
        model_path='internlm/internlm-xcomposer2-vl-1_8b'  # noqa
    ),
    'XComposer2_4KHD': partial(
        XComposer2_4KHD,  # noqa
        model_path='internlm/internlm-xcomposer2-4khd-7b'  # noqa
    ),
    'XComposer2d5': partial(
        XComposer2d5,  # noqa
        model_path='internlm/internlm-xcomposer2d5-7b'  # noqa
    ),
}

minigpt4_series = {
    'MiniGPT-4-v2': partial(MiniGPT4, mode='v2', root=MiniGPT4_ROOT),  # noqa
    'MiniGPT-4-v1-7B': partial(MiniGPT4, mode='v1_7b', root=MiniGPT4_ROOT),  # noqa
    'MiniGPT-4-v1-13B': partial(MiniGPT4, mode='v1_13b', root=MiniGPT4_ROOT),  # noqa
}

idefics_series = {
    'idefics_9b_instruct': partial(
        IDEFICS,  # noqa
        model_path='HuggingFaceM4/idefics-9b-instruct'  # noqa
    ),
    'idefics_80b_instruct': partial(
        IDEFICS,  # noqa
        model_path='HuggingFaceM4/idefics-80b-instruct'  # noqa
    ),
    'idefics2_8b': partial(IDEFICS2, model_path='HuggingFaceM4/idefics2-8b'),  # noqa
    # Idefics3 follows Idefics2 Pattern
    'Idefics3-8B-Llama3': partial(
        IDEFICS2,  # noqa
        model_path='HuggingFaceM4/Idefics3-8B-Llama3'  # noqa
    ),
}

smolvlm_series = {
    'SmolVLM-256M': partial(SmolVLM, model_path='HuggingFaceTB/SmolVLM-256M-Instruct'),  # noqa
    'SmolVLM-500M': partial(SmolVLM, model_path='HuggingFaceTB/SmolVLM-500M-Instruct'),  # noqa
    'SmolVLM': partial(SmolVLM, model_path='HuggingFaceTB/SmolVLM-Instruct'),  # noqa
    'SmolVLM-DPO': partial(SmolVLM, model_path='HuggingFaceTB/SmolVLM-Instruct-DPO'),  # noqa
    'SmolVLM-Synthetic': partial(SmolVLM, model_path='HuggingFaceTB/SmolVLM-Synthetic'),  # noqa
    'SmolVLM2-256M': partial(
        SmolVLM2,  # noqa
        model_path='HuggingFaceTB/SmolVLM2-256M-Video-Instruct'  # noqa
    ),
    'SmolVLM2-500M': partial(
        SmolVLM2,  # noqa
        model_path='HuggingFaceTB/SmolVLM2-500M-Video-Instruct'  # noqa
    ),
    'SmolVLM2': partial(SmolVLM2, model_path='HuggingFaceTB/SmolVLM2-2.2B-Instruct'),  # noqa
}

instructblip_series = {
    'instructblip_7b': partial(InstructBLIP, name='instructblip_7b'),  # noqa
    'instructblip_13b': partial(InstructBLIP, name='instructblip_13b'),  # noqa
}

deepseekvl_series = {
    'deepseek_vl_7b': partial(DeepSeekVL, model_path='deepseek-ai/deepseek-vl-7b-chat'),  # noqa
    'deepseek_vl_1.3b': partial(
        DeepSeekVL,  # noqa
        model_path='deepseek-ai/deepseek-vl-1.3b-chat'  # noqa
    ),
}

deepseekvl2_series = {
    'deepseek_vl2_tiny': partial(
        DeepSeekVL2,  # noqa
        model_path='deepseek-ai/deepseek-vl2-tiny'  # noqa
    ),
    'deepseek_vl2_small': partial(
        DeepSeekVL2,  # noqa
        model_path='deepseek-ai/deepseek-vl2-small'  # noqa
    ),
    'deepseek_vl2': partial(DeepSeekVL2, model_path='deepseek-ai/deepseek-vl2'),  # noqa
}

janus_series = {
    'Janus-1.3B': partial(Janus, model_path='deepseek-ai/Janus-1.3B'),  # noqa
    'Janus-Pro-1B': partial(Janus, model_path='deepseek-ai/Janus-Pro-1B'),  # noqa
    'Janus-Pro-7B': partial(Janus, model_path='deepseek-ai/Janus-Pro-7B'),  # noqa
}

cogvlm_series = {
    'cogvlm-grounding-generalist':
    partial(
        CogVlm,  # noqa
        model_path='THUDM/cogvlm-grounding-generalist-hf',
        tokenizer_name='lmsys/vicuna-7b-v1.5',
    ),
    'cogvlm-chat':
    partial(
        CogVlm,  # noqa
        model_path='THUDM/cogvlm-chat-hf',
        tokenizer_name='lmsys/vicuna-7b-v1.5'  # noqa
    ),
    'cogvlm2-llama3-chat-19B':
    partial(
        CogVlm,  # noqa
        model_path='THUDM/cogvlm2-llama3-chat-19B'  # noqa
    ),
    'glm-4v-9b':
    partial(GLM4v, model_path='THUDM/glm-4v-9b'),  # noqa
    'GLM4_1VThinking-9b':
    partial(GLMThinking, model_path='THUDM/GLM-4.1V-9B-Thinking'),  # noqa
    'GLM4_5V':
    partial(GLMThinking, model_path='THUDM/GLM-4.5V'),  # noqa
}

wemm_series = {
    'WeMM': partial(WeMM, model_path='feipengma/WeMM'),  # noqa
}

cambrian_series = {
    'cambrian_8b': partial(Cambrian, model_path='nyu-visionx/cambrian-8b'),  # noqa
    'cambrian_13b': partial(Cambrian, model_path='nyu-visionx/cambrian-13b'),  # noqa
    'cambrian_34b': partial(Cambrian, model_path='nyu-visionx/cambrian-34b'),  # noqa
}

chameleon_series = {
    'chameleon_7b': partial(Chameleon, model_path='facebook/chameleon-7b'),  # noqa
    'chameleon_30b': partial(Chameleon, model_path='facebook/chameleon-30b'),  # noqa
}

vila_series = {
    'VILA1.5-3b': partial(VILA, model_path='Efficient-Large-Model/VILA1.5-3b'),  # noqa
    'Llama-3-VILA1.5-8b': partial(VILA, model_path='Efficient-Large-Model/Llama-3-VILA1.5-8b'),  # noqa
    'VILA1.5-13b': partial(VILA, model_path='Efficient-Large-Model/VILA1.5-13b'),  # noqa
    'VILA1.5-40b': partial(VILA, model_path='Efficient-Large-Model/VILA1.5-40b'),  # noqa
    'NVILA-8B': partial(NVILA, model_path='Efficient-Large-Model/NVILA-8B'),  # noqa
    'NVILA-15B': partial(NVILA, model_path='Efficient-Large-Model/NVILA-15B'),  # noqa
}

ovis_series = {
    'Ovis1.5-Llama3-8B': partial(Ovis, model_path='AIDC-AI/Ovis1.5-Llama3-8B'),  # noqa
    'Ovis1.5-Gemma2-9B': partial(Ovis, model_path='AIDC-AI/Ovis1.5-Gemma2-9B'),  # noqa
    'Ovis1.6-Gemma2-9B': partial(Ovis1_6, model_path='AIDC-AI/Ovis1.6-Gemma2-9B'),  # noqa
    'Ovis1.6-Llama3.2-3B': partial(Ovis1_6, model_path='AIDC-AI/Ovis1.6-Llama3.2-3B'),  # noqa
    'Ovis1.6-Gemma2-27B': partial(
        Ovis1_6_Plus,  # noqa
        model_path='AIDC-AI/Ovis1.6-Gemma2-27B'  # noqa
    ),
    'Ovis2-1B': partial(Ovis2, model_path='AIDC-AI/Ovis2-1B'),  # noqa
    'Ovis2-2B': partial(Ovis2, model_path='AIDC-AI/Ovis2-2B'),  # noqa
    'Ovis2-4B': partial(Ovis2, model_path='AIDC-AI/Ovis2-4B'),  # noqa
    'Ovis2-8B': partial(Ovis2, model_path='AIDC-AI/Ovis2-8B'),  # noqa
    'Ovis2-16B': partial(Ovis2, model_path='AIDC-AI/Ovis2-16B'),  # noqa
    'Ovis2-34B': partial(Ovis2, model_path='AIDC-AI/Ovis2-34B'),  # noqa
    'Ovis-U1-3B': partial(OvisU1, model_path='AIDC-AI/Ovis-U1-3B'),  # noqa
    'Ovis2.5-2B': partial(Ovis2_5, model_path='AIDC-AI/Ovis2.5-2B'),  # noqa
    'Ovis2.5-9B': partial(Ovis2_5, model_path='AIDC-AI/Ovis2.5-9B')  # noqa
}

mantis_series = {
    'Mantis-8B-siglip-llama3': partial(
        Mantis,  # noqa
        model_path='TIGER-Lab/Mantis-8B-siglip-llama3'  # noqa
    ),
    'Mantis-8B-clip-llama3': partial(
        Mantis,  # noqa
        model_path='TIGER-Lab/Mantis-8B-clip-llama3'  # noqa
    ),
    'Mantis-8B-Idefics2': partial(Mantis, model_path='TIGER-Lab/Mantis-8B-Idefics2'),  # noqa
    'Mantis-8B-Fuyu': partial(Mantis, model_path='TIGER-Lab/Mantis-8B-Fuyu'),  # noqa
}

phi3_series = {
    'Phi-3-Vision': partial(
        Phi3Vision,  # noqa
        model_path='microsoft/Phi-3-vision-128k-instruct'),
    'Phi-3.5-Vision': partial(Phi3_5Vision, model_path='microsoft/Phi-3.5-vision-instruct'),  # noqa
}

phi4_series = {
    'Phi-4-Vision': partial(Phi4Multimodal, model_path='microsoft/Phi-4-multimodal-instruct'),  # noqa
}

xgen_mm_series = {
    'xgen-mm-phi3-interleave-r-v1.5':
    partial(
        XGenMM,  # noqa
        model_path='Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5'  # noqa
    ),
    'xgen-mm-phi3-dpo-r-v1.5':
    partial(
        XGenMM,  # noqa
        model_path='Salesforce/xgen-mm-phi3-mini-instruct-dpo-r-v1.5'  # noqa
    ),
}

hawkvl_series = {
    'HawkVL-2B':
    partial(
        HawkVL,  # noqa
        model_path='xjtupanda/HawkVL-2B',
        min_pixels=4 * 28 * 28,
        max_pixels=6800 * 28 * 28,
        use_custom_prompt=True)
}

qwen2vl_series = {
    'Qwen-VL-Max-20250813':
    partial(
        Qwen2VLAPI,  # noqa # noqa
        model='qwen-vl-max-2025-08-13',
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        max_length=8192,
    ),
    'Qwen-VL-Max-0809':
    partial(
        Qwen2VLAPI,  # noqa
        model='qwen-vl-max-0809',
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    'Qwen-VL-Plus-0809':
    partial(
        Qwen2VLAPI,  # noqa
        model='qwen-vl-plus-0809',
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    'QVQ-72B-Preview':
    partial(
        Qwen2VLChat,  # noqa
        model_path='Qwen/QVQ-72B-Preview',
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        system_prompt='''
You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.
''',  # noqa
        max_new_tokens=8192,
        post_process=False,
    ),
    'Qwen2-VL-72B-Instruct':
    partial(
        Qwen2VLChat,  # noqa
        model_path='Qwen/Qwen2-VL-72B-Instruct',
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    'Qwen2-VL-7B-Instruct':
    partial(
        Qwen2VLChat,  # noqa
        model_path='Qwen/Qwen2-VL-7B-Instruct',
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    'Qwen2-VL-7B-Instruct-AWQ':
    partial(
        Qwen2VLChat,  # noqa
        model_path='Qwen/Qwen2-VL-7B-Instruct-AWQ',
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    'Qwen2-VL-7B-Instruct-GPTQ-Int4':
    partial(
        Qwen2VLChat,  # noqa
        model_path='Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4',
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    'Qwen2-VL-7B-Instruct-GPTQ-Int8':
    partial(
        Qwen2VLChat,  # noqa
        model_path='Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8',
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    'Qwen2-VL-2B-Instruct':
    partial(
        Qwen2VLChat,  # noqa
        model_path='Qwen/Qwen2-VL-2B-Instruct',
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    'Qwen2-VL-2B-Instruct-AWQ':
    partial(
        Qwen2VLChat,  # noqa
        model_path='Qwen/Qwen2-VL-2B-Instruct-AWQ',
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    'Qwen2-VL-2B-Instruct-GPTQ-Int4':
    partial(
        Qwen2VLChat,  # noqa
        model_path='Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4',
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    'Qwen2-VL-2B-Instruct-GPTQ-Int8':
    partial(
        Qwen2VLChat,  # noqa
        model_path='Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8',
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    'XinYuan-VL-2B-Instruct':
    partial(
        Qwen2VLChat,  # noqa
        model_path='Cylingo/Xinyuan-VL-2B',
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    'Qwen2.5-VL-3B-Instruct':
    partial(
        Qwen2VLChat,  # noqa
        model_path='Qwen/Qwen2.5-VL-3B-Instruct',
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    'Qwen2.5-VL-3B-Instruct-AWQ':
    partial(
        Qwen2VLChat,  # noqa
        model_path='Qwen/Qwen2.5-VL-3B-Instruct-AWQ',
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    'Qwen2.5-VL-7B-Instruct':
    partial(
        Qwen2VLChat,  # noqa
        model_path='Qwen/Qwen2.5-VL-7B-Instruct',
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    'Qwen2.5-VL-7B-Instruct-ForVideo':
    partial(
        Qwen2VLChat,  # noqa
        model_path='Qwen/Qwen2.5-VL-7B-Instruct',
        min_pixels=128 * 28 * 28,
        max_pixels=768 * 28 * 28,
        total_pixels=24576 * 28 * 28,
        use_custom_prompt=False,
    ),
    'Qwen2.5-VL-7B-Instruct-AWQ':
    partial(
        Qwen2VLChat,  # noqa
        model_path='Qwen/Qwen2.5-VL-7B-Instruct-AWQ',
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    'Qwen2.5-VL-32B-Instruct':
    partial(
        Qwen2VLChat,  # noqa
        model_path='Qwen/Qwen2.5-VL-32B-Instruct',
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    'Qwen2.5-VL-72B-Instruct':
    partial(
        Qwen2VLChat,  # noqa
        model_path='Qwen/Qwen2.5-VL-72B-Instruct',
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    'MiMo-VL-7B-SFT':
    partial(
        Qwen2VLChat,  # noqa
        model_path='XiaomiMiMo/MiMo-VL-7B-SFT',
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
        use_lmdeploy=True),
    'MiMo-VL-7B-RL':
    partial(
        Qwen2VLChat,  # noqa
        model_path='XiaomiMiMo/MiMo-VL-7B-RL',
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
        use_lmdeploy=True),
    'Qwen2.5-VL-72B-Instruct-ForVideo':
    partial(
        Qwen2VLChat,  # noqa
        model_path='Qwen/Qwen2.5-VL-72B-Instruct',
        min_pixels=128 * 28 * 28,
        max_pixels=768 * 28 * 28,
        total_pixels=24576 * 28 * 28,
        use_custom_prompt=False,
    ),
    'Qwen2.5-VL-72B-Instruct-AWQ':
    partial(
        Qwen2VLChat,  # noqa
        model_path='Qwen/Qwen2.5-VL-72B-Instruct-AWQ',
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    'Qwen2.5-Omni-7B-ForVideo':
    partial(
        Qwen2VLChat,  # noqa
        model_path='Qwen/Qwen2.5-Omni-7B',
        min_pixels=128 * 28 * 28,
        max_pixels=768 * 28 * 28,
        total_pixels=24576 * 28 * 28,
        use_custom_prompt=False,
        use_audio_in_video=True,  # set use audio in video
    ),
    'Qwen2.5-Omni-7B':
    partial(
        Qwen2VLChat,  # noqa
        model_path='Qwen/Qwen2.5-Omni-7B',
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    'VLM-R1':
    partial(
        VLMR1Chat,  # noqa
        model_path='omlab/VLM-R1-Qwen2.5VL-3B-Math-0305',
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False),
    'VLAA-Thinker-Qwen2.5VL-3B':
    partial(
        VLAAThinkerChat,  # noqa
        model_path='UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-3B',
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
        post_process=True,  # post processing for evaluation
        system_prompt=(
            ''
            'You are VL-Thinking🤔, a helpful assistant with excellent reasoning ability.'
            ' A user asks you a question, and you should try to solve it.'
            ' You should first think about the reasoning process in the mind and then provides the user with the answer.'  # noqa
            ' The reasoning process and answer are enclosed within <think> </think> and'
            '<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>'
            '<answer> answer here </answer>'),
    ),
    'VLAA-Thinker-Qwen2.5VL-7B':
    partial(
        VLAAThinkerChat,  # noqa
        model_path='UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-7B',
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
        post_process=True,  # post processing for evaluation
        system_prompt=(
            ''
            'You are VL-Thinking🤔, a helpful assistant with excellent reasoning ability.'
            ' A user asks you a question, and you should try to solve it.'
            ' You should first think about the reasoning process in the mind and then provides the user with the answer.'  # noqa
            ' The reasoning process and answer are enclosed within <think> </think> and'
            '<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>'
            '<answer> answer here </answer>'),
    ),
    'WeThink-Qwen2.5VL-7B':
    partial(
        WeThinkVL,  # noqa
        model_path='yangjie-cv/WeThink-Qwen2.5VL-7B',
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
        system_prompt=(
            'You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\nThe reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE enclosed within <answer> </answer> tags.'  # noqa
        ),
    ),
}

slime_series = {
    'Slime-7B': partial(SliME, model_path='yifanzhang114/SliME-vicuna-7B'),  # noqa
    'Slime-8B': partial(SliME, model_path='yifanzhang114/SliME-Llama3-8B'),  # noqa
    'Slime-13B': partial(SliME, model_path='yifanzhang114/SliME-vicuna-13B'),  # noqa
}

eagle_series = {
    'Eagle-X4-8B-Plus': partial(Eagle, model_path='NVEagle/Eagle-X4-8B-Plus'),  # noqa
    'Eagle-X4-13B-Plus': partial(Eagle, model_path='NVEagle/Eagle-X4-13B-Plus'),  # noqa
    'Eagle-X5-7B': partial(Eagle, model_path='NVEagle/Eagle-X5-7B'),  # noqa
    'Eagle-X5-13B': partial(Eagle, model_path='NVEagle/Eagle-X5-13B'),  # noqa
    'Eagle-X5-13B-Chat': partial(Eagle, model_path='NVEagle/Eagle-X5-13B-Chat'),  # noqa
    'Eagle-X5-34B-Chat': partial(Eagle, model_path='NVEagle/Eagle-X5-34B-Chat'),  # noqa
    'Eagle-X5-34B-Plus': partial(Eagle, model_path='NVEagle/Eagle-X5-34B-Plus'),  # noqa
}

moondream_series = {
    'Moondream1': partial(Moondream1, model_path='vikhyatk/moondream1'),  # noqa
    'Moondream2': partial(Moondream2, model_path='vikhyatk/moondream2'),  # noqa
}

llama_series = {
    'Llama-3.2-11B-Vision-Instruct':
    partial(
        llama_vision,  # noqa
        model_path='meta-llama/Llama-3.2-11B-Vision-Instruct'  # noqa
    ),
    'LLaVA-CoT':
    partial(llama_vision, model_path='Xkev/Llama-3.2V-11B-cot'),  # noqa
    'Llama-3.2-90B-Vision-Instruct':
    partial(
        llama_vision,  # noqa
        model_path='meta-llama/Llama-3.2-90B-Vision-Instruct'  # noqa
    ),
    'Llama-4-Scout-17B-16E-Instruct':
    partial(
        llama4,  # noqa
        model_path='meta-llama/Llama-4-Scout-17B-16E-Instruct',
        use_vllm=True  # noqa
    ),
}

molmo_series = {
    'molmoE-1B-0924': partial(molmo, model_path='allenai/MolmoE-1B-0924'),  # noqa
    'molmo-7B-D-0924': partial(molmo, model_path='allenai/Molmo-7B-D-0924'),  # noqa
    'molmo-7B-O-0924': partial(molmo, model_path='allenai/Molmo-7B-O-0924'),  # noqa
    'molmo-72B-0924': partial(molmo, model_path='allenai/Molmo-72B-0924'),  # noqa
}

kosmos_series = {
    'Kosmos2': partial(Kosmos2, model_path='microsoft/kosmos-2-patch14-224')  # noqa
}

points_series = {
    'POINTS-Yi-1.5-9B-Chat': partial(
        POINTS,  # noqa
        model_path='WePOINTS/POINTS-Yi-1-5-9B-Chat'  # noqa
    ),
    'POINTS-Qwen-2.5-7B-Chat': partial(
        POINTS,  # noqa
        model_path='WePOINTS/POINTS-Qwen-2-5-7B-Chat'  # noqa
    ),
    'POINTSV15-Qwen-2.5-7B-Chat': partial(
        POINTSV15,  # noqa
        model_path='WePOINTS/POINTS-1-5-Qwen-2-5-7B-Chat'  # noqa
    ),
}

nvlm_series = {
    'NVLM': partial(NVLM, model_path='nvidia/NVLM-D-72B'),  # noqa
}

vintern_series = {
    'Vintern-3B-beta': partial(VinternChat, model_path='5CD-AI/Vintern-3B-beta'),  # noqa
    'Vintern-1B-v2': partial(VinternChat, model_path='5CD-AI/Vintern-1B-v2'),  # noqa
}

aria_series = {'Aria': partial(Aria, model_path='rhymes-ai/Aria')}  # noqa

h2ovl_series = {
    'h2ovl-mississippi-2b': partial(H2OVLChat, model_path='h2oai/h2ovl-mississippi-2b'),  # noqa
    'h2ovl-mississippi-1b': partial(
        H2OVLChat,  # noqa
        model_path='h2oai/h2ovl-mississippi-800m'),
}

valley_series = {
    'valley2': partial(
        Valley2Chat,  # noqa
        model_path='bytedance-research/Valley-Eagle-7B'),
    'valley2_dpo': partial(
        Valley2Chat,  # noqa
        model_path='bytedance-research/Valley2-DPO'),
    'valley3': partial(
        Valley3Chat,  # noqa
        use_gthinker_thinking=True,
        model_path='bytedance-research/Valley3'),
}

ola_series = {
    'ola': partial(Ola, model_path='THUdyh/Ola-7b'),  # noqa
}

xvl_series = {
    'X-VL-4B': partial(X_VL_HF, model_path='YannQi/X-VL-4B', temperature=0, retry=10),  # noqa
}

ross_series = {
    'ross-qwen2-7b': partial(Ross, model_path='HaochenWang/ross-qwen2-7b'),  # noqa
}

ursa_series = {
    'URSA-8B': partial(UrsaChat, model_path='URSA-MATH/URSA-8B'),  # noqa
    'URSA-8B-PS-GRPO': partial(UrsaChat, model_path='URSA-MATH/URSA-8B-PS-GRPO')  # noqa
}

gemma_series = {
    'paligemma-3b-mix-448': partial(
        PaliGemma,  # noqa
        model_path='google/paligemma-3b-mix-448'  # noqa
    ),

    # 3B
    'paligemma2-3b-pt-224': partial(PaliGemma, model_path='google/paligemma2-3b-pt-224'),  # noqa
    'paligemma2-3b-pt-448': partial(PaliGemma, model_path='google/paligemma2-3b-pt-448'),  # noqa
    'paligemma2-3b-mix-224': partial(PaliGemma, model_path='google/paligemma2-3b-mix-224'),  # noqa
    'paligemma2-3b-mix-448': partial(PaliGemma, model_path='google/paligemma2-3b-mix-448'),  # noqa

    # 10B
    'paligemma2-10b-pt-224': partial(PaliGemma, model_path='google/paligemma2-10b-pt-224'),  # noqa
    'paligemma2-10b-pt-448': partial(PaliGemma, model_path='google/paligemma2-10b-pt-448'),  # noqa
    'paligemma2-10b-mix-224': partial(PaliGemma, model_path='google/paligemma2-10b-mix-224'),  # noqa
    'paligemma2-10b-mix-448': partial(PaliGemma, model_path='google/paligemma2-10b-mix-448'),  # noqa

    # 28B
    'paligemma2-28b-pt-224': partial(PaliGemma, model_path='google/paligemma2-28b-pt-224'),  # noqa
    'paligemma2-28b-pt-448': partial(PaliGemma, model_path='google/paligemma2-28b-pt-448'),  # noqa
    'paligemma2-28b-mix-224': partial(PaliGemma, model_path='google/paligemma2-28b-mix-224'),  # noqa
    'paligemma2-28b-mix-448': partial(PaliGemma, model_path='google/paligemma2-28b-mix-448'),  # noqa
    'Gemma3-4B': partial(Gemma3, model_path='google/gemma-3-4b-it'),  # noqa
    'Gemma3-12B': partial(Gemma3, model_path='google/gemma-3-12b-it'),  # noqa
    'Gemma3-27B': partial(Gemma3, model_path='google/gemma-3-27b-it')  # noqa
}

aguvis_series = {
    'aguvis_7b':
    partial(
        Qwen2VLChatAguvis,  # noqa
        model_path=os.getenv(
            'EVAL_MODEL',
            'xlangai/Aguvis-7B-720P',
        ),
        min_pixels=256 * 28 * 28,
        max_pixels=46 * 26 * 28 * 28,
        use_custom_prompt=False,
        mode='grounding',
    )
}

kimi_series = {
    'Kimi-VL-A3B-Thinking':
    partial(KimiVL, model_path='moonshotai/Kimi-VL-A3B-Thinking'),  # noqa
    'Kimi-VL-A3B-Instruct':
    partial(KimiVL, model_path='moonshotai/Kimi-VL-A3B-Instruct'),  # noqa
    'Kimi-VL-A3B-Thinking-2506':
    partial(
        KimiVL,  # noqa
        model_path='moonshotai/Kimi-VL-A3B-Thinking-2506',
        temperature=0.8,
        max_tokens=32768,
        extract_summary=True)  # noqa
}

flash_vl = {
    'Flash-VL-2B-Dynamic-ISS': partial(FlashVL, model_path='FlashVL/FlashVL-2B-Dynamic-ISS')  # noqa
}

oryx_series = {
    'oryx': partial(Oryx, model_path='THUdyh/Oryx-1.5-7B'),  # noqa
}

# recommend: vllm serve moonshotai/Kimi-VL-A3B-Thinking-2506
# --served-model-name api-kimi-vl-thinking-2506 --trust-remote-code
# --tensor-parallel-size 2 --max-num-batched-tokens 131072
# --max-model-len 131072 --limit-mm-per-prompt image=256
kimi_vllm_series = {
    'api-kimi-vl-thinking-2506': partial(
        KimiVLAPI,  # noqa
        model='api-kimi-vl-thinking-2506',
    ),
    'api-kimi-vl-thinking': partial(
        KimiVLAPI,  # noqa
        model='api-kimi-vl-thinking',
    ),
    'api-kimi-vl': partial(
        KimiVLAPI,  # noqa
        model='api-kimi-vl',
        max_new_tokens=2048,
        temperature=0,
    ),
}

treevgr_series = {
    'TreeVGR-7B':
    partial(
        TreeVGR,  # noqa
        model_path='HaochenWang/TreeVGR-7B',
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
}

# QTuneVL series
qtunevl_series = {
    'QTuneVL1_5-2B':
    partial(
        QTuneVLChat,  # noqa
        model_path='hanchaow/QTuneVL1_5-2B',
        version='V1.5'  # noqa
    ),
    'QTuneVL1_5-3B':
    partial(
        QTuneVL,  # noqa
        model_path='hanchaow/QTuneVL1_5-3B',
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=True,
        post_process=True),
}

# RbdashMM series via lmdeploy API
rbdashmm_api_series_lmdeploy = {
    'rbdashmm3_DPO_38B_api':
    partial(
        RBdashMMChat3_API,  # noqa
        api_base='http://0.0.0.0:23333/v1/chat/completions',
        temperature=0,
        retry=3,
        timeout=600),
    'rbdashmm3_5_DPO_38B_api':
    partial(
        RBdashChat3_5_API,  # noqa
        api_base='http://0.0.0.0:23333/v1/chat/completions',
        temperature=0,
        retry=3,
        timeout=600),
    'rbdashmm3_5_38B_api':
    partial(
        RBdashMMChat3_5_38B_API,  # noqa
        api_base='http://0.0.0.0:23333/v1/chat/completions',
        temperature=0,
        retry=3,
        timeout=600),
    'rbdashmm3_78B_api':
    partial(
        RBdashMMChat3_78B_API,  # noqa
        api_base='http://0.0.0.0:23333/v1/chat/completions',
        temperature=0,
        retry=3,
        timeout=600)
}

logics_series = {
    'Logics-Thinking-8B': partial(Logics_Thinking, model_path='Logics-MLLM/Logics-Thinking-8B'),  # noqa
    'Logics-Thinking-32B': partial(Logics_Thinking, model_path='Logics-MLLM/Logics-Thinking-32B'),  # noqa
}

insight_v_series = {
    'insightv':
    partial(
        InsightV,  # noqa
        pretrained_reason='THUdyh/Insight-V-Reason-LLaMA3',
        pretrained_summary='THUdyh/Insight-V-Summary-LLaMA3'),  # noqa
}

cosmos_series = {
    'Cosmos-Reason1-7B': partial(Cosmos, model_path='nvidia/Cosmos-Reason1-7B', use_vllm=True),  # noqa
}

keye_series = {
    'Keye-VL-1.5-8B-auto': partial(KeyeChat, model_path='Kwai-Keye/Keye-VL-1_5-8B'),  # noqa
    'Keye-VL-1.5-8B-think': partial(KeyeChat, model_path='Kwai-Keye/Keye-VL-1_5-8B', think=True),  # noqa
    'Keye-VL-1.5-8B-nothink': partial(KeyeChat, model_path='Kwai-Keye/Keye-VL-1_5-8B', no_think=True),  # noqa
    'Keye-VL-8B-Preview-think': partial(KeyeChat, model_path='Kwai-Keye/Keye-VL-8B-Preview', think=True),  # noqa
}

qianfanvl_series = {
    'Qianfan-VL-3B': partial(Qianfan_VL, model_path='baidu/Qianfan-VL-3B'),  # noqa
    'Qianfan-VL-8B': partial(Qianfan_VL, model_path='baidu/Qianfan-VL-8B'),  # noqa
    'Qianfan-VL-70B': partial(Qianfan_VL, model_path='baidu/Qianfan-VL-70B'),  # noqa
}

lfm2vl_series = {
    'LFM2-VL-450M': partial(LFM2VL, model_path='LiquidAI/LFM2-VL-450M'),  # noqa
    'LFM2-VL-1.6B': partial(LFM2VL, model_path='LiquidAI/LFM2-VL-1.6B'),  # noqa
    'LFM2-VL-3B': partial(LFM2VL, model_path='LiquidAI/LFM2-VL-3B'),  # noqa
}

internvl_groups = [internvl, internvl2, internvl2_5, mini_internvl, internvl2_5_mpo, internvl3, internvl3_5]
internvl_series = {}
for group in internvl_groups:
    internvl_series.update(group)

interns1_groups = [interns1_mini]
interns1_series = {}
for group in interns1_groups:
    interns1_series.update(group)

supported_VLM = {}

model_groups = [
    ungrouped, o1_apis, api_models, xtuner_series, qwen_series, llava_series, granite_vision_series, internvl_series,
    yivl_series, xcomposer_series, minigpt4_series, idefics_series, instructblip_series, deepseekvl_series,
    deepseekvl2_series, janus_series, minicpm_series, cogvlm_series, wemm_series, cambrian_series, chameleon_series,
    video_models, ovis_series, vila_series, mantis_series, mmalaya_series, phi3_series, phi4_series, xgen_mm_series,
    qwen2vl_series, qwen3vl_series, slime_series, eagle_series, moondream_series, llama_series, molmo_series,
    kosmos_series, points_series, nvlm_series, vintern_series, h2ovl_series, aria_series, smolvlm_series, sail_series,
    valley_series, vita_series, ross_series, emu_series, ola_series, ursa_series, gemma_series, long_vita_series,
    ristretto_series, kimi_series, aguvis_series, hawkvl_series, flash_vl, kimi_vllm_series, oryx_series,
    treevgr_series, varco_vision_series, qtunevl_series, xvl_series, thyme_series, logics_series, cosmos_series,
    keye_series, qianfanvl_series, lfm2vl_series, rbdashmm_api_series_lmdeploy, interns1_series, insight_v_series
]

for grp in model_groups:
    supported_VLM.update(grp)
