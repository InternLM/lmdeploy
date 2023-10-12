#!/usr/bin/env python3
from lmdeploy.serve.openai.api_server import main as serve
from pathlib import Path

# # Register your custom model.
#
# from lmdeploy.model import MODELS, BaseModel
# @MODELS.register_module(name='custom_model')
# class CustomModel(BaseModel):
#     pass


if __name__ == '__main__':
    serve(
        model_path=str(Path(__file__).resolve().parent),
        server_name="0.0.0.0",
        server_port=23333,
        instance_num=32,
        tp=1
    )
