import json
import os
import random
import time
from urllib import parse, request

from dreambooth_test import dreambooth_test

MODEL_LIST = [
    "myProjectName-step00002000.safetensors",
    "myProjectName-step00004000.safetensors",
    "myProjectName-step00006000.safetensors",
    "myProjectName-step00008000.safetensors",
    "myProjectName-step00010000.safetensors",
    "myProjectName-step00012000.safetensors",
    "myProjectName-step00014000.safetensors",
    "myProjectName-step00016000.safetensors",
]


def get_mode_id(model):
    # モデルのファイル名から拡張子を取り除いたものをIDとする
    model_id = os.path.splitext(model)[0]
    return model_id


def main_all():
    for model in MODEL_LIST:
        main(model)
        break


def main(model):
    model_id = get_mode_id(model)
    final_output_dir = dreambooth_test(model, model_id)
    time.sleep(40)
    os.system(
        "python overfitting_detector.py -i " + final_output_dir + " -o " + model_id
    )


if __name__ == "__main__":
    main_all()
