import argparse
import datetime
import json
import os
import random
from urllib import parse, request


def load_prompt_from_json(json_path: str) -> dict[str:dict]:
    with open(json_path) as f:
        data = json.load(f)
    return data


def update_prompt(prompt, model):
    prompt["4"]["inputs"]["ckpt_name"] = model
    return prompt


def queue_prompt(prompt):
    p = {"prompt": prompt}
    data = json.dumps(p).encode("utf-8")
    req = request.Request("http://127.0.0.1:8188/prompt", data=data)
    request.urlopen(req)


def make(prompt):
    seed_id = "3"
    ckpt_id = "4"
    clip_id_positive = "6"
    clip_id_negative = "7"
    print(prompt[clip_id_positive])

    # set the text prompt for our positive CLIPTextEncode
    prompt[clip_id_positive]["inputs"][
        "text"
    ] += ", doing day-to-day activities, looking happynice"

    # set the seed for our KSampler node
    seed = random.randint(0, 10000000)
    prompt[seed_id]["inputs"]["seed"] = seed


def make_all(prompt, count=2):
    for i in range(count):
        make(prompt)
        queue_prompt(prompt)


def get_final_output_dir(prompt, output_dir=""):
    select_folder_path_easy = prompt["15"]["inputs"]
    if output_dir:
        select_folder_path_easy["folder_name"] = output_dir
    folder_name = select_folder_path_easy["folder_name"]
    file_name = select_folder_path_easy["file_name"]
    data_str = datetime.datetime.now().strftime("%Y-%m-%d")
    final_output_dir = os.path.join("../output", data_str, folder_name)
    final_output_dir = os.path.abspath(final_output_dir)
    print("final_output_dir=", final_output_dir)
    return final_output_dir


def arg_parser():
    parser = argparse.ArgumentParser(description="dreambooth_test")
    parser.add_argument("--output_dir", "-o", help="output_dir", default="output/html")
    parser.add_argument(
        "--model",
        "-m",
        help="model",
        default="myProjectName.safetensors",
    )
    options = parser.parse_args()
    return options


def dreambooth_test(model, output_dir):
    json_dir = "config"
    json_file = "workflow_api.json"
    json_file_test = "workflow_api_test.json"
    json_file_lcm = "workflow_api_lcm.json"
    json_path = os.path.join(json_dir, json_file_lcm)
    prompt = load_prompt_from_json(json_path)
    prompt = update_prompt(prompt, model)
    final_output_dir = get_final_output_dir(prompt, output_dir)
    make_all(prompt)
    return final_output_dir


def main():
    options_ = arg_parser()
    random.seed(1)
    model = options_.model
    output_dir = options_.output_dir
    final_output_dir = dreambooth_test(model, output_dir)
    return final_output_dir


if __name__ == "__main__":
    main()
