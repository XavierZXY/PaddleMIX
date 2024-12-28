import copy
import json
import logging as logger
import os
import shutil

import paddle
import torch
from safetensors.numpy import save_file
from safetensors.torch import load_file

model_path = "/home/zxy/.cache/huggingface/hub/models--lmms-lab--LLaVA-Video-7B-Qwen2/snapshots/013210b3aff822f1558b166d39c1046dd109520f"
dst_path = model_path + "_pd"

# # 这里不修改，xxxxx代表随机名称，完全不会匹配到对应的key
src_prefix_key = "xxxxx."
dst_prefix_key = "xxxxx."

if not os.path.exists(dst_path):
    os.mkdir(dst_path)

need_transpose = {
    # language_model
    "mlp.down_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "lm_head.weight",
    # vision_model
    "self_attn.out_proj.weight",
    "mlp.fc1.weight",
    "mlp.fc2.weight",
    # resampler
    "resampler.attn.in_proj_weight",
    "resampler.attn.out_proj.weight",
    "resampler.attn.kv_proj.weight",
    "resampler.kv_proj.weight",
}


def check_trans(key):
    for x in need_transpose:
        if x in key:
            return True

    return False


def translate_one_safetensors(file_name):
    tensors = load_file(os.path.join(model_path, file_name))
    for key in list(tensors.keys()):
        dst_key = key.replace(src_prefix_key, dst_prefix_key)
        dst_key = dst_key.replace("llm.model.", "llm.qwen2.")  ###
        logger.info("{} {}".format(key, tensors[key].shape))
        shape_ = tensors[key].shape
        if check_trans(key) and len(shape_) == 2:
            t = tensors.pop(key).cuda().t().contiguous()
            capsule = torch.utils.dlpack.to_dlpack(t)
            t = paddle.utils.dlpack.from_dlpack(capsule)
            tensors[dst_key] = t.numpy()
        else:
            t = tensors.pop(key).cuda()
            capsule = torch.utils.dlpack.to_dlpack(t)
            t = paddle.utils.dlpack.from_dlpack(capsule)
            tensors[dst_key] = t.numpy()
        logger.info("{} {}".format(dst_key, tensors[dst_key].shape))

    save_file(
        tensors, os.path.join(dst_path, file_name), metadata={"format": "np"}
    )


def execute_cmd(cmd, file_path):
    cmd = cmd + " " + file_path
    os.system(cmd)


def main():
    if os.path.exists(os.path.join(model_path, "model.safetensors.index.json")):
        index = json.load(
            open(os.path.join(model_path, "model.safetensors.index.json"))
        )
        print(" ******** I have read the index file")
        dst_index = copy.deepcopy(index)
        for key in list(dst_index["weight_map"].keys()):
            # print(f"key: {key} \n")
            dst_key = key.replace(src_prefix_key, dst_prefix_key)
            dst_index["weight_map"][dst_key] = dst_index["weight_map"].pop(key)

        files = set(index["weight_map"].values())
        logger.info(files)
        print(f"files: {files}")

        for file_name in sorted(os.listdir(model_path)):
            # skip hidden files
            if file_name.startswith("."):
                continue

            logger.info(file_name)
            if file_name in files:
                # convert safetensors to safetensors(paddle)
                translate_one_safetensors(file_name)
            else:
                # copy config.json and other files
                shutil.copy(
                    os.path.join(model_path, file_name),
                    os.path.join(dst_path, file_name),
                )

        json.dump(
            dst_index,
            open(os.path.join(dst_path, "model.safetensors.index.json"), "w"),
            indent=2,
        )

    else:
        for file_name in sorted(os.listdir(model_path)):
            # skip hidden files
            if file_name.startswith("."):
                continue

            logger.info(file_name)
            if file_name == "model.safetensors":
                # convert safetensors to safetensors(paddle)
                translate_one_safetensors(file_name)
            else:
                # copy config.json and other files
                shutil.copy(
                    os.path.join(model_path, file_name),
                    os.path.join(dst_path, file_name),
                )

    execute_cmd(
        cmd="sed -i -e  's/torch_dtype/dtype/g' ",
        file_path=os.path.join(dst_path, "config.json"),
    )

    execute_cmd(
        cmd="sed -i /transformers_version/d ",
        file_path=os.path.join(dst_path, "config.json"),
    )

    logger.info(model_path)
    logger.info(dst_path)


if __name__ == "__main__":
    main()
