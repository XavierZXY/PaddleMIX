# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import sys
import warnings

import numpy as np
import paddle
import requests
from decord import VideoReader, cpu
from paddlenlp.transformers import Qwen2Tokenizer
from PIL import Image

from paddlemix.models.llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from paddlemix.models.llava.conversation import SeparatorStyle, conv_templates
from paddlemix.models.llava.language_model.llava_qwen import LlavaQwenForCausalLM
from paddlemix.models.llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from paddlemix.models.llava.multimodal_encoder.siglip_encoder import (
    SigLipImageProcessor,
)

# from paddlemix.models.llava.model.builder import load_pretrained_model

warnings.filterwarnings("ignore")


def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps() / fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [(i / fps) for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(
            0, total_frame_num - 1, sample_fps, dtype=int
        )
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [(i / vr.get_avg_fps()) for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames, frame_time, video_time


def main():
    pretrained = "/home/zxy/.cache/huggingface/hub/models--lmms-lab--LLaVA-Video-7B-Qwen2/snapshots/013210b3aff822f1558b166d39c1046dd109520f_pd"
    # pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
    model_name = "llava_qwen"
    device = "cuda:0,1,2,3,4,5,6,7"
    device_map = "auto"
    #########################################\
    # TODO: Implement load_pretrained_model
    # original code:
    # tokenizer, model, image_processor, max_length = load_pretrained_model(
    #     pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map
    # )
    # after refactoring:
    model = LlavaQwenForCausalLM.from_pretrained(
        pretrained, dtype=paddle.bfloat16
    ).eval()
    tokenizer = Qwen2Tokenizer.from_pretrained(pretrained)
    image_processor = SigLipImageProcessor()

    #########################################
    model.eval()
    video_path = "./videos/IU.mp4"
    max_frames_num = 4
    video, frame_time, video_time = load_video(
        video_path, max_frames_num, 1, force_sample=True
    )
    video = (
        image_processor.preprocess(video, return_tensors="pt")["pixel_values"]
        .cuda(blocking=True)
        .astype(dtype="float16")
    )
    video = [video]
    conv_template = "qwen_1_5"
    time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
    question = (
        DEFAULT_IMAGE_TOKEN
        + f"""
    {time_instruciton}
    Please describe this video in detail."""
    )
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = (
        tokenizer_image_token(
            prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        .unsqueeze(0)
        .to(device)
    )
    cont = model.generate(
        input_ids,
        images=video,
        modalities=["video"],
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[
        0
    ].strip()
    print(text_outputs)


if __name__ == "__main__":
    main()
