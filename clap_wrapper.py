import sys

import torch
from transformers import ClapModel, ClapProcessor

from config import config

models = dict()
processor = ClapProcessor.from_pretrained("./emotional/clap-htsat-fused")


def get_clap_audio_feature(audio_data, device=config.bert_gen_config.device):
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    if not device:
        device = "cuda"
    if device not in models.keys():
        models[device] = ClapModel.from_pretrained("./emotional/clap-htsat-fused").to(
            device
        )
    with torch.no_grad():
        inputs = processor(
            audios=audio_data, return_tensors="pt", sampling_rate=48000
        ).to(device)
        emb = models[device].get_audio_features(**inputs)
    return emb.T


def get_clap_text_feature(text, device=config.bert_gen_config.device):
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    if not device:
        device = "cuda"
    if device not in models.keys():
        models[device] = ClapModel.from_pretrained("./emotional/clap-htsat-fused").to(
            device
        )
    if text == "开心":
        text = "happy, enthusiastic, joyful"
    elif text == "伤心":
        text = "sad, gloomy, devastating"
    elif text == "恐惧":
        text = "fearful, anxiety, scared"
    elif text == "愤怒":
        text = "angry, mad, hysteric"
    elif text == "平静":
        text = "calm, normally, peaceful"
    elif text == "低语":
        text = "whisper, slightly, calmly"
    else:
        pass
    with torch.no_grad():
        inputs = processor(text=text, return_tensors="pt").to(device)
        emb = models[device].get_text_features(**inputs)
    return emb.T
