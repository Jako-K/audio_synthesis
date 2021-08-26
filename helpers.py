import numpy as np
import matplotlib.pyplot as plt

import datetime
import os
import random

import torch 
import torch.nn as nn
import torchaudio

from IPython.display import Audio
from IPython.core.display import display



def get_model_save_name(to_add: dict, model_name: str, separator: str = "  .  ", include_time: bool = True):

    """
    Function description:
    Adds useful information to model save file such as date, time and metrics.

    Example:
    >> get_model_save_name( {"valid_loss":valid_mean}, "model.pth", "  |  ")
    "time 17.25.32 03-05-2021  |  valid_loss 0.72153  |  model_name.pth"

    @param to_add: Dictionary which contain information which will be added to the model save name e.g. loss
    @param model_name: Actual name of the model. Will be the last thing appended to the save path
    @param separator: The separator symbol used between information e.g. "thing1 <separator> thing2 <separator> ...
    @param include_time: If true, include full date and time  e.g. 17.25.32 03-05-2021 <separator> ...
    """
    return_string = ""
    if include_time:
        time_plus_date = datetime.datetime.now().strftime('%H.%M.%S %d-%m-%Y')
        return_string = f"time {time_plus_date}{separator}" if include_time else ""

    # Adds everything from to_add dict
    for key, value in to_add.items():
        if type(value) in [float, np.float16, np.float32, np.float64]:
            value = f"{value:.5}".replace("+", "")  # Rounding to 5 decimals
        return_string += str(key) + " " + str(value) + separator

    return_string += model_name

    return return_string



def get_parameter_count(model: nn.Module, only_trainable: bool = False, decimals: int = 3):
    """ Number of total or trainable parameters in a pytorch model i.e. nn.Module child """
    if decimals < 1:
        raise ValueError(f"Expected `decimals` >= 1, but received {decimals}")

    if only_trainable:
        temp = sum(p.numel() for p in model.parameters())
    else:
        temp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return format(temp, f".{decimals}E")



def seed_torch(seed: int = 12, deterministic: bool = False):
        """
        Function description:
        Seed python, random, os, bumpy, torch and torch.cuda.

        @param seed: Used to seed everything
        @param deterministic: Set `torch.backends.cudnn.deterministic`. NOTE can drastically increase run time if True

        """

        torch.backends.cudnn.deterministic = deterministic
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


def jupyter_play_audio(path:str, plot:bool = True):
    """ Load a sound and display it if `plot` is True. Use torchaudio, so only support what they do."""

    # Audio load and play
    sound, sample_rate = torchaudio.load(path)
    audio_bar = Audio(path)
    display(audio_bar)

    if plot:
        duration = round(len(sound[0]) / sample_rate, 3)
        plt.plot(sound[0])
        plt.title(f"type: {audio_bar.mimetype} | duration: {duration} s | sample rate: {sample_rate}")
        plt.show()