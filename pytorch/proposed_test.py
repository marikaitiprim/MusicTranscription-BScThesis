from torchlibrosa.stft import Spectrogram, LogmelFilterBank, STFT
import torchlibrosa as tl
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import soundfile
import librosa
import audioread
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import time
from utilities import pad_truncate_sequence
from models import Regress_onset_offset_frame_velocity_CRNN as Model

def load_audio(path, sr=22050, mono=True, offset=0.0, duration=None,
    dtype=np.float32, res_type='kaiser_best', 
    backends=[audioread.ffdec.FFmpegAudioFile]):
    """Load audio. Copied from librosa.core.load() except that ffmpeg backend is 
    always used in this function."""

    y = []
    with audioread.audio_open(os.path.realpath(path), backends=backends) as input_file:
        sr_native = input_file.samplerate
        n_channels = input_file.channels

        s_start = int(np.round(sr_native * offset)) * n_channels

        if duration is None:
            s_end = np.inf
        else:
            s_end = s_start + (int(np.round(sr_native * duration))
                               * n_channels)

        n = 0

        for frame in input_file:
            frame = librosa.core.audio.util.buf_to_float(frame, dtype=dtype)
            n_prev = n
            n = n + len(frame)

            if n < s_start:
                # offset is after the current frame
                # keep reading
                continue

            if s_end < n_prev:
                # we're off the end.  stop reading
                break

            if s_end < n:
                # the end is in this frame.  crop.
                frame = frame[:s_end - n_prev]

            if n_prev <= s_start <= n:
                # beginning is in this frame
                frame = frame[(s_start - n_prev):]

            # tack on the current frame
            y.append(frame)

    if y:
        y = np.concatenate(y)

        if n_channels > 1:
            y = y.reshape((-1, n_channels)).T
            if mono:
                y = librosa.core.audio.to_mono(y)

        if sr is not None:
            y = librosa.core.audio.resample(y, sr_native, sr, res_type=res_type)

        else:
            sr = sr_native

    # Final cleanup for dtype and contiguity
    y = np.ascontiguousarray(y, dtype=dtype)

    return (y, sr)

def enframe(x, segment_samples):
    """Enframe long sequence to short segments.

    Args:
        x: (1, audio_samples)
        segment_samples: int

    Returns:
        batch: (N, segment_samples)
    """
    assert x.shape[1] % segment_samples == 0
    batch = []

    pointer = 0
    while pointer + segment_samples <= x.shape[1]:
        batch.append(x[:, pointer : pointer + segment_samples])
        pointer += segment_samples // 2

    batch = np.concatenate(batch, axis=0)
    return batch

def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)

sample_rate = 16000
win_length = 2048
begin_note = 21     # MIDI note of A0, the lowest note of a piano.
frames_per_second = 100  # 1 frame = 10ms --> 100 frames = 1000ms = 1 sec
hop_size = sample_rate // frames_per_second
fmin = 30
fmax = sample_rate // 2
window = 'hann'
center = True
pad_mode = 'reflect'
segment_samples = 16000*3  #sample_rate*duration (of every batch)


top_db = None
ref = 1.0
n_mels = 128
mel_bins = 229
amin = 1e-10

model = Model(frames_per_second=100, classes_num=88)

full_checkpoint = torch.load('../midifiles/piano_trans/checkpoints/main/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=none/max_note_shift=0/batch_size=11/100000_iterations-pretrained.pth')
note_model_weights = full_checkpoint['model']
model.load_state_dict(note_model_weights)

# Load 9 seconds of a file, starting 6 seconds in
(audio, _) = load_audio("../waltz.mp3", sr=sample_rate, mono=True)        

audio = audio[None, :]  # (1, audio_samples)

# Pad audio to be evenly divided by segment_samples
audio_len = audio.shape[1]
pad_len = int(np.ceil(audio_len / segment_samples)) \
    * segment_samples - audio_len

audio = np.concatenate((audio, np.zeros((1, pad_len))), axis=1)

# Enframe to segments
segments = enframe(audio, segment_samples)
"""(N, segment_samples)"""

batch_size=1            #inference.py
pointer = 0
while True:                 #pytorch_utils.py
    if pointer >= len(segments):
        break

    batch_waveform = move_data_to_device(segments[pointer : pointer + batch_size], 'cpu')
    pointer += batch_size

    batch_output_dict = model(batch_waveform)

    break