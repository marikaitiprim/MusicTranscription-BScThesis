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

class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50):

        super(SincConv_fast,self).__init__()

        if in_channels != 1:
            #msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1
            
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 21  #30 
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        # mel = np.linspace(self.to_mel(low_hz),
        #                   self.to_mel(high_hz),
        #                   self.out_channels + 1)
        # hz = self.to_hz(mel)

        hz = np.linspace(low_hz,
                         high_hz,
                         self.out_channels + 1)
        

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        #self.window_ = torch.hamming_window(self.kernel_size)
        n_lin=torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2))) # computing only half of the window
        self.window_=0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size)  #hamming window


        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2*math.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate # Due to symmetry, I only need half of the time axes

 


    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz  + torch.abs(self.low_hz_)
        
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),self.min_low_hz,self.sample_rate/2)
        band=(high-low)[:,0]
        
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left=((torch.sin(f_times_t_high)-torch.sin(f_times_t_low))/(self.n_/2))*self.window_ # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations. 
        band_pass_center = 2*band.view(-1,1)
        band_pass_right= torch.flip(band_pass_left,dims=[1])
        
        
        band_pass=torch.cat([band_pass_left,band_pass_center,band_pass_right],dim=1)

        
        band_pass = band_pass / (2*band[:,None])
        

        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                        bias=None, groups=1)


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
segment_samples = 16000*9  #sample_rate*duration (of every batch)


top_db = None
ref = 1.0
n_mels = 128
mel_bins = 229
amin = 1e-10

# Load 9 seconds of a file, starting 6 seconds in
(audio, _) = load_audio("../resources/Cmajor-logic.mp3", sr=sample_rate, mono=True)

audio = audio[None, :]  # (1, audio_samples)

print(hop_size)

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

    # Save each batch as a WAV file using soundfile
    for i, waveform in enumerate(batch_waveform):
        waveform_np = waveform.cpu().numpy()  # Convert to NumPy array
    #     soundfile.write(f'waveform{pointer+i}.wav', waveform_np, sample_rate)
    #     (baudio, _) = load_audio(f"waveform{pointer+i}.wav", sr=sample_rate, mono=True)

        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        librosa.display.waveplot(waveform_np, sr=sample_rate)
        plt.title('Original Audio')
        # plt.show()
        

    with torch.no_grad():

        logmel_extractor = LogmelFilterBank(sr=sample_rate, 
        n_fft=win_length, n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, 
        amin=amin, top_db=top_db, freeze_parameters=True)           #is_log = True


        sinc_extractor = SincConv_fast(out_channels=win_length//2 + 1, kernel_size=win_length, sample_rate=sample_rate, 
                                    min_low_hz=begin_note, min_band_hz=20, stride=hop_size, padding= win_length // 2 +1)
        sinc = sinc_extractor.forward(batch_waveform.unsqueeze(1))
        sinc = abs(sinc) ** 2                           #power
        sinc = sinc[:, None, :, :].transpose(2, 3)

        sinc = logmel_extractor.forward(sinc)

        print("output of sinc_conv: ", sinc.size())

        # Spectrogram
        spectrogram_extractor = Spectrogram(n_fft=win_length, hop_length=hop_size, win_length=win_length, window=window, 
                    center=center, pad_mode=pad_mode, freeze_parameters=True)
        sp = spectrogram_extractor.forward(batch_waveform)   # (batch_size, 1, time_steps, freq_bins=n_fft // 2 + 1)  

        sp = logmel_extractor.forward(sp)

        print("output of spectrogram: ", sp.size())


        # Convert to numpy array
        spectrogram = sp[0,0].transpose(0,1).numpy()
        sinc_spectrogram = sinc[0,0].transpose(0,1).numpy()


        # # Display the spectrogram of librosa
        # # plt.figure(figsize=(10, 4))
        # plt.subplot(3, 1, 2)
        # librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max),
        #                         y_axis='linear', x_axis='s', sr=sample_rate, hop_length=hop_size)
        # plt.title('Spectrogram')
        # plt.colorbar(format='%+2.0f dB')
        # # plt.show()

        # # Display the spectrogram of sincConv
        # # plt.figure(figsize=(10, 4))
        # plt.subplot(3, 1, 3)
        # librosa.display.specshow(librosa.power_to_db(sinc_spectrogram, ref=np.max),
        #                         y_axis='linear', x_axis='s', sr=sample_rate, hop_length=hop_size)    
        # plt.title('Spectrogram')
        # plt.colorbar(format='%+2.0f dB')
        # plt.tight_layout()
        # plt.show()


        # Display the logmelspectrogram of librosa
        # plt.figure(figsize=(10, 4))
        plt.subplot(3, 1, 2)
        librosa.display.specshow(spectrogram,
                                y_axis='linear', x_axis='s', sr=sample_rate, hop_length=hop_size, cmap='viridis')
        plt.title('Spectrogram')
        plt.colorbar(format='%+2.0f dB')
        # plt.show()

        # Display the spectrogram of sincConv
        # plt.figure(figsize=(10, 4))
        plt.subplot(3, 1, 3)
        librosa.display.specshow(sinc_spectrogram,
                                y_axis='linear', x_axis='s', sr=sample_rate, hop_length=hop_size, cmap='viridis')    
        plt.title('Spectrogram')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.show()











# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import numpy as np

# class SincConv(nn.Module):
#     def __init__(self, out_channels, kernel_size, sample_rate=16000, min_low_hz=50, min_band_hz=50):
#         super(SincConv, self).__init__()

#         if kernel_size % 2 == 0:
#             kernel_size += 1  # Ensure odd kernel size for symmetry

#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.sample_rate = sample_rate
#         self.min_low_hz = min_low_hz
#         self.min_band_hz = min_band_hz

#         # Experiment with different low and high frequencies
#         low_hz_values = [50, 100, 200]  # Example values, you can adjust
#         high_hz_values = [500, 1000, 2000]  # Example values, you can adjust

#         # Create a set of filters for each combination of low and high frequencies
#         filters = []
#         for low_hz in low_hz_values:
#             for high_hz in high_hz_values:
#                 filters.append(self.create_sinc_filter(low_hz, high_hz))

#         self.filters = nn.ParameterList(filters)

#     def create_sinc_filter(self, low_hz, high_hz):
#         low = torch.abs(low_hz)
#         high = torch.clamp(torch.abs(high_hz), self.min_low_hz, self.sample_rate / 2 - self.min_band_hz)
#         band = high - low

#         n = (torch.arange(0, self.kernel_size) - (self.kernel_size - 1) / 2).float()
#         window = 0.54 - 0.46 * torch.cos(2 * np.pi * n / self.kernel_size)

#         sinc_filter = band * torch.sinc(2 * np.pi * high / self.sample_rate * n)
#         sinc_filter = sinc_filter * window

#         return sinc_filter.view(1, 1, -1)

#     def forward(self, waveforms):
#         outputs = []
#         for filter in self.filters:
#             outputs.append(F.conv1d(waveforms, filter, stride=1, padding=self.kernel_size // 2))

#         return torch.cat(outputs, dim=1)


# # Example usage
# waveform = torch.randn(1, 16000)
# sinc_conv = SincConv(out_channels=9, kernel_size=251)
# output = sinc_conv(waveform)

# # Visualize the filters and their responses
# plt.figure(figsize=(12, 6))
# for i, filter in enumerate(sinc_conv.filters):
#     plt.subplot(3, 3, i + 1)
#     plt.plot(filter.view(-1).numpy())
#     plt.title(f'Filter {i + 1}')

# plt.tight_layout()
# plt.show()

# # Visualize the output
# plt.figure(figsize=(12, 6))
# for i in range(output.size(1)):
#     plt.subplot(3, 3, i + 1)
#     plt.plot(output[0, i].detach().numpy())
#     plt.title(f'Filter Response {i + 1}')

# plt.tight_layout()
# plt.show()
