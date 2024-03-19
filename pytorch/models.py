import os
import sys
import math
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from pytorch_utils import move_data_to_device

import librosa.display
import librosa

import soundfile

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


def init_gru(rnn):
    """Initialize a GRU layer. """
    
    def _concat_init(tensor, init_funcs):
        (length, fan_out) = tensor.shape
        fan_in = length // len(init_funcs)
    
        for (i, init_func) in enumerate(init_funcs):
            init_func(tensor[i * fan_in : (i + 1) * fan_in, :])
        
    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))
    
    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, 'weight_ih_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_ih_l{}'.format(i)), 0)

        _concat_init(
            getattr(rnn, 'weight_hh_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_hh_l{}'.format(i)), 0)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, momentum):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels, momentum)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        """
        Args:
          input: (batch_size, in_channels, time_steps, freq_bins)

        Outputs:
          output: (batch_size, out_channels, classes_num)
        """

        x = F.relu_(self.bn1(self.conv1(input)))
        x = F.relu_(self.bn2(self.conv2(x)))
        
        if pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        
        return x


class AcousticModelCRnn8Dropout(nn.Module):
    def __init__(self, classes_num, midfeat, momentum):
        super(AcousticModelCRnn8Dropout, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=48, momentum=momentum)
        self.conv_block2 = ConvBlock(in_channels=48, out_channels=64, momentum=momentum)
        self.conv_block3 = ConvBlock(in_channels=64, out_channels=96, momentum=momentum)
        self.conv_block4 = ConvBlock(in_channels=96, out_channels=128, momentum=momentum)

        self.fc5 = nn.Linear(midfeat, 768, bias=False)
        self.bn5 = nn.BatchNorm1d(768, momentum=momentum)

        self.gru = nn.GRU(input_size=768, hidden_size=256, num_layers=2, 
            bias=True, batch_first=True, dropout=0., bidirectional=True)

        self.fc = nn.Linear(512, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_layer(self.fc5)
        init_bn(self.bn5)
        init_gru(self.gru)
        init_layer(self.fc)

    def forward(self, input):
        """
        Args:
          input: (batch_size, channels_num, time_steps, freq_bins)

        Outputs:
          output: (batch_size, time_steps, classes_num)
        """

        # x = self.conv_block1(input, pool_size=(1, 2), pool_type='avg')            
        # x = F.dropout(x, p=0.2, training=self.training)
        # x = self.conv_block2(x, pool_size=(1, 2), pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)
        # x = self.conv_block3(x, pool_size=(1, 2), pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)
        # x = self.conv_block4(x, pool_size=(1, 2), pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)

        # x = x.transpose(1, 2).flatten(2)
        # x = F.relu(self.bn5(self.fc5(x).transpose(1, 2)).transpose(1, 2))
        # x = F.dropout(x, p=0.5, training=self.training, inplace=True)
        
        # (x, _) = self.gru(x)
        # x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        # output = torch.sigmoid(self.fc(x))
        # return output
        
        x = self.conv_block1(input, pool_size=(1, 2), pool_type='avg')            
        x = F.dropout(x.clone(), p=0.2, training=self.training)  # Use clone() to avoid inplace
        x = self.conv_block2(x.clone(), pool_size=(1, 2), pool_type='avg')  # Use clone() to avoid inplace
        x = F.dropout(x.clone(), p=0.2, training=self.training)  # Use clone() to avoid inplace
        x = self.conv_block3(x.clone(), pool_size=(1, 2), pool_type='avg')  # Use clone() to avoid inplace
        x = F.dropout(x.clone(), p=0.2, training=self.training)  # Use clone() to avoid inplace
        x = self.conv_block4(x.clone(), pool_size=(1, 2), pool_type='avg')  # Use clone() to avoid inplace
        x = F.dropout(x.clone(), p=0.2, training=self.training)  # Use clone() to avoid inplace

        x = x.transpose(1, 2).flatten(2)
        x = F.relu(self.bn5(self.fc5(x).transpose(1, 2)).transpose(1, 2)).clone().detach()
        x = F.dropout(x, p=0.5, training=self.training)
        
        (x, _) = self.gru(x)
        x = F.dropout(x.clone(), p=0.5, training=self.training)  # Use clone() to avoid inplace
        output = torch.sigmoid(self.fc(x))

        return output

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
        low_hz = 30
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
        self.window_=0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size)


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


class Regress_onset_offset_frame_velocity_CRNN(nn.Module):
    def __init__(self, frames_per_second, classes_num):
        super(Regress_onset_offset_frame_velocity_CRNN, self).__init__()

        sample_rate = 16000
        window_size = 2048
        frames_per_second = 100
        hop_size = sample_rate // frames_per_second
        mel_bins = 229
        fmin = 30
        fmax = sample_rate // 2
        begin_note = 21
        classes_num = 88 

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        midfeat = 1792 
        momentum = 0.01

#########################################################################################################

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, 
            hop_length=hop_size, win_length=window_size, window=window, 
            center=center, pad_mode=pad_mode, freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, 
            n_fft=window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, 
            amin=amin, top_db=top_db, freeze_parameters=True)

        self.sincconv = SincConv_fast(out_channels=window_size//2 + 1, kernel_size=window_size, sample_rate=sample_rate, 
                                    min_low_hz=begin_note, min_band_hz=20, stride=hop_size, padding= window_size // 2 +1)

#############################################################################################################

        self.bn0 = nn.BatchNorm2d(mel_bins, momentum)

        self.frame_model = AcousticModelCRnn8Dropout(classes_num, midfeat, momentum)
        self.reg_onset_model = AcousticModelCRnn8Dropout(classes_num, midfeat, momentum)
        self.reg_offset_model = AcousticModelCRnn8Dropout(classes_num, midfeat, momentum)
        self.velocity_model = AcousticModelCRnn8Dropout(classes_num, midfeat, momentum)

        self.reg_onset_gru = nn.GRU(input_size=88 * 2, hidden_size=256, num_layers=1, 
            bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.reg_onset_fc = nn.Linear(512, classes_num, bias=True)

        self.frame_gru = nn.GRU(input_size=88 * 3, hidden_size=256, num_layers=1, 
            bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.frame_fc = nn.Linear(512, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_gru(self.reg_onset_gru)
        init_gru(self.frame_gru)
        init_layer(self.reg_onset_fc)
        init_layer(self.frame_fc)
 
    def forward(self, input):
        """
        Args:
          input: (batch_size, data_length)

        Outputs:
          output_dict: dict, {
            'reg_onset_output': (batch_size, time_steps, classes_num),
            'reg_offset_output': (batch_size, time_steps, classes_num),
            'frame_output': (batch_size, time_steps, classes_num),
            'velocity_output': (batch_size, time_steps, classes_num)
          }
        """
#################################################################################################################
        sample_rate = 16000
        frames_per_second = 100
        hop_size = sample_rate // frames_per_second

        # for i, waveform in enumerate(input):
        #     waveform_np = waveform.cpu().numpy()  # Convert to NumPy array
        #     soundfile.write(f'test.wav', waveform_np, sample_rate)

        #     plt.figure(figsize=(12, 8))
        #     plt.subplot(3, 1, 1)
        #     librosa.display.waveplot(waveform_np, sr=sample_rate)
        #     plt.title('Original Audio')
        #     break

        sinc = self.sincconv(input.unsqueeze(1))
        sinc = abs(sinc) ** 2
        x = sinc[:, None, :, :].transpose(2, 3)

        sp = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins) 

        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        sp = self.logmel_extractor(sp)

        # Convert to numpy array
        # spectrogram = sp[0,0].transpose(0,1).cpu().detach().numpy()
        # sinc_spectrogram = x[0,0].transpose(0,1).cpu().detach().numpy()

        # # Display the logmelspectrogram of librosa
        # plt.subplot(3, 1, 2)
        # librosa.display.specshow(spectrogram,
        #                         y_axis='linear', x_axis='s', sr=sample_rate, hop_length=hop_size, cmap='viridis')
        # plt.title('Original method Log-mel Spectrogram')
        # plt.colorbar(format='%+2.0f dB')

        # # Display the spectrogram of sincConv
        # plt.subplot(3, 1, 3)
        # librosa.display.specshow(sinc_spectrogram,
        #                         y_axis='linear', x_axis='s', sr=sample_rate, hop_length=hop_size, cmap='viridis')    
        # plt.title('Proposed method Log-Mel Spectrogram')
        # plt.colorbar(format='%+2.0f dB')
        # plt.tight_layout()

        # plt.savefig('proposed.png', format='png')

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

#################################################################################################################

        frame_output = self.frame_model(x)  # (batch_size, time_steps, classes_num)
        reg_onset_output = self.reg_onset_model(x)  # (batch_size, time_steps, classes_num)
        reg_offset_output = self.reg_offset_model(x)    # (batch_size, time_steps, classes_num)
        velocity_output = self.velocity_model(x)    # (batch_size, time_steps, classes_num)
 
        # Use velocities to condition onset regression
        x = torch.cat((reg_onset_output, (reg_onset_output ** 0.5) * velocity_output.detach()), dim=2)
        (x, _) = self.reg_onset_gru(x)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        reg_onset_output = torch.sigmoid(self.reg_onset_fc(x))
        """(batch_size, time_steps, classes_num)"""

        # Use onsets and offsets to condition frame-wise classification
        x = torch.cat((frame_output, reg_onset_output.detach(), reg_offset_output.detach()), dim=2)
        (x, _) = self.frame_gru(x)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        frame_output = torch.sigmoid(self.frame_fc(x))  # (batch_size, time_steps, classes_num)
        """(batch_size, time_steps, classes_num)"""

        output_dict = {
            'reg_onset_output': reg_onset_output, 
            'reg_offset_output': reg_offset_output, 
            'frame_output': frame_output, 
            'velocity_output': velocity_output}

        return output_dict


class Regress_pedal_CRNN(nn.Module):
    def __init__(self, frames_per_second, classes_num):
        super(Regress_pedal_CRNN, self).__init__()

        sample_rate = 16000
        window_size = 2048
        hop_size = sample_rate // frames_per_second
        mel_bins = 229
        fmin = 30
        fmax = sample_rate // 2

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        midfeat = 1792
        momentum = 0.01

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, 
            hop_length=hop_size, win_length=window_size, window=window, 
            center=center, pad_mode=pad_mode, freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, 
            n_fft=window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, 
            amin=amin, top_db=top_db, freeze_parameters=True)

        self.bn0 = nn.BatchNorm2d(mel_bins, momentum)

        self.reg_pedal_onset_model = AcousticModelCRnn8Dropout(1, midfeat, momentum)
        self.reg_pedal_offset_model = AcousticModelCRnn8Dropout(1, midfeat, momentum)
        self.reg_pedal_frame_model = AcousticModelCRnn8Dropout(1, midfeat, momentum)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        
    def forward(self, input):
        """
        Args:
          input: (batch_size, data_length)

        Outputs:
          output_dict: dict, {
            'reg_onset_output': (batch_size, time_steps, classes_num),
            'reg_offset_output': (batch_size, time_steps, classes_num),
            'frame_output': (batch_size, time_steps, classes_num),
            'velocity_output': (batch_size, time_steps, classes_num)
          }
        """

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        reg_pedal_onset_output = self.reg_pedal_onset_model(x)  # (batch_size, time_steps, classes_num)
        reg_pedal_offset_output = self.reg_pedal_offset_model(x)  # (batch_size, time_steps, classes_num)
        pedal_frame_output = self.reg_pedal_frame_model(x)  # (batch_size, time_steps, classes_num)
        
        output_dict = {
            'reg_pedal_onset_output': reg_pedal_onset_output, 
            'reg_pedal_offset_output': reg_pedal_offset_output,
            'pedal_frame_output': pedal_frame_output}

        return output_dict


# This model is not trained, but is combined from the trained note and pedal models.
class Note_pedal(nn.Module):
    def __init__(self, frames_per_second, classes_num):
        """The combination of note and pedal model.
        """
        super(Note_pedal, self).__init__()

        self.note_model = Regress_onset_offset_frame_velocity_CRNN(frames_per_second, classes_num)
        self.pedal_model = Regress_pedal_CRNN(frames_per_second, classes_num)

    def load_state_dict(self, m, strict=False):
        self.note_model.load_state_dict(m['note_model'], strict=strict)
        self.pedal_model.load_state_dict(m['pedal_model'], strict=strict)

    def forward(self, input):
        note_output_dict = self.note_model(input)
        pedal_output_dict = self.pedal_model(input)

        full_output_dict = {}
        full_output_dict.update(note_output_dict)
        full_output_dict.update(pedal_output_dict)
        return full_output_dict