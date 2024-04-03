import os.path

import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset

from constants import *


class UrbanSoundDataset(Dataset):
    def __init__(self, annotation_file, audio_directory, available_device, transform, sample_rate, num_samples):
        self.annotation_file = pd.read_csv(annotation_file)
        self.audio_directory = audio_directory
        self.device = available_device
        self.transform = transform.to(self.device)
        self.sample_rate = sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotation_file)

    def __getitem__(self, index):
        audio = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio)
        signal = signal.to(self.device)
        signal = self._resample(signal, sr)
        signal = self._mix_down(signal)
        signal = self._cut(signal)
        signal = self._right_pad(signal)
        signal = self.transform(signal)
        return signal, label

    def _get_audio_sample_path(self, index):
        fold = f'fold{self.annotation_file.iloc[index, 5]}'
        path = os.path.join(self.audio_directory, fold, self.annotation_file.iloc[index, 0])
        return path

    def _get_audio_sample_label(self, index):
        return self.annotation_file.iloc[index, 6]

    def _resample(self, signal, sr):
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            signal = resampler(signal)
        return signal

    @staticmethod
    def _mix_down(signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _cut(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad(self, signal):
        signal_length = signal.shape[1]
        if signal_length < self.num_samples:
            num_missing_samples = self.num_samples - signal_length
            last_dim_padding = (0, num_missing_samples)
            signal = F.pad(input=signal, pad=last_dim_padding)
        return signal


if __name__ == '__main__':
    dataset = UrbanSoundDataset(annotation_file=ANNOTATION_FILE,
                                audio_directory=AUDIO_DIRECTORY,
                                available_device=device,
                                transform=mel_spectrogram,
                                sample_rate=SAMPLE_RATE,
                                num_samples=NUM_SAMPLES)

    print(f"There are {len(dataset)} samples in the dataset.")
