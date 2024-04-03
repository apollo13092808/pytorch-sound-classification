import torch
import torchaudio

device = (
    'cuda'
    if torch.cuda.is_available()
    else 'mps'
    if torch.backends.mps.is_available()
    else 'cpu'
)

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-4

ANNOTATION_FILE = 'data/UrbanSound8K/metadata/UrbanSound8K.csv'
AUDIO_DIRECTORY = 'data/UrbanSound8K/audio'

SAMPLE_RATE = 22050
NUM_SAMPLES = 22050

mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE,
                                                       n_fft=1024,
                                                       hop_length=512,
                                                       n_mels=64)
