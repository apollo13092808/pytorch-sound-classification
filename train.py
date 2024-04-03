import torch.nn as nn
from torch.utils.data import DataLoader

from constants import *
from dataset import UrbanSoundDataset
from model import NeuralNetwork


def train_single_epoch(model, dataloader, criterion, optimizer, device):
    loss = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        prediction = model(X)
        loss = criterion(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Loss: {loss.item():.6f}')


def train(model, epochs, dataloader, criterion, optimizer, device):
    for e in range(1, epochs + 1):
        print(f'Epoch: {e:>2}')
        train_single_epoch(model, dataloader, criterion, optimizer, device)
        print('-' * 100)
    print('[INFO] Training completed!')


if __name__ == '__main__':
    dataset = UrbanSoundDataset(annotation_file=ANNOTATION_FILE,
                                audio_directory=AUDIO_DIRECTORY,
                                available_device=device,
                                transform=mel_spectrogram,
                                sample_rate=SAMPLE_RATE,
                                num_samples=NUM_SAMPLES)

    train_dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = NeuralNetwork().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    train(model, EPOCHS, train_dataloader, criterion, optimizer, device)

    torch.save(obj=model.state_dict(), f='model.pth')
    print('[INFO] Model successfully saved!')
