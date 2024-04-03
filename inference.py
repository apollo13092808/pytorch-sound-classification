from constants import *
from dataset import UrbanSoundDataset
from model import NeuralNetwork

class_mapping = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]


def predict(model, feature, target, class_mapping):
    model.eval()
    with torch.inference_mode():
        predictions = model(feature)
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected


if __name__ == "__main__":
    model = NeuralNetwork()
    state_dict = torch.load("model.pth")
    model.load_state_dict(state_dict)

    dataset = UrbanSoundDataset(annotation_file=ANNOTATION_FILE,
                                audio_directory=AUDIO_DIRECTORY,
                                available_device=device,
                                transform=mel_spectrogram,
                                sample_rate=SAMPLE_RATE,
                                num_samples=NUM_SAMPLES)

    feature, target = dataset[0][0], dataset[0][1]
    feature.unsqueeze_(0)

    predicted, expected = predict(model, feature, target, class_mapping)
    print(f"Predicted: '{predicted}'")
    print(f" Expected: '{expected}'")
