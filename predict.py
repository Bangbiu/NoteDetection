import librosa
import numpy as np
import skimage.io
from torchvision import transforms
from PIL import Image

from models import *

model = load_model()

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def toNSpectro(audio, sr, spp = 512):

    hl = spp  # number of samples per time-step in spectrogram
    hi = 256  # Height of image
    mels = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=hi, fmax=8000,hop_length=hl)
    mels = np.log(mels + 1e-9)  # add small number to avoid log(0)
    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mels, 0, 255).astype(np.uint8)
    # img = 255-img # invert. make black==more energy
    img = np.flip(img, axis=0).astype(np.uint8)

    return img

def toSpectro_Warp(audio, sr, spp = 512):
    hl = spp  # number of samples per time-step in spectrogram
    hi = 256  # Height of image

    mels = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=hi, fmax=8000, hop_length=hl)

    mels = np.log(mels + 1e-9)  # add small number to avoid log(0)
    # invert. make black==more energy
    img = 255 - np.flip(mels, axis=0).astype(np.uint8)

    return img

def toSpectro(audio, sr, spp = 512, imgH = 256):
    mels = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=imgH, fmax=8000, hop_length=spp, norm = None)
    mels = np.log(mels + 1e-9)#.astype(np.uint8)
    # [-30, 20] + 30 -> [0, 50] * 5.1 -> [0, 255]
    mels = ((mels + 30) * 5.1).astype(np.uint8)

    return mels


def spec_to_input(spec, normalize = True):
    spec = torch.tensor(spec).float()
    if normalize:
        spec = torch.divide(spec, 255)
    spec = torch.unsqueeze(spec, dim=0)
    spec = torch.unsqueeze(spec, dim=0)
    return spec


# with Shape (1,img_x, n>img_y)
def predict(x):
    inputs = x.split(img_x, 3)
    inputs = inputs[:len(inputs) - 1]

    print(inputs[0].shape)

    for x in inputs:
        x.to(device)
        outputs = model(x)
        predict_y = torch.round(outputs)
        print(predict_y.tolist())

if __name__ == "__main__":
    # Loading Data

    smp_audio_path = os.path.join(root, "notesrc", "1.wav")
    audio, sr = librosa.load(smp_audio_path, sr=None)

    slice_len = int(windowSize * sr)
    aud_slice = audio[:slice_len]

    spec_1 = toSpectro(aud_slice, sr)
    spec_1 = spec_to_input(spec_1)
    # skimage.io.imsave(spec_path + "/smp.png", spec_1)

    output = model(spec_1)
    output = output.detach().numpy()
    print(output)

