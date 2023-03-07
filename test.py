import librosa
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
from predict import toSpectro
#"notesrc/0/0.wav"

#mels = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=256, fmax=8000, hop_length=512)
#mels = np.log(mels + 1e-9)  # add small number to avoid log(0)

audio, sr = librosa.load("record/录音 (4).wav")
plt.imshow(toSpectro(audio,sr))
plt.show()

#skimage.io.imsave("spectrogram/test.png", spec)