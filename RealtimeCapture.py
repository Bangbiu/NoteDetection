import numpy as np
import pyaudio
import cv2
import soundfile

from predict import toSpectro, toSpectro_Warp, spec_to_input
from models import *


def fadeWindow(in_data):
    fadeSize = 200
    in_data[:fadeSize] = in_data[:fadeSize] * (np.arange(fadeSize) / fadeSize)
    in_data[len(in_data) - fadeSize:] = in_data[len(in_data) - fadeSize:] * (np.arange(fadeSize) / fadeSize)
    return in_data


class AudioHandler(object):
    def __init__(self):
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 48000
        self.CHUNK = 2048 * 2
        self.p = None
        self.stream = None
        self.bufferSize = 256
        self.window = np.zeros([256, self.bufferSize])
        # CHUNK / 512
        self.spec_wid = 5
        self.cache = np.zeros(int(self.RATE * 1))
        self.cacheWid = round(self.RATE * 1 / 512)

    def start(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  output=False,
                                  # stream_callback=self.callback,
                                  frames_per_buffer=self.CHUNK)

    def stop(self):
        self.stream.close()
        self.p.terminate()

    def cacheData(self, in_data):
        cache = self.cache
        cache[:len(cache) - self.CHUNK] = cache[self.CHUNK:]
        cache[len(cache) - self.CHUNK:] = in_data

    def enqueueSpec(self, spec):
        # print(spec.shape)
        sentinel = self.bufferSize - spec.shape[1]
        self.window[:, sentinel:] = spec
        # Push forward
        self.window[:, :sentinel - self.spec_wid] = self.window[:, self.spec_wid:sentinel]
        self.window[:, sentinel - self.spec_wid:sentinel] = spec[:, :self.spec_wid]

    # Input Raw Data
    def parse_chunk(self, in_data):
        return np.frombuffer(np.copy(in_data), dtype=np.float32)

    def load_chunk(self):
        in_data = self.parse_chunk(self.stream.read(self.CHUNK))

        self.cacheData(in_data)

        spec = toSpectro(self.cache, self.RATE)
        spec = np.divide(spec, 255)

        self.enqueueSpec(spec)

        return self.window

    def get_recent_tensor(self, wid=5):
        x = np.clip(self.window[:, (self.bufferSize - wid):], 0.0, 1.0)
        return spec_to_input(x, False)

    def callback(self, in_data, frame_count, time_info, flag):
        self.load_chunk(self.parse_chunk(in_data))
        return None, pyaudio.paContinue


class NoteControllerDisplay:
    def __init__(self, labelNum, width=1000, height=256, cfdWin = 3):
        self.width = width
        self.height = height
        label_width = int(width / labelNum)
        self.label_width = label_width

        self.image = np.zeros([height, width])
        self.labelCache = np.zeros([cfdWin, labelNum])
        self.update(np.zeros(labelNum))


    def update(self, labels):
        lw = self.label_width
        image = self.image
        cache = self.labelCache

        cache[:len(cache) - 1, :] = cache[1:, :]
        cache[len(cache) - 1, :] = labels

        labels = np.round(np.sum(cache, axis=0) / len(cache))

        for x in range(len(labels)):
            field_x = x * lw
            image[:, field_x:(field_x + lw)] = labels[x]
            image = cv2.putText(image, str(x + 1), (int(field_x), int(self.height / 2)), cv2.FONT_HERSHEY_SIMPLEX,
                                1, 255, 2, cv2.LINE_AA)
            image = cv2.rectangle(image, [field_x, 0], [field_x + lw, self.height], 255, 2)

        self.image = image


audio = AudioHandler()
model = load_model()
display = NoteControllerDisplay(labelNum=21)

audio.start()  # open the the stream
# audio.mainloop()  # main operations with librosa
print("Refreshing Rate: {} pix/s".format(audio.spec_wid))

while (True):

    data = audio.load_chunk()
    inputs = audio.get_recent_tensor(img_x)
    outputs = model(inputs).detach().numpy()[0] * .6
    display.update(outputs)

    cv2.imshow('frame', data)
    cv2.imshow('control', display.image)
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# print(len(audio.cache))
# soundfile.write("cache.wav", audio.cache, audio.RATE)
audio.stop()
