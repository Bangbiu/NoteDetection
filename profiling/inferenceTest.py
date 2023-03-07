import time
from predict import toSpectro, spec_to_input, predict
from models import *
import numpy as np

# from tqdm import tqdm
# import matplotlib.pyplot as plt

model = load_model()

sr = 48000
audio = np.zeros([int(windowSize * sr)])


print("Start 100 Inference:")

# inputData = torch.randn(1,1,47,256)
# Record Runtime Start

with torch.autograd.profiler.profile(use_cuda=False) as prof:
    cts = time.time();
    for i in range(100):
        spec = toSpectro(audio, sr)
        model(spec_to_input(spec))
    print("Time Elapsed: {}".format(time.time() - cts))

print(prof)
