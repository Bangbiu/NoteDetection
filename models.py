import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Directory
root = os.path.dirname(__file__)
spec_path = os.path.join(root, "spectrogram")

# Default Model
model_name = "GZNet_MoreChannel"

# Input Size = Sample Rate / 512 * windowSize
img_x = 47
img_y = 256

# Default Sample Rate: 48000
# Window Size for the Interception (s)
windowSize = 0.5

# Window Moving Step for the Interception (s)
windowStep = 0.05

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(name=model_name):
    weight_path = os.path.join(root, "weights", name + ".pth")
    assert os.path.exists(weight_path), "{} path does not exist.".format(weight_path)
    model = eval(name)(notes=21)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    return model

#@title Classic
class GZNet(nn.Module):
    input_shape = (1, img_x, img_y)

    def __init__(self, notes=21):
        super(GZNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.fc = nn.Linear(in_features=75392, out_features=notes)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))  # output[16, 43, 252]
        x = F.dropout(x, 0.5)
        x = self.maxpool(x)             # output[16, 21, 126]

        x = torch.tanh(self.conv2(x))  # output[32, 19, 124]
        x = F.dropout(x, 0.5)

        x = x.reshape(x.shape[0], -1)  # reshape to 32 * 19 * 124 = 75392
        x = torch.sigmoid(self.fc(x))
        return x

# @title Classical-2Conv-Maxpool
class GZNet_SingeStringFinal(nn.Module):
    input_shape = (1, img_y, img_x)

    def __init__(self, notes=21):
        super(GZNet_SingeStringFinal, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.fc = nn.Linear(in_features=17856, out_features=notes)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))  # output[16, 43, 252]
        x = F.dropout(x, 0.5)
        x = self.maxpool(x)  # output[16, 21, 126]

        x = torch.tanh(self.conv2(x))  # output[32, 19, 124]
        x = F.dropout(x, 0.5)
        x = self.maxpool(x)  # output[32, 62, 9]

        x = x.reshape(x.shape[0], -1)  # reshape to 32 * 19 * 124 = 75392
        x = torch.sigmoid(self.fc(x))
        return x


# @title Classical-MoreChannel
class GZNet_MoreChannel(nn.Module):
    input_shape = (1, img_y, img_x)

    def __init__(self, notes=21):
        super(GZNet_MoreChannel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.fc = nn.Linear(in_features=35712, out_features=notes)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))  # output[16, 43, 252]
        x = F.dropout(x, 0.5)
        x = self.maxpool(x)  # output[16, 21, 126]

        x = torch.tanh(self.conv2(x))  # output[32, 19, 124]
        x = F.dropout(x, 0.5)
        x = self.maxpool(x)  # output[32, 62, 9]

        x = x.reshape(x.shape[0], -1)  # reshape to 32 * 19 * 124 = 75392
        x = torch.sigmoid(self.fc(x))
        return x


#@title Taller-Conv
class GZNet_TallerConv(nn.Module):
  input_shape = (1, img_y, img_x)
  def __init__(self, notes = 21):
    super(GZNet, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1,10), stride=1)
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1,10), stride=1)
    self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)


    self.fc = nn.Linear(in_features=40960, out_features=notes)

  def forward(self, x):
    x = torch.tanh(self.conv1(x))
    x = F.dropout(x,0.5)
    x = self.maxpool(x)
    x = torch.tanh(self.conv2(x))
    x = F.dropout(x, 0.5)

    x = x.reshape(x.shape[0], -1)
    x = torch.sigmoid(self.fc(x))
    return x

#@title Wide-Conv
class GZNet_WideConv(nn.Module):
  input_shape = (1, img_y, img_x)
  def __init__(self, notes = 21):
    super(GZNet, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5,1), stride=1)
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5,1), stride=1)
    self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)


    self.fc = nn.Linear(in_features=89792, out_features=notes)

  def forward(self, x):
    x = torch.tanh(self.conv1(x))
    x = F.dropout(x,0.5)
    x = self.maxpool(x)

    x = torch.tanh(self.conv2(x))
    x = F.dropout(x, 0.5)

    x = x.reshape(x.shape[0], -1)
    x = torch.sigmoid(self.fc(x))
    return x

#@title Tall-Conv
class GZNet_TallConv(nn.Module):
  input_shape = (1, img_y, img_x)
  def __init__(self, notes = 21):
    super(GZNet, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1,5), stride=1)
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1,5), stride=1)
    self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)


    self.fc = nn.Linear(in_features=69632, out_features=notes)

  def forward(self, x):
    x = torch.tanh(self.conv1(x))
    x = F.dropout(x,0.5)
    x = self.maxpool(x)

    x = torch.tanh(self.conv2(x))
    x = F.dropout(x, 0.5)

    x = x.reshape(x.shape[0], -1)
    x = torch.sigmoid(self.fc(x))
    return x

class GZNet_Conv3(nn.Module):
    input_shape = (1, img_x, img_y)

    def __init__(self, notes=21):
        super(GZNet_Conv3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5)
        self.fc = nn.Linear(in_features=49088, out_features=notes)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))  # output[16, 43, 252]
        x = F.dropout(x, 0.5)
        x = self.maxpool(x)  # output[16, 21, 126]

        x = torch.tanh(self.conv2(x))  # output[32, 19, 124]
        x = F.dropout(x, 0.5)

        x = torch.tanh(self.conv3(x))

        x = x.reshape(x.shape[0], -1)  # reshape to 32 * 19 * 124 = 75392
        x = torch.sigmoid(self.fc(x))
        return x



