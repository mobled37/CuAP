import torch
import timm
import clip
import cv2
import librosa
import numpy as np
import torchvision.utils as tvu
from PIL import Image


class AudioEncoder(torch.nn.Module):
    def __init__(self, backbone_name="resnet18"):
        super(AudioEncoder, self).__init__()
        self.backbone_name = backbone_name
        self.conv = torch.nn.Conv2d(1, 3, (3, 3))
        self.feature_extractor = timm.create_model(self.backbone_name, num_classes=512, pretrained=True)

    def forward(self, x):
        x = x.float().to("cuda")
        x = self.conv(x)
        x = self.feature_extractor(x)
        return x

def ImageEncoder(image):
    model, preprocess = clip.load("ViT-B/32", device="cuda")
    upsample = torch.nn.Upsample(scale_factor=7)
    avg_pool = torch.nn.AvgPool2d(kernel_size=256 // 32)

    image = avg_pool(upsample(image))
    image_features = model.encode_image(image).float()
    return image_features

if __name__=="__main__":
    audio_path = "audiosample/thunderstorm.wav"
    image_path = "data/imgs/church1.png"

    batch_size_test = 1
    n = batch_size_test

    # Load Audio
    y , sr = librosa.load(audio_path, sr=44100)
    n_mels = 128
    time_length = 864
    audio_inputs = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)

    zero = np.zeros((n_mels, time_length))
    resize_resolution = 512
    h, w = audio_inputs.shape

    if w >= time_length:
        j = 0
        j = random.randint(0, w - time_length)
        audio_inputs = audio_inputs[:, j : j + time_length]
    else:
        zero[:,:w] = audio_inputs[:,:w]
        audio_inputs = zero

    audio_inputs = cv2.resize(audio_inputs, (n_mels, resize_resolution))
    audio_inputs = np.array([audio_inputs])
    audio_inputs = torch.from_numpy(audio_inputs.reshape(1, 1, n_mels, resize_resolution))
    audio_inputs = audio_inputs.to("cuda:0")

    # Load Image
    img = Image.open(image_path)
    img = img.resize((256, 256), Image.ANTIALIAS)
    img = np.array(img)/255
    img = torch.from_numpy(img).type(torch.FloatTensor).permute(2, 0, 1).unsqueeze(0).repeat(n, 1, 1, 1)
    img = img.to("cuda:0")

    # Calculate Latent
    audio_encoder = AudioEncoder()
    audio_encoder.load_state_dict(torch.load("pretrained/resnet18_57.pth"))
    audio_encoder = audio_encoder.to("cuda:0")

    audio_features = audio_encoder(audio_inputs).float()
    image_features = ImageEncoder(img).float()

    print(audio_features.shape)
    print(image_features.shape)

    print(img.shape)
    x0 = (img - 0.5) * 2
    print(x0.shape)