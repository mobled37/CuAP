import pandas as pd
import os
from tqdm import tqdm
from glob import glob
import yt_dlp as youtube_dl
import librosa
from pydub import AudioSegment
from multiprocessing import Pool
import argparse

# make arg parser

parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', type=str, default='vggsound.csv', help='csv file path')
args = parser.parse_args()

def trim_audio_data(audio_file, save_file, start):
    sr = 44100

    y, sr = librosa.load(audio_file, sr=sr)
    print("Save!")
    ny = y[sr*start:sr*(start+10)]
    librosa.write_wav(save_file + '.wav', ny, sr)

# ydl_opts = {
#     'format': 'bestaudio/best',
#     'postprocessors': [{
#         'key': 'FFmpegExtractAudio',
#         'preferredcodec': 'mp3',
#         'preferredquality': '320',
#     }],
# }
path0 = args.csv_path.split(".")[0] # vggsound0
ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '320',
    }],
    'outtmpl': f'{path0}/%(title)s.%(ext)s'
    }

# vggsound.csv : https://www.robots.ox.ac.uk/~vgg/data/vggsound/
vgg = pd.read_csv(f"{args.csv_path}", names=["YouTube ID", "start seconds", "label", "train/test split"], skiprows=[0])

slink = "https://www.youtube.com/watch?v="

sumofError = 0
cnt = 0

# os.makedirs("./vggsound", exist_ok=True)

for idx, row in tqdm(enumerate(vgg.iterrows())):
    try:
        _, row = row
        url, sttime, label, split = row["YouTube ID"], row["start seconds"], row["label"], row["train/test split"]
        endtime = int(sttime) + 10

        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([slink + url])

        # Save 10 sec Wav File with Text Prompt

        path = glob(f"{path0}/*.mp3")[0]
        print(path)
        sound = AudioSegment.from_mp3(path)
        sound = sound[int(sttime) * 1000:int(endtime) * 1000]
        print(idx)
        # i want to extract last str in path0
        # ex) path0 = "vggsound0" -> num = 0
        path1 = path0.split("d")[1]
        num = 11080 * int(path1) + int(idx)
        print(num)
        sound.export("/Data/dataset/vggsound2/"+label+str("_")+str(num)+".wav", format="wav")
        os.remove(path)

    except:
        sumofError += 1
        continue

print(sumofError , "The number of error cases")