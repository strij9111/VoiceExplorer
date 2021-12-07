from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import audioSegmentation as aS
import scipy.io.wavfile as wavfile
from os import remove, system, path
import psycopg2
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torchaudio
import wavencoder
from wavencoder.models import SincNet
from joblib import Parallel, delayed

from os import getenv
from dotenv import load_dotenv

import avec

torchaudio.set_audio_backend("soundfile")


class AudioClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = wavencoder.models.Wav2Vec(pretrained=True)
        self.classifier = nn.Linear(512, 32)

    def forward(self, x):
        z = self.encoder(x)
        z = torch.mean(z, dim=2)
        out = self.classifier(z)
        return out


class SincNetAudioClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SincNet(pretrained=True)
        self.classifier = nn.Linear(2048, 32)

    def forward(self, x):
        z = self.encoder(x)
        out = self.classifier(z)
        return out


# загружаем окружение из .env
load_dotenv()

params = {
    'database': getenv('DB_NAME'),
    'user':     getenv('DB_USER'),
    'password': getenv('DB_PASSWORD'),
    'host':     getenv('DB_HOST'),
    'port':     getenv('DB_PORT'),
}

conn = psycopg2.connect(**params)
cursor = conn.cursor()

yid = ''  # youtube ID

print('starting download %s' % yid)

system('youtube-dl -cwi -o "%(id)s.%(ext)s" --extract-audio --ffmpeg-location ./ --audio-format wav --audio-quality 0 "https://www.youtube.com/watch?v=' + yid + '"')
inputFile = yid + '.wav'
print('download complete')

if path.isfile(inputFile) and Path(inputFile).stat().st_size < 1888888888:

    Path("./" + yid).mkdir(parents=True, exist_ok=True)
    [Fs, x] = aIO.read_audio_file(inputFile)
    print("start segmentation")
    try:
        segments = aS.silence_removal(
            x, Fs, 0.020, 0.020, smooth_window=0.3, weight=0.2, plot=False)
    except Exception as err:
        segments = []

    filenames = []
    embeds = []
    for i, s in enumerate(segments):
        if (s[1] - s[0] > 0.5):
            strOut = "d:\\{1:s}\\{0:d}.wav".format(i, yid)
            filenames.append(strOut)
            wavfile.write(strOut, Fs, x[int(Fs * s[0]):int(Fs * s[1])])

    print('complete split on %d segments' % len(filenames))

    results = Parallel(n_jobs=-1)(delayed(avec.audio2vec)(f)
                                  for f in filenames)

    for r in results:
        if ('file' in r):
            try:
                cursor.execute('''
                insert into voice_vectors (file, pyaudio, resemblyzer, wav2vec, rawnet, speech_score) 
                values (%s, %s, %s::double precision[], %s::double precision[], %s::double precision[], %s::double precision[], %s)
                ''',
                               [
                                   r['file'],
                                   np.array(r['pyAudio']).astype(
                                       float).tolist(),
                                   r['resemblyzer'],
                                   r['wav2vec'],
                                   r['rawnet'],
                                   r['score']
                               ])
                conn.commit()
            except Exception as err:
                print(err.pgcode)

    conn.commit()
    remove(inputFile)
