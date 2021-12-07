from resemblyzer import VoiceEncoder, preprocess_wav
from pyAudioAnalysis import audioTrainTest as aT

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import wavencoder

from os import remove

torchaudio.set_audio_backend("soundfile")


# Пороговое значение для определения речь/музыка
THRESHHOLD_VOICE = 0.7
DEVICE = 'cuda'

# %%
"""
### Инициализация моделей
"""

# %%

# wav2vec
model1 = nn.Sequential(
    wavencoder.models.Wav2Vec(pretrained=True, device=torch.device(DEVICE)),
    wavencoder.models.LSTM_Attn_Classifier(
        512, 128, 64, return_attn_weights=True, attn_type='soft')
)

# rawnet
rawnet_encoder = wavencoder.models.RawNet2Model(
    pretrained=True, device=DEVICE, return_code=True, class_dim=100)

# resemblyzer
encoder2 = VoiceEncoder()


# %%
"""
### Преобразование аудиофайла в вектор
"""

# %%


def audio2vec(file):
    # Модель для определения речь/музыка
    z = aT.file_classification(file,
                               "./models/svm_rbf_sm",
                               "svm")
    if (z[1][0] < THRESHHOLD_VOICE):
        remove(file)
        return {}

    wav = preprocess_wav(file)

    # resemblyzer
    embeds_b = np.array(encoder2.embed_utterance(wav))

    x, sample_rate = torchaudio.load(file)

    # wav2vec
    vector1, attn_weights = model1(x)

    # rawnet
    vector2 = rawnet_encoder(x.resize_(1, 16000))

    return {'file':        file,
            'score':       z[1][0],
            'pyAudio':     z[3],
            'resemblyzer': embeds_b.astype(float),
            'wav2vec':     vector1.detach().numpy().astype(float),
            'rawnet':      vector2.detach().numpy().astype(float),
            }
