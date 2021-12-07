from pyAudioAnalysis import audioSegmentation as aS
from pyAudioAnalysis import audioBasicIO as aIO
import scipy.io.wavfile as wavfile

from pathlib import Path

# Минимальный размер сегмента в секундах
MIN_SEGMENT_SIZE = 0.9

# Максимальный размер сегмента в секундах
MAX_SEGMENT_SIZE = 30


def audio_segmentation(inputFile, directory):
    Path(directory).mkdir(parents=True, exist_ok=True)
    [Fs, x] = aIO.read_audio_file(inputFile)

    # Сегментация аудио: обучаем SVM на 10% сегментов с наименьшей энергией и 10% с наибольшей
    try:
        segments = aS.silence_removal(
            x, Fs, 0.020, 0.020, smooth_window=0.3, weight=0.2, plot=False)
    except Exception as err:
        segments = []

    filenames = []

    for i, s in enumerate(segments):
        if (s[1] - s[0] > MAX_SEGMENT_SIZE):
            s[1] = s[0] + MAX_SEGMENT_SIZE

        if (s[1] - s[0] > MIN_SEGMENT_SIZE):
            strOut = "{1:s}\{0:d}.wav".format(i, directory)
            filenames.append(strOut)
            wavfile.write(strOut, Fs, x[int(Fs * s[0]):int(Fs * s[1])])

    return filenames
