import numpy as np
import pandas as pd

# вектор параметров из разных моделей
VECTOR = 'resemblyzer123,wav2vec41,wav2vec7,wav2vec14,wav2vec86,wav2vec13,wav2vec81,wav2vec91,wav2vec96,wav2vec112,resemblyzer51,wav2vec106,wav2vec67,wav2vec17,wav2vec22,wav2vec87,wav2vec94,resemblyzer143,wav2vec125,wav2vec8,wav2vec30,resemblyzer86,wav2vec36,resemblyzer28,wav2vec117,wav2vec21,wav2vec52,wav2vec120,wav2vec89,resemblyzer220,resemblyzer244,resemblyzer8,resemblyzer39,wav2vec35,wav2vec114,pyaudio110,wav2vec95,wav2vec0,resemblyzer167,wav2vec25,resemblyzer116,wav2vec38,wav2vec80,wav2vec16,wav2vec62,wav2vec15,wav2vec24,resemblyzer127,wav2vec82,wav2vec10,wav2vec61,wav2vec103,pyaudio122,pyaudio86,resemblyzer104,resemblyzer211,resemblyzer181,resemblyzer124,wav2vec107,wav2vec34,resemblyzer205,wav2vec113,wav2vec50,wav2vec5,wav2vec44,wav2vec42,resemblyzer253,wav2vec48,resemblyzer6,resemblyzer213,resemblyzer190,pyaudio120,wav2vec53,wav2vec104,pyaudio1,wav2vec73,resemblyzer56,resemblyzer221,resemblyzer188,wav2vec33,wav2vec115,resemblyzer247,wav2vec47,wav2vec63,resemblyzer85,resemblyzer242,wav2vec57,wav2vec11,wav2vec123,wav2vec118,wav2vec28,resemblyzer70,wav2vec43,wav2vec102,wav2vec70,rawnet379,wav2vec110,wav2vec108,wav2vec98,resemblyzer92,wav2vec32,resemblyzer201,resemblyzer18,resemblyzer43,resemblyzer87,wav2vec12,wav2vec18,resemblyzer2,pyaudio99,resemblyzer173,rawnet43,wav2vec46,resemblyzer63,resemblyzer135,resemblyzer11,wav2vec72,wav2vec1,wav2vec45,resemblyzer120,resemblyzer75,resemblyzer42,wav2vec69,wav2vec65,resemblyzer118,wav2vec116,wav2vec49,wav2vec122,pyaudio12,wav2vec71,resemblyzer94,rawnet211,resemblyzer170,wav2vec126,resemblyzer165,wav2vec4,wav2vec101,resemblyzer35,resemblyzer130,resemblyzer184,wav2vec58,wav2vec97,resemblyzer33,wav2vec40,pyaudio5,resemblyzer251,pyaudio7,wav2vec60,wav2vec100,pyaudio9,wav2vec79,wav2vec26,pyaudio126,wav2vec3,resemblyzer141,pyaudio118,wav2vec77,resemblyzer209,wav2vec56,resemblyzer128,wav2vec84,resemblyzer37,rawnet31,pyaudio123,resemblyzer183,wav2vec39,resemblyzer49,wav2vec27,wav2vec31,wav2vec88,wav2vec59,resemblyzer249,resemblyzer178'


def transform_columns(r):
    pyaudio = np.array(r['pyAudio']).astype(float)
    cols_pyaudio = ['pyaudio'+str(i) for i in range(0, r['pyAudio'].shape[0])]
    pd_pyaudio = pd.DataFrame(pyaudio.reshape(1, -1), columns=cols_pyaudio)

    resemblyzer = r['resemblyzer']
    cols_resemblyzer = ['resemblyzer'+str(i)
                        for i in range(0, resemblyzer.shape[0])]
    pd_resemblyzer = pd.DataFrame(
        resemblyzer.reshape(1, -1), columns=cols_resemblyzer)

    wav2vec = r['wav2vec'].ravel()
    cols_wav2vec = ['wav2vec'+str(i) for i in range(0, wav2vec.shape[0])]
    pd_wav2vec = pd.DataFrame(wav2vec.reshape(1, -1), columns=cols_wav2vec)

    rawnet = r['rawnet'].ravel()
    cols_rawnet = ['rawnet'+str(i) for i in range(0, rawnet.shape[0])]
    pd_rawnet = pd.DataFrame(rawnet.reshape(1, -1), columns=cols_rawnet)

    train = pd.concat([pd_pyaudio, pd_resemblyzer, pd_wav2vec,
                      pd_rawnet], axis=1, join="inner")
    columns = VECTOR
    train['vector'] = train[columns.split(",")].apply(
        lambda row: ','.join(row.values.astype(str)), axis=1)

    train = train.drop(columns.split(","), 1)
    train = train.drop(train.columns.difference(['vector']), axis=1)

    return train
