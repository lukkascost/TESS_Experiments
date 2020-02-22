# from comet_ml import Experiment
import librosa
import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt

Classes = ['Angry', 'disgust', 'fear', 'happy', 'surprise', 'sad', 'Neutral']
cortes = []
global_num = []
classes_qtd = np.zeros(7)

for c in range(1, 8):
    data = []
    labels = []
    for i in range(1, 3):
        for j in range(1, 201):
            if (j == 48 and i == 2 and c == 1): continue
            if (j == 112 and i == 2 and c == 3): continue
            sample_rate_1,X_1 =  wav.read('../Dataset_tess/C{2}/C{2}_{0}_{1}.wav'.format(i, j, c))
            X, sample_rate = librosa.load('../Dataset_tess/C{2}/C{2}_{0}_{1}.wav'.format(i, j, c),
                                          res_type='kaiser_fast', sr=None)
            data.append(X)
            labels.append('{}_{}'.format(i, Classes[c - 1]))

    x = [k.shape[0] for k in data]
    global_num = global_num + x
    x = np.array(x)
    x = (x - 50214) * 100 // x
    classes_qtd[c - 1] = len(x[np.logical_and(x >= -10, x <= 10)])
    cortes = cortes + list(x)
    # data_matrix = np.zeros((len(x), int(np.mean(x)+np.std(x)) +1), dtype=object)
    data_matrix = np.zeros((int(classes_qtd[c - 1]), 50214), dtype=object)
    index = 0
    for i, k in enumerate(data):
        size = len(k)
        if np.logical_and(x >= -10, x <= 10)[i]:
            if size > data_matrix.shape[1] - 1: size = data_matrix.shape[1] - 1
            data_matrix[index][:size] = k[:size]
            index += 1
    index = 0
    for i, k in enumerate(labels):
        if np.logical_and(x >= -10, x <= 10)[i]:
            data_matrix[index][-1] = k
            index += 1
    print(data_matrix.shape)

    np.savetxt('Dataset/GLOBAL_MEAN/wav_data_c{}.data'.format(c), data_matrix, delimiter=',', fmt="%s")
