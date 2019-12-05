# from comet_ml import Experiment
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
            data.append(wav.read('../Dataset_tess/C{2}/C{2}_{0}_{1}.wav'.format(i, j, c))[1])
            labels.append('{}_{}'.format(i, Classes[c - 1]))

    x = [k.shape[0] for k in data]
    global_num = global_num + x
    x = np.array(x)
    x = (x - 50214) * 100 // x
    classes_qtd[c - 1] = len(x[np.logical_and(x >= -10, x <= 10)])
    cortes = cortes + list(x)
    # data_matrix = np.zeros((len(x), int(np.mean(x)+np.std(x)) +1), dtype=object)
    # data_matrix = np.zeros((len(x), 47166), dtype=object)

    # for i, k in enumerate(data):
    #     # print(i, len(k))
    #     size = len(k)
    #     if size > data_matrix.shape[1] - 1: size = data_matrix.shape[1] - 1
    #     data_matrix[i][:size] = k[:size]
    # for i, k in enumerate(labels):
    #     data_matrix[i][-1] = k
    # print(data_matrix.shape)
    #
    # np.savetxt('Dataset/wav_data_c{}.txt'.format(c), data_matrix, delimiter=',', fmt="%s")
X, Y = np.unique(cortes, return_counts=True)
print np.mean(cortes), np.std(cortes)
print np.mean(global_num), np.std(global_num)
cortes = np.array(cortes)
print len(cortes[np.logical_and(cortes >= -20, cortes <= 20)])
print len(cortes)
print classes_qtd
plt.bar(X, Y)
plt.show()
