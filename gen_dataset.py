from comet_ml import Experiment
import scipy.io.wavfile as wav
import numpy as np

Classes = ['Angry', 'disgust']
for c in range(1,3):
    data = []
    labels = []
    for i in range (1,3):
        for j in range(1,201):
            if (j== 48 and i ==2 and c == 1): continue
            data.append(wav.read('../Dataset_tess/C{2}/C{2}_{0}_{1}.wav'.format(i,j,c))[1])
            labels.append('{}_{}'.format(i,Classes[c-1]))

    x = [k.shape[0] for k in data]


    # data_matrix = np.zeros((len(x), int(np.mean(x)+np.std(x)) +1), dtype=object)
    data_matrix = np.zeros((len(x), 52233), dtype=object)
    print(data_matrix)

    for i,k in enumerate(data):
        print(i,len(k))
        size = len(k)
        if size > data_matrix.shape[1]-1 : size = data_matrix.shape[1]-1
        data_matrix[i][:size] = k[:size]
    for i,k in enumerate(labels):
        data_matrix[i][-1] = k
    print(data_matrix.shape)

    np.savetxt('Dataset/wav_data_c{}.txt'.format(c),data_matrix,delimiter=',', fmt="%s")