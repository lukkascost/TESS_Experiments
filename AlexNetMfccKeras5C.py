import numpy as np
from comet_ml import Experiment
import librosa
from keras import Sequential, optimizers, losses
from keras.layers import Conv1D, AveragePooling1D, Flatten, Dense, Reshape, Activation, BatchNormalization, Dropout, \
    MaxPooling1D
from keras.utils import to_categorical, np_utils
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from keras import backend as K

data = np.loadtxt('Dataset/ORIGINAL/wav_data_c1.data', delimiter=',', dtype=object)
print(data[0])
print(data[:,:-1].shape,np.zeros([data.shape[0], 109580-data.shape[1]-1]).shape, np.matrix(data[:,-1]).T.shape )
data = np.hstack([data[:,:-1], np.zeros([data.shape[0], 109580-data.shape[1]]), np.matrix(data[:,-1]).T])
print(data[0])

data4 = np.loadtxt('Dataset/ORIGINAL/wav_data_c4.data', delimiter=',', dtype=object)
data4 = np.hstack([data4[:,:-1], np.zeros([data4.shape[0], 109580-data4.shape[1]]), np.matrix(data4[:,-1]).T])

data5 = np.loadtxt('Dataset/ORIGINAL/wav_data_c5.data', delimiter=',', dtype=object)
data5 = np.hstack([data5[:,:-1], np.zeros([data5.shape[0], 109580-data5.shape[1]]), np.matrix(data5[:,-1]).T])

data6 = np.loadtxt('Dataset/ORIGINAL/wav_data_c6.data', delimiter=',', dtype=object)

data7 = np.loadtxt('Dataset/ORIGINAL/wav_data_c7.data', delimiter=',', dtype=object)
data7 = np.hstack([data7[:,:-1], np.zeros([data7.shape[0], 109580-data7.shape[1]]), np.matrix(data7[:,-1]).T])

data = np.vstack((data, data4, data5, data6, data7))

sample_rate = 22050*2
atts = data[:, :-1]
labels = data[:, -1]
atts = np.array([np.mean(librosa.feature.mfcc(y=np.float32(np.array(x)[0]), sr=sample_rate, n_mfcc=13), axis=0) for x in list(atts)])

labels[labels == '1_Angry'] = 0
labels[labels == '2_Angry'] = 0
labels[labels == '1_happy'] = 1
labels[labels == '2_happy'] = 1
labels[labels == '1_surprise'] = 2
labels[labels == '2_surprise'] = 2
labels[labels == '1_sad'] = 3
labels[labels == '2_sad'] = 3
labels[labels == '1_Neutral'] = 4
labels[labels == '2_Neutral'] = 4

for it in range(1, 11):
    experiment = Experiment(api_key="9F7edG4BHTWFJJetI2XctSUzM",
                            project_name="alex-net-mfcc-tess-dataset-5c-without-cut", workspace="lukkascost")
    experiment.set_name("TESS_05_EP_{:02d}".format(it))
    strate = np.zeros(labels.shape)
    strate[int((strate.shape[0] / 5)) * 1:] += 1
    strate[int((strate.shape[0] / 5)) * 2:] += 1
    strate[int((strate.shape[0] / 5)) * 3:] += 1
    strate[int((strate.shape[0] / 5)) * 4:] += 1
    X_train, X_test, y_train, y_test = train_test_split(atts, labels, train_size=560, test_size=145, random_state=42,
                                                        stratify=strate)
    print(np.unique(y_test, return_counts=True))
    print(np.unique(y_train, return_counts=True))
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    print('AFTER: ', y_test)
    n_timesteps, n_features, n_outputs = atts.shape[1], 1, 5
    EPOCHS = 50
    BATCH_SIZE = 8
    print(n_timesteps, n_features)
    model = Sequential()
    model.add(Reshape((n_timesteps, n_features), input_shape=(atts.shape[1],)))
    model.add(Conv1D(256, 8, padding='same', input_shape=(X_train.shape[1], 1)))  # X_train.shape[1] = No. of Columns
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=(2)))

    model.add(Conv1D(256, 8, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling1D(pool_size=(2)))

    model.add(Conv1D(128, 8, padding='same'))
    model.add(Activation('relu'))

    model.add(Conv1D(128, 8, padding='same'))
    model.add(Activation('relu'))

    model.add(Conv1D(128, 8, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # model.add(Conv1D(128, 8, padding='same'))
    # model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling1D(pool_size=(2)))
    # model.add(Conv1D(64, 8, padding='same'))
    # model.add(Activation('relu'))
    # model.add(Conv1D(64, 8, padding='same'))
    # model.add(Activation('relu'))
    model.add(Flatten())

    # model.add(Dense(4096))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.4))

    # model.add(Dense(4096))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.4))

    # model.add(Dense(1000))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.4))

    model.add(Dense(n_outputs))  # Target class number
    model.add(Activation('softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(), metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=True)
    _, accuracy = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=0)
    print(accuracy)
    experiment.log_metric("test_accuracy", accuracy)
    model.save('model.h5')
    experiment.log_asset("model.h5")
    model.save_weights('model.weights')
    experiment.log_asset("model.weights")
    experiment.log_other('model', model)
    experiment.end()