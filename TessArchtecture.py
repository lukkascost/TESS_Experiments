import numpy as np
from comet_ml import Experiment
experiment = Experiment(api_key="9F7edG4BHTWFJJetI2XctSUzM",
                        project_name="teste-lenet", workspace="lukkascost")
experiment.set_name("TESS_07_EP_02")
from keras import Sequential, optimizers, losses
from keras.layers import Conv1D, AveragePooling1D, Flatten, Dense, Reshape, Activation, BatchNormalization, Dropout, \
    MaxPooling1D
from keras.utils import to_categorical, np_utils
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder

data = np.loadtxt('wav_data_c1.txt', delimiter=',', dtype=object)
data2 = np.loadtxt('wav_data_c2.txt', delimiter=',', dtype=object)
data3 = np.loadtxt('wav_data_c3.txt', delimiter=',', dtype=object)
data4 = np.loadtxt('wav_data_c4.txt', delimiter=',', dtype=object)
data5 = np.loadtxt('wav_data_c5.txt', delimiter=',', dtype=object)
data6 = np.loadtxt('wav_data_c6.txt', delimiter=',', dtype=object)
data7 = np.loadtxt('wav_data_c7.txt', delimiter=',', dtype=object)
data = np.vstack((data,data2, data3, data4, data5, data6, data7))

atts = data[:, :-1]
labels = data[:, -1]

labels[labels == '1_Angry'] = 0
labels[labels == '2_Angry'] = 0
labels[labels == '1_disgust'] = 1
labels[labels == '2_disgust'] = 1
labels[labels == '1_fear'] = 2
labels[labels == '2_fear'] = 2
labels[labels == '1_happy'] = 3
labels[labels == '2_happy'] = 3
labels[labels == '1_surprise'] = 4
labels[labels == '2_surprise'] = 4
labels[labels == '1_sad'] = 5
labels[labels == '2_sad'] = 5
labels[labels == '1_Neutral'] = 6
labels[labels == '2_Neutral'] = 6

X_train, X_test, y_train, y_test = train_test_split(atts, labels, test_size=0.20, random_state=42)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print('AFTER: ', y_test)
n_timesteps, n_features, n_outputs = atts.shape[1], 1, 7
EPOCHS = 50
BATCH_SIZE = 8

model = Sequential()
model.add(Reshape((n_timesteps, n_features), input_shape=(atts.shape[1],)))
model.add(Conv1D(256, 8, padding='same',input_shape=(X_train.shape[1],1)))  # X_train.shape[1] = No. of Columns
model.add(Activation('relu'))
model.add(Conv1D(256, 8, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(64, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(64, 8, padding='same'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(n_outputs)) # Target class number
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(), metrics=['accuracy'])

model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=True)
_, accuracy = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=0)
print(accuracy)
model.save('model.h5')
experiment.log_asset("model.h5")
model.save_weights('model.weights')
experiment.log_asset("model.weights")
experiment.log_other('model', model)
experiment.end()