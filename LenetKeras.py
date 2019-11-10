import numpy as np
from comet_ml import Experiment
experiment = Experiment(api_key="9F7edG4BHTWFJJetI2XctSUzM",
                        project_name="teste-lenet", workspace="lukkascost")
from keras import Sequential, optimizers, losses
from keras.layers import Conv1D, AveragePooling1D, Flatten, Dense, Reshape
from keras.utils import to_categorical
from sklearn.model_selection import KFold, train_test_split

data = np.loadtxt('wav_data_c1.txt', delimiter=',', dtype=object)
data2 = np.loadtxt('wav_data_c2.txt', delimiter=',', dtype=object)
data = np.vstack((data,data2))

atts = data[:, :-1]
labels = data[:, -1]

labels[labels == '1_Angry'] = 0
labels[labels == '2_Angry'] = 0
labels[labels == '1_disgust'] = 1
labels[labels == '2_disgust'] = 1

X_train, X_test, y_train, y_test = train_test_split(atts, labels, test_size=0.20, random_state=42)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

n_timesteps, n_features, n_outputs = atts.shape[1], 1, 2
EPOCHS = 100
BATCH_SIZE = 8

model = Sequential()
model.add(Reshape((n_timesteps, n_features), input_shape=(atts.shape[1],)))
model.add(Conv1D(filters=6, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
model.add(AveragePooling1D())
model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
model.add(AveragePooling1D())
model.add(Flatten())
model.add(Dense(units=120, activation='relu'))
model.add(Dense(units=84, activation='relu'))
model.add(Dense(units=2, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(), metrics=['accuracy'])

model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=True)
_, accuracy = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=0)
print(accuracy)
experiment.log_other('model', model)
experiment.end()