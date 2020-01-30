import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input
from keras.losses import mean_squared_logarithmic_error

X_train = np.load('../data/X_train.npy', allow_pickle=True)
y_train_e = np.load('../data/y_train_e.npy', allow_pickle=True)
y_train_coord = np.load('../data/y_train_coord.npy', allow_pickle=True)
y_train = [y_train_e, y_train_coord]

model_in= Input(shape=(X_train.shape[1], X_train.shape[2]))

RNN = LSTM(400)(model_in)
RNN = LSTM(400)(RNN)
RNN = LSTM(400)(RNN)
end_strok_out = Dense(1,  activation='softmax')(RNN)
coord_out = Dense(2,  activation='linear')(RNN)


model = Model(inputs=model_in, outputs=[end_strok_out, coord_out])
model.compile(optimizer = 'adam', loss = ['binary_crossentropy', 'mean_squared_error'])

model.summary()

model.fit(X_train, y_train, batch_size=128, epochs=20, validation_split=0.1, verbose=1)
model.save('../models/unconditional_generation_model_double_output_hydra.h5')