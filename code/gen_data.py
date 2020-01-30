from sklearn.model_selection import train_test_split
from numpy import save

seq_len = 40

X_raw = []
y_raw = []

for stroke in strokes:
    i = 0
    while i+seq_len < len(stroke):
        X_raw.append(stroke[i:i+seq_len])
        y_raw.append(stroke[i+seq_len])
        i += 1

X_raw = np.array(X_raw)
y_raw = np.array(y_raw)

X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, random_state=42)

save('../data/X_train.npy', X_train)
save('../data/X_test.npy', X_test)
save('../data/y_train.npy', y_train)
save('../data/y_test.npy', y_test)

print("training set shape : ", X_train.shape)
print("training goals shape : ", y_train.shape)
print("testing set shape : ", X_test.shape)
print("testing goals shape : ", y_test.shape)

import numpy as np
from numpy import load, save

y_train = load('../data/y_train.npy', allow_pickle=True)
y_test = load('../data/y_test.npy', allow_pickle=True)

y_train_e = []
y_train_coord = []

for y in y_train:
    y_train_e.append(y[0])
    y_train_coord.append(y[1:])
    
y_train_e = np.array(y_train_e)
y_train_coord = np.array(y_train_coord)
save('../data/y_train_e.npy', y_train_e)
save('../data/y_train_coord.npy', y_train_coord)



y_train_reshaped = [y_train_e, y_train_coord]

print(y_train_reshaped[0].shape, y_train_reshaped[1].shape)
    
y_test_e = []
y_test_coord = []

for y in y_test:
    y_test_e.append(y[0])
    y_test_coord.append(y[1:])
    
y_test_e = np.array(y_test_e)
y_test_coord = np.array(y_test_coord)
save('../data/y_test_e.npy', y_test_e)
save('../data/y_test_coord.npy', y_test_coord)

y_test_reshaped = [y_test_e, y_test_coord]

print(y_test_reshaped[0].shape, y_test_reshaped[1].shape)