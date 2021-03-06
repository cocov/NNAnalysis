import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import RepeatVector

X_train = np.array([[1,2], [6,5], [8,2]])
y_train = np.array([2,3,7])
print(y_train.shape)
input_dim = X_train.shape[1]

model = Sequential()

model.add(Dense(output_dim=64, input_dim=input_dim))
model.add(Activation("relu"))
model.add(Dense(output_dim=10))
model.add(Activation("softmax"))
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.summary()
print(y_train)
model.fit(X_train, y_train, nb_epoch=5, batch_size=32)