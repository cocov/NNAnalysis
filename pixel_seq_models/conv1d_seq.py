from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

class Conv1DModel(Sequential) :
    def __init__(self,optimizer,loss,metric):

        Sequential.__init()
        self.add(Conv1D(64, 3, activation='relu', input_shape=(seq_length, 100)))
        self.add(Conv1D(64, 3, activation='relu'))
        self.add(MaxPooling1D(3))
        self.add(Conv1D(128, 3, activation='relu'))
        self.add(Conv1D(128, 3, activation='relu'))
        self.add(GlobalAveragePooling1D())
        self.add(Dropout(0.5))
        self.add(Dense(1, activation='sigmoid'))
        self.compile(optimizer=optimizer,loss=loss)

        self.compile(loss='binary_crossentropy',
                        optimizer='rmsprop',
                          metrics=['accuracy'])


