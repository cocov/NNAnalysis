from keras.optimizers import RMSprop
import numpy as np
from pixel_seq_models.lstm_seq import LSTM_Seq
from data_import.data_loader import DataLoader
from display_model.trace_binned import check_model
from utils.custom_losses import custom_mean_squared_error_traces,custom_mean_squared_error_traces_relative
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers.core import Dense,  Dropout
from keras.layers.recurrent import  LSTM
from display_model.trace_binned import check_model_binned
from keras.layers import  Bidirectional,RepeatVector,TimeDistributed,Activation
from keras.utils import np_utils

from keras.preprocessing import sequence
# Load the data
loader = DataLoader('/data/datasets/CTA/ToyNN/test_nsb_binned_cat.hdf5',labels = ['traces','labels'])
loader.load(n_training=9000, n_testing=1000, train_start=0)


N_MAX_PHOTONS_IN_WINDOW = loader.data['train']['labels'].data.shape[1]



nb_classes = np.max(loader.data['test']['labels'].shape)
'''
for i in ['input','test']:
    for l in ['labels']:
        print(l)
        loader.data['train'][l] = np_utils.to_categorical(loader.data['train'][l][], nb_classes)
        loader.data['test'][l] =  np_utils.to_categorical(loader.data['test'][l], nb_classes)
       #     sequence.pad_sequences(loader.data['test'][l], maxlen=N_MAX_PHOTONS_IN_WINDOW, padding='post',truncating='post',value = 0)
'''
# concatenate
# Pad and truncate the sample
#loader.data['train']['input'].data=loader.data['train']['input'].reshape(-1,1)
#loader.data['test']['input'].data=loader.data['test']['input'].reshape(-1,1)
print(loader.data['test']['labels'].shape)

N_SAMPLES = loader.data['train']['traces'].shape[1]
INPUT_DIM = 1
HIDDEN_LAYER = 50

HIGHEST_ADC = 300
HIGHEST_REP = 6


model = Sequential()
# Conversion of ADCs to dense embedded representation

model.add(Embedding(HIGHEST_ADC, HIGHEST_REP , input_length= 76))
# First encoder LSTM
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM( units=HIDDEN_LAYER, return_sequences=False), input_shape=(None, 1)))
model.add(Dropout(0.2))
model.add(RepeatVector(N_MAX_PHOTONS_IN_WINDOW))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM( units=HIDDEN_LAYER, return_sequences=True)))
model.add(TimeDistributed(Dense(20)))
model.add(Activation('relu'))
#model.add(Dense(1, activation='linear'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True)

# Fit the model
model.fit(loader.data['train']['traces'], loader.data['train']['labels'],
          validation_data=[loader.data['test']['traces'], loader.data['test']['labels']], epochs=5, batch_size=10, verbose=2,shuffle='batch')

check_model_binned(data=loader.data['test'],model=model,n_display=100)


#model.compile(loss='custom_mean_squared_error_traces', optimizer=RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0), metrics=['accuracy'])
