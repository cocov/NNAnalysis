from keras.optimizers import RMSprop
import numpy as np
from pixel_seq_models.lstm_seq import LSTM_Seq
from data_import.data_loader import DataLoader
from display_model.trace_binned import check_model
from utils.custom_losses import custom_mean_squared_error_traces,custom_mean_squared_error_traces_relative
from keras.models import Sequential,Model
from keras.layers import Embedding,Input,Reshape,Flatten,Permute
from keras.layers.core import Dense,  Dropout
from keras.layers.recurrent import  LSTM
from display_model.trace_binned import check_model_binned
from keras.layers import  Bidirectional,RepeatVector,TimeDistributed,Activation
from keras.utils import np_utils
from keras import metrics
from keras.callbacks import EarlyStopping

from keras.preprocessing import sequence
import numpy as np
np.random.seed(1337)
# Load the data
loader = DataLoader('/data/datasets/CTA/ToyNN/test_nsb.hdf5',labels = ['traces','label_1','label_2','weight_1','weight_2'])
loader.load(n_training=80000, n_testing=20000, train_start=0)

for k in loader.data.keys():
    for k1 in loader.data[k].keys():
        if k == 'test':
            loader.data[k][k1]=np.array(loader.data[k][k1].data)[80000:100000]
        else:
            loader.data[k][k1]=np.array(loader.data[k][k1].data)[0:80000]

N_MAX_PHOTONS_IN_WINDOW = loader.data['train']['label_1'].data.shape[1]
N_MAX_PHOTONS_PER_BIN = loader.data['train']['label_1'].data.shape[2]
N_MAX_TIMING = loader.data['train']['label_2'].data.shape[2]
N_INPUT_BINS = loader.data['train']['traces'].data.shape[1]

INPUT_DIM = 1
HIDDEN_LAYER = 100
HIGHEST_ADC = 300
HIGHEST_REP = 8

# Model definition
trace = Input(shape=(N_INPUT_BINS,), dtype='int32', name='input_trace')
x = Embedding(input_dim=HIGHEST_ADC, output_dim=HIGHEST_REP , input_length= N_INPUT_BINS)(trace)
x = Bidirectional(LSTM( units=HIDDEN_LAYER, return_sequences=False), input_shape=(None, 1))(x)
x = Dropout(0.2)(x)
x = RepeatVector(N_MAX_PHOTONS_IN_WINDOW)(x)
x = Dropout(0.2)(x)
x = LSTM( units=HIDDEN_LAYER, return_sequences=True)(x)
# now split to two output
yields = TimeDistributed(Dense(N_MAX_PHOTONS_PER_BIN,activation='softmax'))(x)
timing = TimeDistributed(Dense(N_MAX_TIMING,activation='softmax'))(x)
model = Model(inputs=[trace], outputs=[yields, timing]) #, timing

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_usingseq.png', show_shapes=True)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              sample_weight_mode='temporal',loss_weights=[1.,53./10.],metrics=[metrics.categorical_accuracy])

model.summary()


early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=2, mode='min')
print(loader.data['train']['traces'].shape,loader.data['test']['traces'].shape)
history = model.fit(loader.data['train']['traces'], [loader.data['train']['label_1'],loader.data['train']['label_2']],
          validation_data=[loader.data['test']['traces'], [loader.data['test']['label_1'],loader.data['test']['label_2']]],
          sample_weight=[loader.data['train']['weight_1'],loader.data['train']['weight_2']],
          epochs=100, batch_size=32, shuffle='batch',verbose =2, callbacks=[early_stop])

model.save('model_seq2seq.h5')

import matplotlib.pyplot as plt
plt.ion()
plt.subplots(2,2)
plt.subplot(2,2,1)
for l in ['loss','time_distributed_1_loss','time_distributed_2_loss']:
    plt.plot(history.history[l], label=l)
plt.legend()
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.subplot(2,2,2)
for l in ['val_loss','val_time_distributed_1_loss','val_time_distributed_2_loss']:
    plt.plot(history.history[l], label=l)
plt.legend()
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.subplot(2,2,3)
for l in ['time_distributed_1_categorical_accuracy','time_distributed_2_categorical_accuracy']:
    plt.plot(history.history[l], label=l)
plt.legend()
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.subplot(2,2,4)
for l in ['val_time_distributed_1_categorical_accuracy','val_time_distributed_2_categorical_accuracy']:
    plt.plot(history.history[l], label=l)




plt.legend()
plt.ylabel('Accuracy')
plt.xlabel('Epoch')





for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.legend()
    plt.show()

