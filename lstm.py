from keras.optimizers import RMSprop
import numpy as np
from pixel_seq_models.lstm_seq import LSTM_Seq
from data_import.data_loader import DataLoader
from display_model.trace_binned import check_model
from utils.custom_losses import custom_mean_squared_error_traces_relative_10,custom_mean_squared_error_traces
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
loader = DataLoader('/data/datasets/CTA/ToyNN/test_nsb_simple_4_40-80.hdf5',labels = ['traces','labels'])
loader.load(n_training=90000, n_testing=10000, train_start=0)

for k in loader.data.keys():
    for k1 in loader.data[k].keys():
        if k == 'test':
            loader.data[k][k1]=np.array(loader.data[k][k1].data)[90000:100000]
        else:
            loader.data[k][k1]=np.array(loader.data[k][k1].data)[0:90000]
        if k1 == 'labels':
            loader.data[k][k1]=loader.data[k][k1].reshape(loader.data[k][k1].shape+(1,))

N_OUPUT_BINS = loader.data['train']['labels'].data.shape[1]
N_INPUT_BINS = loader.data['train']['traces'].data.shape[1]
weights = np.copy(loader.data['train']['labels']).astype(float)
weights[weights>1.]=1.
weights=weights.reshape(weights.shape[0],weights.shape[1])
print(loader.data['train']['traces'].shape)
print(loader.data['test']['traces'].shape)

print(loader.data['train']['labels'].shape)
print(loader.data['test']['labels'].shape)

INPUT_DIM = 1
HIDDEN_LAYER = 100
HIGHEST_ADC = 300
HIGHEST_REP = 8

# Model definition
trace = Input(shape=(N_INPUT_BINS,), dtype='int32', name='input_trace')
x = Embedding(input_dim=HIGHEST_ADC, output_dim=HIGHEST_REP , input_length= N_INPUT_BINS)(trace)
x = Bidirectional(LSTM( units=HIDDEN_LAYER, return_sequences=True), input_shape=(None, 1))(x)
x = Dropout(0.2)(x)
#x = RepeatVector(N_OUPUT_BINS)(x)
#x = Dropout(0.2)(x)
#x = LSTM( units=HIDDEN_LAYER, return_sequences=True)(x)
x = Dense(10,activation='relu')(x)
x = Dense(1,activation='linear')(x)

model = Model(inputs=[trace], outputs=[x]) #, timing

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_noseq.png', show_shapes=True)
model.compile(loss=custom_mean_squared_error_traces,
              optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
              #,sample_weight_mode='temporal'
              )

model.summary()


early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=3, verbose=2, mode='min')
history = model.fit( loader.data['train']['traces'], loader.data['train']['labels'],
          validation_data=[loader.data['test']['traces'], loader.data['test']['labels']],
          #sample_weight=weights,
          epochs=50, batch_size=100, shuffle='batch',verbose =2, callbacks=[early_stop])

model.save('model_lstm.h5')

import matplotlib.pyplot as plt
plt.ion()
plt.subplots(2,2)
plt.subplot(2,2,1)
for l in ['loss','val_loss']:
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

