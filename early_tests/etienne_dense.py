from keras.optimizers import RMSprop
from keras.optimizers import SGD

from pixel_seq_models.lstm_seq import LSTM_Seq
from data_import.data_loader import DataLoader
from display_model.trace_binned import check_model
from utils.custom_losses import custom_mean_squared_error_traces,custom_mean_squared_error_traces_relative

from keras.models import Sequential
from keras.layers.core import Dense,  Dropout
from keras.layers.recurrent import  LSTM
from keras.layers import  Bidirectional, Convolution1D, Conv1D, Flatten, MaxPooling1D
from keras.models import load_model
from keras.callbacks import EarlyStopping

import numpy as np
import matplotlib.pyplot as plt

# Load the data
loader = DataLoader('/data/datasets/CTA/ToyNN/test_nsb.hdf5')
loader.load(n_training=9000, n_testing=1000, train_start=0)

#dimensions reduction to feed to Dense layers directly
print ( loader.data['train']['input'].shape[0])
#question for Victor: why the hell do the inputs have one extra dimension ? To notify that there is only one channel ?

train_input = np.squeeze(loader.data['train']['input'])
train_label = np.squeeze(loader.data['train']['label'])
test_input  = np.squeeze(loader.data['test']['input'])
test_label  = np.squeeze(loader.data['test']['label'])

test = {'input': None, 'label': None}
test['input'] = test_input
test['label'] = test_label

model = Sequential();

model.add(Dense(304, activation='sigmoid', input_shape=(76,)))
model.add(Dense(152, activation='sigmoid'))
model.add(Dense(76, activation='relu'))

optimizer = SGD(lr=0.5, momentum=0.0, decay=0.0, nesterov=False)
model.compile(optimizer=optimizer,loss='mean_squared_error')
model.summary()

early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=50, verbose=2, mode='min')
history = model.fit(train_input, train_label,
          validation_data=[test_input, test_label], 
          nb_epoch=1000, batch_size=1000, verbose=2,shuffle='batch', callbacks=[early_stop])

print(history.history.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# Fit the model
#model.train(loader.data, nb_epoch=5, batch_size=10, verbose=2,shuffle='batch')
# Save the model
#model.save_model(name_tag='test')

# Visualise the trace
check_model(data=test,model=model,n_display=100)