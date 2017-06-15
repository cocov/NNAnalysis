
from data_import.data_loader import DataLoader
from display_model.trace_binned import check_model

from keras.models import load_model
import numpy as np
from utils.custom_losses import custom_mean_squared_error_traces_relative_10,custom_mean_squared_error_traces_relative,custom_mean_squared_error_traces

np.random.seed(1337)
# Load the data
loader = DataLoader('/data/datasets/CTA/ToyNN/test_nsb_simple_4.hdf5',labels = ['traces','labels'])
loader.load(n_training=8000, n_testing=2000, train_start=0)
for k in loader.data.keys():
    for k1 in loader.data[k].keys():
        if k == 'test':
            loader.data[k][k1]=np.array(loader.data[k][k1].data)[8000:10000]
        else:
            loader.data[k][k1]=np.array(loader.data[k][k1].data)[0:8000]
        if k1 == 'labels':
            loader.data[k][k1]=loader.data[k][k1].reshape(loader.data[k][k1].shape+(1,))

model = load_model('model_lstm.h5')#,custom_objects={'custom_mean_squared_error_traces_relative_10': custom_mean_squared_error_traces_relative_10})

check_model(data=loader.data['test'],model=model,n_display=100)


#model.compile(loss='custom_mean_squared_error_traces', optimizer=RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0), metrics=['accuracy'])
