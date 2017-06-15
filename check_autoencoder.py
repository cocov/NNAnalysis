
from data_import.data_loader import DataLoader
from display_model.trace_binned import check_model_autoencoder

from keras.models import load_model
import numpy as np
np.random.seed(1337)
# Load the data
loader = DataLoader('/data/datasets/CTA/ToyNN/test_nsb.hdf5',labels = ['traces','label_1','label_2','weight_1','weight_2'])

loader.load(n_training=10000, n_testing=2000, train_start=0)

for k in loader.data.keys():
    for k1 in loader.data[k].keys():
        if k == 'test':
            loader.data[k][k1]=np.array(loader.data[k][k1].data)[10000:12000].reshape(2000,-1,1)
        else:
            loader.data[k][k1]=np.array(loader.data[k][k1].data)[0:10000].reshape(10000,-1,1)


model = load_model('model_autoencoder.h5')

check_model_autoencoder(data=loader.data['test'],model=model,n_display=100)


#model.compile(loss='custom_mean_squared_error_traces', optimizer=RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0), metrics=['accuracy'])
