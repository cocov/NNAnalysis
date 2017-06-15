
from data_import.data_loader import DataLoader
from display_model.trace_binned import check_model_seq2seq

from keras.models import load_model
import numpy as np
np.random.seed(1337)
# Load the data
loader = DataLoader('/data/datasets/CTA/ToyNN/test_nsb.hdf5',labels = ['traces','label_1','label_2','weight_1','weight_2'])
loader.load(n_training=4000, n_testing=1000, train_start=0)

for k in loader.data.keys():
    for k1 in loader.data[k].keys():
        if k == 'test':
            loader.data[k][k1]=np.array(loader.data[k][k1].data)[4000:5000]
        else:
            loader.data[k][k1]=np.array(loader.data[k][k1].data)[0:4000]

model = load_model('model_seq2seq.h5')

check_model_seq2seq(data=loader.data['test'],model=model,n_display=100)


#model.compile(loss='custom_mean_squared_error_traces', optimizer=RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0), metrics=['accuracy'])
