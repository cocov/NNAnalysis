from keras.optimizers import RMSprop

from pixel_seq_models.lstm_seq import LSTM_Seq
from data_import.data_loader import DataLoader
from display_model.trace_binned import check_model
from utils.custom_losses import custom_mean_squared_error_traces,custom_mean_squared_error_traces_relative


# Load the data
loader = DataLoader('/data/datasets/CTA/ToyNN/test_nsb.hdf5')
loader.load(n_training=9000, n_testing=1000, train_start=0)


# Load the model and compile it
model = LSTM_Seq(optimizer = RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0),
                 loss = lambda x , y : custom_mean_squared_error_traces(x,y,margin=10) , input_size=loader.data['train']['input'].shape[1])

# Fit the model
model.train(loader.data, epochs=50, batch_size=100, verbose=2,shuffle='batch')
# Save the model
model.save_model(name_tag='test')

# Visualise the trace
check_model(data=loader.data['test'],model=model,n_display=100)