from keras.models import load_model
from data_import.data_loader import DataLoader
from display_model.trace_binned import check_model
from utils.custom_losses import custom_mean_squared_error_traces,custom_mean_squared_error_traces_relative

# Load the model
model = load_model('test_model.h5', custom_objects={'custom_mean_squared_error': custom_mean_squared_error_traces})
model.summary()

# Load the data
loader = DataLoader('/data/datasets/CTA/ToyNN/test_nsb.hdf5')
loader.load(n_training=10000, n_testing=0, train_start=0)

# Visualise the trace
check_model(data=loader.data['test'],model=model,n_display=100)