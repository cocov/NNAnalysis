
from keras.utils.io_utils import HDF5Matrix

class DataLoader:
    """
    Class to handle HDF5 data for testing and training purpose
    """
    def __init__(self,datapath,labels = ['input','label'],):

        self.data_file_name  = datapath
        self.n_training      = -1
        self.n_testing       = -1
        self.train_start     = 0
        self.test_start      = -1
        self.data = {'train':{},'test': {}}
        self.labels = labels
        for l in labels:
            self.data['train'][l] = None
            self.data['test' ][l] = None

    def load(self,n_training, n_testing, train_start = 0 , test_start = -1 ):
        """
        load the data in self.data dictionnary

        :param n_training:  number of training samples
        :param n_testing:   number of testing samples
        :param train_start: starting training sample  (default 0)
        :param test_start:  starting testing sample   (default -1 which implies it will
                                                       start where the training sample stop)
        :return:
        """
        self.n_training   = n_training
        self.n_testing    = n_testing
        self.train_start  = train_start
        self.test_start   = test_start if test_start > -0.5 else self.train_start + self.n_training
        for l in self.labels:
            self.data['train'][l] = \
                HDF5Matrix(self.data_file_name, l, self.train_start, self.train_start + self.n_training)
            self.data['test'][l] = \
                HDF5Matrix(self.data_file_name, l, self.test_start, self.test_start + self.n_testing)
