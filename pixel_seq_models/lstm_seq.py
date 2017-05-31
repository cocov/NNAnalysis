from keras.models import Sequential
from keras.layers.core import Dense,  Dropout
from keras.layers.recurrent import  LSTM
from keras.layers import  Bidirectional
from keras.models import load_model

class LSTM_Seq(Sequential) :
    def __init__(self,optimizer,loss,input_size):
        Sequential.__init__(self)
        self.add(Bidirectional(LSTM(50, return_sequences=True), name='bidir_0',batch_input_shape=(None,input_size,1)))
        self.add(Dropout(0.2))
        self.add(Dense(10, activation='relu'))
        self.add(Dense(1, activation='linear'))
        self.compile(optimizer=optimizer,loss=loss)

    def train(self, data, **kwargs):
        """
        Wrapper around the fit function
        :param data: data dictionnary
        :param kwargs:
        :return:
        """
        self.fit(data['train']['input'], data['train']['label'],
                 validation_data=[data['test']['input'], data['test']['label']], **kwargs)

    def save_model(self,name_tag, folder = '.'):
        """
        Wrapper around the fit function
        :param data:
        :param kwargs:
        :return:
        """
        super(Sequential, self).save('%s/%s_model.h5'%(folder,name_tag), overwrite=True)
        super(Sequential, self).save_weights('%s/%s_weights.h5'%(folder,name_tag), overwrite=True)

