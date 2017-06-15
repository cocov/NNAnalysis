import matplotlib.pyplot as plt
import numpy as np


def check_model(data,model,n_display=100):
    plt.ion()
    plt.figure()
    for i in range(n_display):
        x = data['traces'][i]
        y = data['labels'][i]
        pred = [y]
        if model:
            pred = model.predict(x.reshape((1,) + x.shape))
        y=y.reshape(y.shape[0])
        x=x.reshape(x.shape[0])
        xbin = 4
        ybin = xbin*x.shape[0]/y.shape[0]
        pred=pred.reshape(pred.shape[1])
        plt.cla()
        plt.clf()
        plt.step(np.arange(0, (x.shape[0])*xbin, xbin), x,label='Input trace',where='mid')
        plt.step(np.arange(0, (y.shape[0])*ybin, ybin), y,label='True trace',where='mid')
        plt.step(np.arange(0, (pred.shape[0])*ybin, ybin), pred,label='Predicted trace',where='mid')
        plt.legend()
        plt.show()
        fk = input('bla')




def check_model_binned(data,model,n_display=100):
    plt.ion()
    plt.figure()
    for i in range(n_display):
        x = data['traces'][i]
        y = data['labels'][i]
        orig_xshape = x.shape
        orig_yshape = y.shape
        x = x.reshape((1,) + x.shape)
        y = y.reshape((1,) + y.shape)
        pred = model.predict(x)
        y = y.reshape(orig_yshape)
        x = x.reshape(orig_xshape)
        pred = pred.reshape(orig_yshape)
        print(pred)
        plt.cla()
        plt.clf()
        plt.step(np.arange(0, 76*4, 4), x,label='Input trace')
        plt.step(np.arange(0, 608*0.5, 0.5), np.argmax(y,axis=-1),label='True Photons')
        plt.step(np.arange(0, 608*0.5, 0.5), np.argmax(pred,axis=-1),label='Predicted Photons')
        plt.legend()
        plt.show()
        fk = input('bla')

def check_model_seq2seq(data,model,n_display=100):
    plt.ion()
    for i in range(n_display):
        x = data['traces'][i]
        y_yield = data['label_1'][i]
        y_time = data['label_2'][i]
        pred = [y_yield,y_time]
        if model:
            pred = model.predict(x.reshape((1,) + x.shape))
        plt.cla()
        plt.clf()
        print(x.shape)
        plt.step(np.arange(0, (x.shape[0])*4, 4), x,label='Input trace',where='mid')
        true_axis,true_value = np.argmax(y_time,axis=-1)*4,np.argmax(y_yield,axis=-1)
        true_axis=true_axis[true_axis>0]-1
        true_value= true_value[true_value>0]
        pred_axis,pred_value = np.argmax(pred[1],axis=-1)*4,np.argmax(pred[0],axis=-1)
        pred_axis=pred_axis[pred_axis>0]-1
        pred_value= pred_value[pred_value>0]

        print ('True',true_axis,true_value)
        print ('Predicted',pred_axis,pred_value)
        plt.plot(true_axis, true_value,label='True Photons',linestyle='None', marker='o')
        plt.plot(pred_axis, pred_value,label='Predicted Photons',linestyle='None', marker='*')
        plt.legend()
        plt.show()
        fk = input('bla')


def check_model_autoencoder(data,model,n_display=100):
    plt.ion()
    for i in range(n_display):
        x = data['traces'][i]
        y = data['traces'][i]
        pred = [y]
        if model:
            pred = model.predict(x.reshape((1,) + x.shape))
        pred=pred.reshape(x.shape[0])
        x=x.reshape(x.shape[0])
        plt.cla()
        plt.clf()
        plt.step(np.arange(0, (x.shape[0])*4, 4), x,label='Input trace',where='mid')
        plt.step(np.arange(0, (x.shape[0])*4, 4), pred,label='Predicted trace',where='mid')
        plt.legend()
        plt.show()
        fk = input('bla')

