import matplotlib.pyplot as plt
import numpy as np


def check_model(data,model,n_display=100):
    plt.ion()
    plt.figure()
    for i in range(n_display):
        x = data['input'][i]
        y = data['label'][i]
        x = x.reshape((1,) + x.shape)
        y = y.reshape((1,) + y.shape)
        pred = model.predict(x)
        y = y.reshape((76,))
        x = x.reshape((76,))
        pred = pred.reshape((76,))
        plt.cla()
        plt.clf()
        plt.step(np.arange(-150, 150 + 4, 4), x,label='Input trace')
        plt.step(np.arange(-150, 150 + 4, 4), y,label='True Photons')
        plt.step(np.arange(-150, 150 + 4, 4), pred,label='Predicted Photons')
        plt.legend()
        plt.show()
        fk = input('bla')

