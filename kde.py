import iminuit,probfit
import scipy.interpolate
import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt
import h5py


time_steps, amplitudes = np.loadtxt('pulse_SST-1M_AfterPreampLowGain.dat', unpack=True, skiprows=1)
amplitudes = amplitudes / min(amplitudes)
interpolated_pulseshape = scipy.interpolate.interp1d(time_steps, amplitudes, kind='cubic',
                                                            bounds_error=False, fill_value=0.)
result = integrate.quad(interpolated_pulseshape , 0., 1000.)[0]
v_pulseshape = np.vectorize(lambda x : (interpolated_pulseshape(x)/result))

np.random.seed(0)



def mytemplate(x,t0):
    y = interpolated_pulseshape(x-t0)*5.6
    return y

def multi4(x,b,t1,t2,t3,t4):
    y = interpolated_pulseshape(x-t1)*5.6\
        +interpolated_pulseshape(x-t2)*5.6\
        +interpolated_pulseshape(x-t3)*5.6\
        +interpolated_pulseshape(x-t4)*5.6+b
    return y


if __name__ == '__main__':
    f = h5py.File('/data/datasets/CTA/ToyNN/test_kde.hdf5', 'r')
    i = 0
    n = 0
    while n!= 4 :
        n=np.nonzero(f['labels'][i].reshape((f['labels'][i].shape[0],)))[0].size
        print(n)
        i+=1
    print('trace',i)
    trace = f['traces'][i-1]
    trace=trace.reshape((trace.shape[0],))
    label = f['labels'][i-1]
    label=label.reshape((label.shape[0],))
    #trace= trace-10.
    f.close()
    plt.ion()
    plt.figure()
    plt.step(np.arange(-150.,154,4),trace)
    plt.step(np.arange(-150.,154,4),label)

    chi2 = probfit.Chi2Regression(multi4, np.arange(-150., 154, 4), trace, np.ones((trace.shape[0],))*1.)
    binned_likelihood = probfit.BinnedLH(multi4, trace,bins=trace.shape[0],bound=[-150.,150.])

    minuit = iminuit.Minuit(chi2,b=20.,error_t1=0.5,error_t2=.5,error_t3=.5,error_t4=.5,limit_t1=(-150.,150.),
                            limit_t2=(-150.,150.),limit_t3=(-150.,150.),limit_t4=(-150.,150.))


    minuit.migrad(ncall=1000000)
    minuit.hesse()
    my_fitarg = minuit.fitarg

    minuit = iminuit.Minuit(chi2,**my_fitarg)
    minuit.migrad(ncall=1000000)
    minuit.hesse()
    #binned_likelihood.draw(minuit)

    v_pulseshape = np.vectorize(lambda x : multi4(x,minuit.get_param_states()[0]['value'],
                                                  minuit.get_param_states()[1]['value'],
                                                  minuit.get_param_states()[2]['value'],
                                                  minuit.get_param_states()[3]['value'],
                                                  minuit.get_param_states()[4]['value']))

    plt.step(np.arange(-150., 154, 4),v_pulseshape(np.arange(-150., 154, 4)) )
    plt.step(np.arange(-150., 154, 4),v_pulseshape(np.arange(-150., 154, 4)) )
    plt.show()
    print(trace.shape)
    print(iminuit.describe(mytemplate))
