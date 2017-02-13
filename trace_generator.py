import os.path
import pickle
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import os
import scipy.interpolate

filename_pulse_shape = 'pulse_SST-1M_AfterPreampLowGain.dat'  # pulse shape template file




def compute_normalized_pulse_shape_area():

    delta_t = np.trapz(amplitudes, time_steps)

    return delta_t

def compute_normalized_pulse_shape():

    time_steps, amplitudes = np.loadtxt(filename_pulse_shape, unpack=True, skiprows=1)
    amplitudes = amplitudes / min(amplitudes)

    return time_steps, amplitudes
def return_interpolant():
    time_steps, amplitudes = compute_normalized_pulse_shape()

    return scipy.interpolate.interp1d(time_steps, amplitudes, kind='cubic', bounds_error=False, fill_value=0.)


interpolant = return_interpolant()

class TraceGenerator:


    def __init__(self, start_time=-100., end_time=100., sampling_time=4., nsb_rate=660 * 1E6 * 1E-9,
                 mean_crosstalk_production=0.08, debug=False, gain_nsb_dependency=False, n_signal_photon=0.,
                 sig_poisson= True, sigma_e=0.86/5.6, sigma_1=0.42/5.6, gain=5.6, seed=0, baseline=10.,
                 *args, **kwargs):

        ## Initialize class attributs
        self.random_bin = True
        self.debug = debug  # boolean for debuging
        self.debug_counter = 0  # debug counter avoiding excessive debug calls
        self.artificial_backward_time = 80.  # artificial backward time to avoid boundary effects
        self.start_time = start_time - self.artificial_backward_time  # start time of the trace
        self.end_time = end_time  # end time of the trace
        self.sampling_time = sampling_time  # FADC sampling time
        self.nsb_rate = nsb_rate  # night sky background or dark count rate per SiPM
        self.mean_crosstalk_production = mean_crosstalk_production  # mean number of crosstalk produced per fired cell
        self.sampling_bins = np.arange(self.start_time, self.end_time,
                                       self.sampling_time)  # FADC sampling times
        self.adc_count = np.zeros(len(self.sampling_bins))  # FADC values
        self.n_signal_photon = n_signal_photon  # number of Cherenkov photons
        self.sig_poisson = sig_poisson # is the number of signal photons distributed as a poisson?

        self.filename_pulse_shape = 'pulse_SST-1M_AfterPreampLowGain.dat'  # pulse shape template file
        self.pe_to_adc = gain
        self.sigma_1 = sigma_1  # spread due to charge resolution in photocounting
        self.sigma_e = sigma_e  # electronic spread in analog to digital conversion
        self.baseline = int(baseline)
        self.cell_capacitance = 85. * 1E-15  # Farad
        self.bias_resistance = 10. * 1E3  # Ohm
        self.gain_nsb_dependency = gain_nsb_dependency
        self.gain = 1. / (
            1. + nsb_rate * self.cell_capacitance * self.bias_resistance * 1E9 * self.gain_nsb_dependency)  # gain taking into account nsb dependency

        time_steps, amplitudes = np.loadtxt(self.filename_pulse_shape, unpack=True, skiprows=1)
        self.time_steps_pulse_shape = time_steps
        amplitudes = amplitudes / min(amplitudes)
        self.amplitudes_pulse_shape = amplitudes


        self.seed = seed
        np.random.seed(seed=self.seed)

        self.plot_title = 'NSB = ' + str(self.nsb_rate * 1E3) + ' [MHz], $\mu_{XT}$ = ' + str(
            self.mean_crosstalk_production) + ' p.e.'
        self.photon_arrival_time = np.zeros(1)
        self.photon = np.zeros(1)

        ## Ploting results

    def __str__(self):

        return ''.join('{}{}'.format(key, val) for key, val in sorted(self.__dict__.items()))


    def __iter__(self):
        return self

    def __next__(self):
         return self.next()

    def fast_config(self,n_signal,nsb_rate):
        self.nsb_rate = nsb_rate
        self.n_signal_photon = n_signal

    def next(self):
        self.photon_arrival_time = np.zeros(1)
        self.photon = np.zeros(1)
        self.sampling_bins = np.arange(self.start_time, self.end_time + self.sampling_time,
                                       self.sampling_time)  # FADC sampling times
        self.adc_count = np.zeros(len(self.sampling_bins))  # FADC values

        ## Compute the FADC signal
        if (self.n_signal_photon > 0 ):
            self.add_signal_photon()

        if self.nsb_rate>0:
            self.generate_nsb()

        if self.mean_crosstalk_production>0:

            self.generate_crosstalk()

        if self.sigma_1>0:

            self.generate_photon_smearing()

        self.compute_analog_signal()
        self.convert_to_digital()
        #return self.n_signal_photon,self.nsb_rate,self.adc_count

        if (self.debug):
            plt.figure()
            plt.title(self.plot_title)
            plt.step(self.sampling_bins, self.adc_count, label='ADC signal', color='b')
            plt.ylabel(r'$ADC_{count}$')
            plt.xlabel(r'$t$ [ns]')
            plt.xlim(np.array([self.start_time + self.artificial_backward_time, self.end_time]))
            # ~ plt.ylim([0, 150])
            plt.legend()

            plt.figure()
            plt.title(self.plot_title)
            plt.hist(self.adc_count, bins=range(min(self.adc_count), max(self.adc_count) + 2, 1), facecolor='green',
                     alpha=0.5, align='left', histtype='step')
            plt.xlabel(r'$ADC_{count}$')
            plt.ylabel(r'count')

            plt.show()

    def interpolated_pulseshape(self, time):

        return interpolant(time)


    def get_adc_count(self):

        return self.adc_count

    def add_signal_photon(self):
        if (self.random_bin):
            self.photon_arrival_time[0] = np.random.uniform(self.start_time ,
                                                        self.end_time)  # + np.random.normal(0, self.sampling_time/100., size=1)
        else:
            self.photon_arrival_time[0] = np.random.uniform(0., self.sampling_time) # + np.random.normal(0, self.sampling_time/100., size=1)
        self.cherenkov_time = self.photon_arrival_time[0]
        self.photon[0] = np.random.poisson(self.n_signal_photon) if self.sig_poisson else self.n_signal_photon

    def generate_nsb(self):
        """ create NSB or dark count photons
        1) the number of photon is randomly created according to NSB level
        2) photons are distrbuted randomly along the time axis
        """
        mean_photon_number = np.random.poisson((self.end_time - self.start_time) * self.nsb_rate)


        #print(mean_photon_number)

        # ~ if (mean_photon_number>0):

        temp = np.random.uniform(size=mean_photon_number) * (
            self.end_time - self.start_time) + self.start_time
        self.photon_arrival_time = np.append(self.photon_arrival_time, temp)
        self.photon = np.append(self.photon, np.ones(len(self.photon_arrival_time) - 1))

        indices_sorted_time = np.argsort(self.photon_arrival_time)
        self.photon_arrival_time = self.photon_arrival_time[indices_sorted_time]
        self.photon = self.photon[indices_sorted_time]

        if (self.debug):

            label = ' '

            if (n_cherenkov_photon > 0):

                label = 'NSB + Cherenkov, $t_{signal} =$ ' + str(self.cherenkov_time) + ' [ns]'

            else:

                label = 'NSB'


            plt.figure()
            plt.title(self.plot_title)
            plt.bar(self.photon_arrival_time, self.photon, label=label, color='k')
            plt.xlabel(r'$t$ [ns]')
            plt.ylabel(r'$N(t)$')
            plt.xlim([self.start_time, self.end_time])
            plt.legend()

        return

    def generate_crosstalk(self):
        """ create crosstalk for each existing photon
        1) the number of crosstalk photons is randomly generated
        2) crosstalk photons are added to the photon list
        """
        crosstalk_photons = np.zeros(len(self.photon))
        for i in range(len(self.photon)):
            for j in range(int(self.photon[i])):
                crosstalk_photons[i] += np.random.poisson(self.mean_crosstalk_production)

        self.photon += crosstalk_photons

        if (self.debug):

            label = ' '

            if (n_cherenkov_photon > 0):

                label = 'NSB + XT + Cherenkov, $t_{signal} =$ ' + str(self.cherenkov_time) + ' [ns]'

            else:

                label = 'NSB + XT'

            plt.figure()
            plt.title(self.plot_title)
            plt.bar(self.photon_arrival_time, self.photon, label=label, color='r')
            plt.xlabel(r'$t$ [ns]')
            plt.ylabel(r'$N(t)$')
            plt.xlim(np.array([self.start_time, self.end_time]))
            plt.legend()

        return

    def generate_photon_smearing(self):
        """ generate smearing due to charge resolution
        1) create a set of gaussian(mean=0, std=N_photon*sigma_1) values (one for each photon)
        2) add the smearing to the photon
        """

        smearing_spread = np.sqrt(self.photon * self.sigma_1 ** 2)

        smearing = np.zeros(len(smearing_spread))

        for i in range(len(smearing_spread)):

            if smearing_spread[i] > 0:

                smearing[i] = np.random.normal(0., smearing_spread[i])

            else:

                smearing[i] = 0.

        self.photon += smearing

        if (self.debug):

            label = ' '

            if (n_cherenkov_photon > 0):

                label = 'NSB + XT + Smearing + Cherenkov, $t_{signal} =$ ' + str(self.cherenkov_time) + ' [ns]'

            else:

                label = 'NSB + XT + Smearing'



            plt.figure()
            plt.title(self.plot_title)
            plt.bar(self.photon_arrival_time, self.photon, label=label, color='b')
            plt.xlabel(r'$t$ [ns]')
            plt.ylabel(r'$N(t)$')
            plt.xlim(np.array([self.start_time, self.end_time]))
            plt.legend()

        return

    def generate_electronic_smearing(self):
        """ generate smearing electronic due to the reading of FADC
        1) create a set of gaussian(mean=0, std=sigma_e) values (one for each photon)
        2) add the smearing to the photon
        """

        smearing_spread = np.sqrt(self.sigma_e ** 2)
        smearing = np.random.normal(0., smearing_spread, size=len(self.sampling_bins))

        self.adc_count += smearing

        return

    def return_pulseshape(self, time):
        """ return an interpolated value of the pulse shape taken from a template file
        1) read the template file
        2) normalize amplitudes such that it is positive and always smaller than 1
        3) return spline value at given time
        """

        if (self.debug and self.debug_counter == 0):
            self.debug_counter += 1
            delta_t = 0.1
            debug_time = np.arange(self.start_time + self.artificial_backward_time, self.end_time + delta_t, delta_t)
            # ~ print debug_time
            # print np.array(self.interpolated_pulseshape(debug_time))

            plt.figure()
            plt.plot(debug_time, self.interpolated_pulseshape(debug_time), label='pulse shape', color='r')
            plt.xlabel(r'$t$ [ns]')
            plt.ylabel(r'amplitude []')
            plt.xlim(np.array([self.start_time, self.end_time]))
            plt.legend()

        return self.interpolated_pulseshape(time)

    def compute_analog_signal(self):

        ## Compute by suming pulse shape

        for i in range(len(self.photon)):
            self.adc_count += self.return_pulseshape(self.sampling_bins - self.photon_arrival_time[i]) * self.photon[i]

        self.adc_count = self.adc_count * self.gain

        if (self.debug):

            delta_t = 0.01
            debug_time = np.arange(self.start_time, self.end_time + delta_t, delta_t)
            debug_analog = np.zeros(len(debug_time))

            for i in range(len(self.photon)):
                debug_analog += self.return_pulseshape(debug_time - self.photon_arrival_time[i]) * self.photon[i]

            plt.figure()
            plt.title(self.plot_title)
            plt.plot(debug_time, debug_analog, label='analog signal', color='r', )
            plt.xlabel(r'$t$ [ns]')
            plt.ylabel(r'$N(t)$')
            plt.xlim(np.array([self.start_time, self.end_time]))
            plt.legend()

        return

    def convert_to_digital(self):

        indices_to_remove = range(0, int(self.artificial_backward_time / self.sampling_time), 1)
        self.adc_count = np.delete(self.adc_count, indices_to_remove)
        self.sampling_bins = np.delete(self.sampling_bins, indices_to_remove)
        self.photon = self.photon[self.photon_arrival_time >= (self.start_time + self.artificial_backward_time)]
        self.photon_arrival_time = self.photon_arrival_time[self.photon_arrival_time>=(self.start_time + self.artificial_backward_time)]

        self.generate_electronic_smearing()
        self.adc_count = self.adc_count * self.pe_to_adc
        self.adc_count = self.adc_count.round().astype(int) + self.baseline

    def save(self, mc_number, path='./'):

        filename = path + 'TRACE' + str(mc_number).zfill(5) + '.p'

        if os.path.isfile(filename):
            print ('file named : ' + filename + ' already exists')
            sys.exit("Error message")

        with open(filename, 'wb') as handle:
            pickle.dump((self.adc_count, self.end_time - self.start_time - self.artificial_backward_time,
                         self.sampling_time, self.nsb_rate, self.mean_crosstalk_production, self.n_cherenkov_photon),
                        handle)

        return


if __name__ == '__main__':

    if len(sys.argv) > 1:

        mc_number = int(sys.argv[1])
        start_time = float(sys.argv[2])
        end_time = float(sys.argv[3])
        n_cherenkov_photon = int(sys.argv[4])
        nsb_rate = float(sys.argv[5])
        debug = False
        path = 'data_dark/'
        N_forced_trigger = 1E3
        save = True

    else:

        mc_number = 0
        start_time = 0.
        end_time = 6000.
        n_cherenkov_photon = 30
        nsb_rate = 100 * 1E-9 * 1E6
        debug = True
        path = './'
        N_forced_trigger = 1
        save = False

    mean_crosstalk_production = 0.08
    sampling_time = 4.
    adc_count = []

    print( 'Number of force trigger generated = ', N_forced_trigger)

    timer = time.time()

    filename = path + 'TRACE' + str(mc_number).zfill(5) + '.p'

    if save:
        if os.path.isfile(filename):
            print( 'file named : ' + filename + ' already exists')
            sys.exit("Error message")

    trace_object = Trace_Generator(start_time, end_time, sampling_time, nsb_rate, mean_crosstalk_production, debug, False,
                            n_cherenkov_photon)

    for i in range(int(N_forced_trigger)):
        trace_object.next()
        adc_count.append(trace_object.adc_count)

    adc_count = np.asarray(adc_count)
    print(adc_count.shape)
    print( time.time() - timer)

    if save:
        with open(filename, 'wb') as handle:
            pickle.dump((adc_count, end_time - start_time, sampling_time, nsb_rate, mean_crosstalk_production,
                         n_cherenkov_photon), handle)
