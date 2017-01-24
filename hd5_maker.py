import numpy as np
import h5py




def createChunk(options,tr):
    """
    Create a data chunck
    :param tr:
    :param chunkSize:
    :return:
    """
    chunkSize = options.batch_max
    ii = 0
    labels,traces = None,None
    while ii < chunkSize:
        tr.nsb_rate = ((np.random.random_sample() * (options.nsb_range[1]-options.nsb_range[0]))+options.nsb_range[0]) * 1e-3
        tr.n_signal_photon = int((np.random.random_sample() * (options.photon_range[1]-options.photon_range[0]))+options.photon_range[0])
        next(tr)
        ravelled_arrival = []
        for k,t0 in enumerate(tr.photon_arrival_time):
            for kk in range(int(round(tr.photon[k]))):
                ravelled_arrival.append(t0)

        n_photons,axis  = np.histogram(ravelled_arrival,
                                  bins=np.arange(options.photon_times[0],
                                                 options.photon_times[1]+options.photon_times[2]+options.target_segmentation,
                                                 options.target_segmentation))

        n_adcs = tr.adc_count.repeat(int(np.round(options.photon_times[2]/options.target_segmentation)))
        if type(traces).__name__ == 'ndarray':
            traces = np.append(traces,n_adcs.reshape((1,)+n_adcs.shape),axis=0)
            labels = np.append(labels,n_photons.reshape((1,)+n_photons.shape),axis=0)
        else:
            traces = n_adcs.reshape((1,)+n_adcs.shape)
            labels = n_photons.reshape((1,)+n_photons.shape)
        ii+=1
    return traces.reshape(traces.shape+(1,)),labels.reshape(labels.shape+(1,))



def createFile( options,tr , filename ):
    """
    Create a large HDF5 data file
    """
    chunckSize, finalSize = options.batch_max,options.evt_max

    print("Progress {:2.1%}".format(0.), end="\r")

    chunks = createChunk(options,tr)

    f = h5py.File(filename+'.hdf5', 'w')
    f.create_dataset('traces', data=chunks[0], chunks=True, maxshape=(None,chunks[0].shape[1],chunks[0].shape[2]))
    f.create_dataset('labels', data=chunks[1], chunks=True, maxshape=(None,chunks[1].shape[1],chunks[1].shape[2]))
    traces_dataset = f['traces']
    labels_dataset = f['labels']

    nChunks = finalSize // chunckSize
    for i in range(nChunks):
        print("Progress {:2.1%}".format(float(i*options.batch_max) / options.evt_max), end="\r")
        chunks = createChunk(options,tr)
        newshape = [traces_dataset.shape[0] + chunks[0].shape[0],chunks[0].shape[1],chunks[0].shape[2]]
        traces_dataset.resize(newshape)
        newshape_label = [labels_dataset.shape[0] + chunks[1].shape[0],chunks[1].shape[1],chunks[1].shape[2]]
        labels_dataset.resize(newshape_label)
        traces_dataset[-chunks[0].shape[0]:] = chunks[0]
        labels_dataset[-chunks[1].shape[0]:] = chunks[1]

    f.close()




if __name__ == '__main__':
    from optparse import OptionParser
    from trace_generator import trace_generator
    import numpy as np
    import h5py

    # Job configuration
    parser = OptionParser()
    parser.add_option("-n", "--evt_max", dest="evt_max",
                      help="maximal number of events", default=100000, type=int)

    parser.add_option("--batch_max", dest="batch_max",
                      help="maximal number of events for batch in memory", default=1000, type=int)

    parser.add_option("-d", "--directory", dest="directory",
                      help="Output directory", default="/data/datasets/CTA/ToyNN/")

    parser.add_option("-f", "--filename", dest="filename",
                      help="Output file name", default="train_0_200_0")

    parser.add_option("-p", "--photon_range", dest="photon_range",
                      help="range of signal photons", default="0.,0.1")

    parser.add_option("-b", "--nsb_range", dest="nsb_range",
                      help="range of NSB", default="0.,200.")

    parser.add_option("--photon_times", dest="photon_times",
                      help="arrival time range", default="-150.,150.,4")

    parser.add_option("--target_segmentation", dest="target_segmentation",
                      help="arrival time range", default="2")

    (options, args) = parser.parse_args()
    options.photon_range = [float(n) for n in options.photon_range.split(',')]
    options.photon_times = [float(n) for n in options.photon_times.split(',')]
    options.nsb_range = [float(n) for n in options.nsb_range.split(',')]
    options.target_segmentation = float(options.target_segmentation)
    # Create the trace generator
    tr = trace_generator(start_time=options.photon_times[0],end_time=options.photon_times[1],sig_poisson=False)
    # Create the file
    filename = options.directory + options.filename
    createFile( options,tr , filename)
