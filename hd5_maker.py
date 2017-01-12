import numpy as np
import h5py




def createChunk(options,tr,chunkSize = 1000):
    """
    Create a data chunck
    :param tr:
    :param chunkSize:
    :return:
    """
    ii = 0
    labels,traces = None,None
    while ii < chunkSize:
        tr.nsb_rate = ((np.random.random_sample() * (options.nsb_range[1]-options.nsb_range[0]))+options.nsb_range[0]) * 1e-3
        tr.n_signal_photon = int((np.random.random_sample() * (options.photon_range[1]-options.photon_range[0]))+options.photon_range[0])
        event = next(tr)
        n_photons = [0] * event[2].shape[0]
        for i in range(event[2].shape[0]):
            for k, t0 in enumerate(tr.photon_arrival_time):
                if t0 + 1e-6 > (4 * i) - (4 * 5) and t0 + 1e-6 < 4 * (i + 1) - 4 * 5:
                    n_photons[i] += np.int(round(tr.photon[k]))
        n_photons = np.array(n_photons)
        if type(traces).__name__ == 'ndarray':
            traces = np.append(traces,event[2].reshape((1,)+event[2].shape),axis=0)
            labels = np.append(labels,n_photons.reshape((1,)+n_photons.shape),axis=0)
        else:
            traces = event[2].reshape((1,)+event[2].shape)
            labels = n_photons.reshape((1,)+n_photons.shape)
        ii+=1
    return traces.reshape(traces.shape+(1,)),labels.reshape(labels.shape+(1,))



def createFile( options,tr , filename , chunckSize = 1000 , finalSize=100000):
    """
    Create a large HDF5 data file
    """

    print("Progress {:2.1%}".format(0.), end="\r")

    chunks = createChunk(options,tr,chunckSize)

    f = h5py.File(filename+'.hdf5', 'w')
    f.create_dataset('traces', data=chunks[0], chunks=True, maxshape=(None,chunks[0].shape[1],chunks[0].shape[2]))
    f.create_dataset('labels', data=chunks[1], chunks=True, maxshape=(None,chunks[1].shape[1],chunks[1].shape[2]))
    traces_dataset = f['traces']
    labels_dataset = f['labels']

    nChunks = finalSize // chunckSize
    for i in range(nChunks):
        print("Progress {:2.1%}".format(float(i*options.batch_max) / options.evt_max), end="\r")
        chunks = createChunk(options,tr,chunckSize)
        newshape = [traces_dataset.shape[0] + chunks[0].shape[0],chunks[0].shape[1],chunks[0].shape[2]]
        traces_dataset.resize(newshape)
        newshape = [labels_dataset.shape[0] + chunks[1].shape[0],chunks[1].shape[1],chunks[1].shape[2]]
        labels_dataset.resize(newshape)
        traces_dataset[-chunks[0].shape[0]:] = chunks[0]
        labels_dataset[-chunks[0].shape[0]:] = chunks[0]

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
                      help="Output file name", default="train_0_660_0")

    parser.add_option("-p", "--photon_range", dest="photon_range",
                      help="range of signal photons", default="0,0.1")

    parser.add_option("-b", "--nsb_range", dest="nsb_range",
                      help="range of NSB", default="0.,660.")

    (options, args) = parser.parse_args()
    options.photon_range = [float(n) for n in options.photon_range.split(',')]
    options.nsb_range = [float(n) for n in options.nsb_range.split(',')]

    # Create the trace generator
    tr = trace_generator(sig_poisson=False)
    # Create the file
    filename = options.directory + options.filename
    createFile( options,tr , filename, chunckSize = options.batch_max , finalSize=options.evt_max)