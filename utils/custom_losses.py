from keras import backend as K


def custom_mean_squared_error_traces(y_true, y_pred,margin = 10):
    """
    Compute mean square error only in the central bins of the trace
    (ie. with a margin on each side of 10 samples)
    :param y_true:
    :param y_pred:
    :param margin:
    :return:
    """
    return K.mean(K.square(y_pred[:,margin:-margin] - y_true[:,margin:-margin]), axis=-1)
