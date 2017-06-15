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

def custom_mean_squared_error_traces_10(y_true, y_pred):
    """
    Compute mean square error only in the central bins of the trace
    (ie. with a margin on each side of 10 samples)
    :param y_true:
    :param y_pred:
    :param margin:
    :return:
    """
    return K.mean(K.square(y_pred[:,10:-10] - y_true[:,10:-10]), axis=-1)

def custom_mean_squared_error_traces_relative(y_true, y_pred,margin = 10):

    return K.mean(K.square((y_true[:,margin:-margin] - y_pred[:,margin:-margin]) / K.clip(K.abs(y_true[:,margin:-margin]),
                                            K.epsilon(),
                                            None)))

def custom_mean_squared_error_traces_relative_10(y_true, y_pred):
    diff = K.clip(K.abs(y_true[:,10:-10] - y_pred[:,10:-10]),K.epsilon()/100.,None)
    denom = K.clip(y_true[:,10:-10],K.epsilon(),None)
    return K.mean(K.square(1+ diff /denom ))
