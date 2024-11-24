def split_data(array, training_ratio):
    """
    Split into train and valid
    seed is always 3
    :param array: 2D array, each row is a sample
    :param ratio: ratio of train in all, e.g. 0.7
    :return: two arrays
    """
    import numpy as np
    N = array.shape[0]
    N1 = int(N * training_ratio)
    np.random.seed(3)
    np.random.shuffle(array)
    np.random.seed()
    train = array[0:N1]
    valid = array[N1:]
    return train, valid

