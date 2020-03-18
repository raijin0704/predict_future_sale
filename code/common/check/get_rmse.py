import numpy as np
from sklearn.metrics import mean_squared_error


def get_rmse(correct, prediction):
    rmse = np.sqrt(mean_squared_error(correct, prediction))

    return rmse