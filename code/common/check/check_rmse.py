import sys, os
sys.path.append(os.getcwd())
from common.check.get_rmse import get_rmse
from common.load.load_model import load_all_numpy_data, load_lstm_model
from common.inference.predict_sales import get_predict_sales
from common.check.get_rmse import get_rmse

def check_rmse(data_name, model_name, y_standard_F):
    _, _, test_x, test_y, _, test_y_mean, test_y_std, _, test_y_ori, _, test_info = load_all_numpy_data(data_name)
    model = load_lstm_model(model_name)
    test_y_pre, _ = get_predict_sales(model, test_x, test_y_mean, test_y_std, y_standard_F, model_name, test_info)
    if y_standard_F:
        rmse = get_rmse(test_y, test_y_pre)
    else:
        rmse = get_rmse(test_y_ori, test_y_pre)
    print('Val RMSE: %.3f' % rmse)

    return rmse

if __name__ == "__main__":
    check_rmse("simple_data", "simple_data_simple_LSTM_1depth_8layers_relu-activation.hdf5", True)