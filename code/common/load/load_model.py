import numpy as np
import json
from keras.models import load_model

def load_all_numpy_data(data_name):
    train_x = np.load("../data_for_lstm/" + data_name + "_train_x.npy")
    train_y = np.load("../data_for_lstm/" + data_name + "_train_y.npy")
    test_x = np.load("../data_for_lstm/" + data_name + "_test_x.npy")
    test_y = np.load("../data_for_lstm/" + data_name + "_test_y.npy")
    test_y_label = np.load("../data_for_lstm/" + data_name + "_test_y_label.npy")
    with open("../data_for_lstm/" + data_name + "_test_y_summary.json", "r") as f:
        info_dic = json.load(f)
    test_y_mean = info_dic["test_y_mean"]
    test_y_std = info_dic["test_y_std"]
    trian_y_ori = np.load("../data_for_lstm/" + data_name + "_train_y_ori.npy")
    test_y_ori = np.load("../data_for_lstm/" + data_name + "_test_y_ori.npy")
    train_info = np.load("../data_for_lstm/" + data_name + "_train_info.npy")
    test_info = np.load("../data_for_lstm/" + data_name + "_test_info.npy")
    train_easy_info = np.load("../data_for_lstm/" + data_name + "_train_easy_info.npy")
    test_easy_info = np.load("../data_for_lstm/" + data_name + "_test_easy_info.npy")


    return train_x, train_y, test_x, test_y, test_y_label, test_y_mean, test_y_std, trian_y_ori, test_y_ori, train_info, test_info, train_easy_info, test_easy_info

def load_lstm_model(model_file_name):
    model = load_model("../model/" + model_file_name)
    model.summary()

    return model


if __name__ == "__main__":
    train_x, train_y, test_x, test_y, test_y_label, test_y_mean, test_y_std, train_y_ori, test_y_ori, train_info, test_info, train_easy_info, test_easy_info = load_all_numpy_data("simple_data")
    model = load_lstm_model("simple_data_simple_LSTM_1depth_8layers_relu-activation.hdf5")


