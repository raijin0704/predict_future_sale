import numpy as np
import json

def save_numpy_data(data_name, train_x, train_y, test_x, test_y, test_y_label, test_y_mean, test_y_std, train_y_ori, test_y_ori, train_info, test_info, train_easy_info, test_easy_info):
    """[summary]
    
    Arguments:
        data_name {[type]} -- [description]
        train_x {[type]} -- [description]
        train_y {[type]} -- [description]
        test_x {[type]} -- [description]
        test_y {[type]} -- [description]
        test_y_label {[type]} -- [description]
        test_y_mean {[type]} -- [description]
        test_y_std {[type]} -- [description]
        train_info{[]} -- []
        test_info{[]} -- []
    """
    np.save("../data_for_lstm/" + data_name + "_train_x.npy", train_x)
    np.save("../data_for_lstm/" + data_name + "_train_y.npy", train_y)
    np.save("../data_for_lstm/" + data_name + "_test_x.npy", test_x)
    np.save("../data_for_lstm/" + data_name + "_test_y.npy", test_y)
    np.save("../data_for_lstm/" + data_name + "_test_y_label.npy", test_y_label)
    with open("../data_for_lstm/" + data_name + "_test_y_summary.json", "w") as f:
        json.dump({"test_y_mean":test_y_mean, "test_y_std":test_y_std}, f, indent=4)
    np.save("../data_for_lstm/" + data_name + "_train_y_ori.npy", train_y_ori)
    np.save("../data_for_lstm/" + data_name + "_test_y_ori.npy", test_y_ori)
    np.save("../data_for_lstm/" + data_name + "_train_info.npy", train_info)
    np.save("../data_for_lstm/" + data_name + "_test_info.npy", test_info)
    np.save("../data_for_lstm/" + data_name + "_train_easy_info.npy", train_easy_info)
    np.save("../data_for_lstm/" + data_name + "_test_easy_info.npy", test_easy_info)
    
    