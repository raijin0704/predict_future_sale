# Adapted from https://www.kaggle.com/sebask/keras-2-0

import time
notebookstart= time.time()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, ThresholdedReLU, MaxPooling2D, Embedding, Dropout
from keras.optimizers import Adam, SGD, RMSprop
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, LSTM, concatenate, Embedding, Flatten
from keras import Input, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import gc
from datetime import datetime

# Viz
import matplotlib.pyplot as plt


def model_study(scale_mode, feature_name, after_dense_F, hidden_depth, hidden_layers1, hidden_layers2):
    file_name = datetime.now().strftime("%m%d%H%M%S") + feature_name + str(hidden_depth) + str(hidden_layers1) + str(hidden_layers2)

    X = np.load("../data_for_kernel/all"+scale_mode+"_X.npy")
    y = np.load("../data_for_kernel/all"+scale_mode+"_y.npy")
    easy_info = np.load("../data_for_kernel/all"+scale_mode+"_EasyInfo.npy")
    test = np.load("../data_for_kernel/all"+scale_mode+"_test.npy")

    # print("Modeling Stage")

    series_input = Input(shape=(X.shape[1], X.shape[2]), dtype='float32', name="series_input")
    
    if hidden_depth==1:
        hidden_series3 = LSTM(hidden_layers1, return_sequences=False)(series_input)
        hidden_series3 = Dropout(0.5)(hidden_series3)

    if hidden_depth==3:
        hidden_series1 = LSTM(hidden_layers2, return_sequences=True)(series_input)
        hidden_series1 = Dropout(0.5)(hidden_series1)
        hidden_series2 = LSTM(hidden_layers2, return_sequences=True)(hidden_series1)
        hidden_series2 = Dropout(0.5)(hidden_series2)
        hidden_series3 = LSTM(hidden_layers1, return_sequences=False)(hidden_series2)
        hidden_series3 = Dropout(0.5)(hidden_series3)

    item_input = Input(shape=(1,), dtype="float32", name="item_input")
    shop_input = Input(shape=(1,), dtype="float32", name="shop_input")
    concatenated_info = concatenate([item_input, shop_input], axis=-1)
    info_dense = Dense(8)(concatenated_info)
    info_dense = Dropout(0.5)(info_dense)
    concatenated_all = concatenate([hidden_series3, info_dense], axis=-1)
    # 1層の全結合を噛ませる場合(２番目にいいモデル)
    if after_dense_F:
        dense_all = Dense(16)(concatenated_all)
        dense_all = Dropout(0.5)(dense_all)
        output = Dense(1, activation="relu", name="model_output")(dense_all)
    # 何もかませない場合(１番いいモデル)
    else:
        output = Dense(1, activation="relu", name="model_output")(concatenated_all)
    
        
    model = Model([series_input, item_input, shop_input], output)
    model.compile(loss="mse", optimizer="adam")

    # Train Model
    # print("\nFit Model")
    VALID = True
    LSTM_PARAM = {"batch_size":128,
                "verbose":1,
                "epochs":100}

    modelstart = time.time()
    if VALID is True:
        X_train, X_valid, y_train, y_valid, easy_info_train, easy_info_valid = train_test_split(X, y, easy_info, test_size=0.10, random_state=1, shuffle=False)
        # del X,y; gc.collect()
        # print("X Train Shape: ",X_train.shape)
        # print("X Valid Shape: ",X_valid.shape)
        # print("y Train Shape: ",y_train.shape)
        # print("y Valid Shape: ",y_valid.shape)
        # print("easy_info Train Shape: ", easy_info_train.shape)
        # print("easy_info Valid Shape: ", easy_info_valid.shape)

        model_dir = '../model/' + file_name
        plot_model(model, to_file= model_dir + '.png', show_shapes=True, show_layer_names=True)
        callbacks_list=[EarlyStopping(monitor="val_loss",min_delta=.001, patience=5,mode='auto'), ModelCheckpoint(model_dir + ".hdf5", save_best_only=True)]
        hist = model.fit({
                            "series_input":X_train,
                            "item_input":easy_info_train[:,0],
                            "shop_input":easy_info_train[:,1]
                        },
                        {
                            "model_output":y_train
                        },
                            validation_data=({
                                                    "series_input":X_valid,
                                                    "item_input":easy_info_valid[:,0],
                                                    "shop_input":easy_info_valid[:,1]
                                                },
                                                {
                                                    "model_output":y_valid
                                                }),
                            callbacks=callbacks_list,
                            **LSTM_PARAM)
        pred = model.predict({
                                "series_input":test,
                                "item_input":easy_info[:,0],
                                "shop_input":easy_info[:,1]
                            }, verbose=1)

        # Model Evaluation
        best = np.argmin(hist.history["val_loss"])
        print(file_name)
        print("Optimal Epoch: {}",best)
        print("Train Score: {}, Validation Score: {}".format(hist.history["loss"][best],hist.history["val_loss"][best]))

        plt.plot(hist.history['loss'], label='train')
        plt.plot(hist.history['val_loss'], label='validation')
        plt.xlabel("Epochs")
        plt.ylabel("Mean Square Error")
        plt.legend()
        # plt.show()
        plt.savefig("../result/loss_graph/" + file_name + " Train and Validation MSE Progression.png")

    if VALID is False:
        pass

    print("\nOutput Submission")
    submission = pd.DataFrame(pred,columns=['item_cnt_month'])
    submission.to_csv('../result/submit/' + file_name + " submission.csv",index_label='ID')
    print(submission.head())
    print("\nModel Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
    print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))



if __name__ == "__main__":
    # model_study("Standard", "all1", False)        
    # model_study("Standard", "all2", True)

    # set_kernel_numpy_data("Standard")
    # set_kernel_numpy_data("MinMax")
    for hidden_depth in [1,3]:
        for hidden_layers1 in [16,32]:
            for hidden_layers2 in [16,32]:
                model_study("Standard", "all1", False, hidden_depth, hidden_layers1, hidden_layers2)