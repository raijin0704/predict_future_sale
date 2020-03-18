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

from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
import gc
from datetime import datetime

# Viz
import matplotlib.pyplot as plt




def set_kernel_numpy_data(scale_mode):
    # Import data
    sales = pd.read_csv('../../data/sales_train_v2.csv', parse_dates=['date'], infer_datetime_format=True, dayfirst=True)
    shops = pd.read_csv('../../data/shops.csv')
    items = pd.read_csv('../../data/items.csv')
    cats = pd.read_csv('../../data/item_categories.csv')
    val = pd.read_csv('../../data/test.csv')

    # Rearrange the raw data to be monthly sales by item-shop
    df = sales.groupby([sales.date.apply(lambda x: x.strftime('%Y-%m')),'item_id','shop_id']).sum().reset_index()
    df = df[['date','item_id','shop_id','item_cnt_day']]
    df["item_cnt_day"].clip(0.,20.,inplace=True)
    df = df.pivot_table(index=['item_id','shop_id'], columns='date',values='item_cnt_day',fill_value=0).reset_index()

    # Merge data from monthly sales to specific item-shops in test data
    test = pd.merge(val,df,on=['item_id','shop_id'], how='left').fillna(0)

    # Strip categorical data so keras only sees raw timeseries
    test = test.drop(labels=['ID','item_id','shop_id'],axis=1)


    # 時系列以外の情報も加える
    item_mean = pd.read_csv("../../data/feature_engineering/item_cnt_month/mean_by_item_id.csv")
    shop_mean = pd.read_csv("../../data/feature_engineering/item_cnt_month/mean_by_shop_id.csv")

    if scale_mode=="MinMax":
        scaler = MinMaxScaler(feature_range=(0, 1))
    elif scale_mode=="Standard":
        scaler = StandardScaler()

    item_mean["item_cnt_month_mean"] = scaler.fit_transform(item_mean["item_cnt_month_mean"].values.reshape(-1,1))
    shop_mean["item_cnt_month_mean"] = scaler.fit_transform(shop_mean["item_cnt_month_mean"].values.reshape(-1,1))

    easy_info_df = pd.merge(val, item_mean, how="left", on="item_id")
    easy_info_df = pd.merge(easy_info_df, shop_mean, how="left", on="shop_id").fillna(0)
    easy_info_df = easy_info_df.drop(labels=["ID",'item_id','shop_id'], axis=1)
    easy_info_df.columns = ["item_info", "shop_info"]
    easy_info = easy_info_df.values

    # Rearrange the raw data to be monthly average price by item-shop
    # Scale Price

    sales["item_price"] = scaler.fit_transform(sales["item_price"].values.reshape(-1,1))
    df2 = sales.groupby([sales.date.apply(lambda x: x.strftime('%Y-%m')),'item_id','shop_id']).mean().reset_index()
    df2 = df2[['date','item_id','shop_id','item_price']].pivot_table(index=['item_id','shop_id'], columns='date',values='item_price',fill_value=0).reset_index()


    b = []
    for i in df2.columns[2:]:
        b.append(int(i.split("-")[1]))
    ohe = OneHotEncoder(sparse=False, dtype=np.float32)
    month = ohe.fit_transform(np.array(b).reshape(-1,1))

    # Merge data from average prices to specific item-shops in test data
    price = pd.merge(val,df2,on=['item_id','shop_id'], how='left').fillna(0)
    price = price.drop(labels=['ID','item_id','shop_id'],axis=1)

    # Create x and y training sets from oldest data points
    y_train = test['2015-10']
    x_sales = test.drop(labels=['2015-10'],axis=1)
    # x_sales = x_sales.values.reshape((x_sales.shape[0], x_sales.shape[1], 1))
    x_sales = scaler.fit_transform(x_sales.values)
    x_sales = x_sales.reshape((x_sales.shape[0], x_sales.shape[1], 1))

    x_prices = price.drop(labels=['2015-10'],axis=1)
    x_prices= x_prices.values.reshape((x_prices.shape[0], x_prices.shape[1], 1))
    X = np.append(x_sales,x_prices,axis=2)

    y = y_train.values.reshape((214200, 1))
    print("Training Predictor Shape: ",X.shape)
    print("Training Predictee Shape: ",y.shape)
    del y_train, x_sales; gc.collect()

    # Transform test set into numpy matrix
    test = test.drop(labels=['2013-01'],axis=1)
    # x_test_sales = test.values.reshape((test.shape[0], test.shape[1], 1))
    x_test_sales = scaler.fit_transform(test.values)
    x_test_sales = x_test_sales.reshape((test.shape[0], test.shape[1], 1))
    x_test_prices = price.drop(labels=['2013-01'],axis=1)
    x_test_prices = x_test_prices.values.reshape((x_test_prices.shape[0], x_test_prices.shape[1], 1))

    # Combine Price and Sales Df
    test = np.append(x_test_sales,x_test_prices,axis=2)
    del x_test_sales,x_test_prices, price; gc.collect()
    print("Test Predictor Shape: ",test.shape)

    # np.save("../data_for_kernel/"+scale_mode+"_X.npy", X)
    # np.save("../data_for_kernel/"+scale_mode+"_y.npy", y)
    # np.save("../data_for_kernel/"+scale_mode+"_EasyInfo.npy", easy_info)
    # np.save("../data_for_kernel/"+scale_mode+"_test.npy", test)

    # np.save("../data_for_kernel/all"+scale_mode+"_X.npy", X)
    # np.save("../data_for_kernel/all"+scale_mode+"_y.npy", y)
    # np.save("../data_for_kernel/all"+scale_mode+"_EasyInfo.npy", easy_info)
    # np.save("../data_for_kernel/all"+scale_mode+"_test.npy", test)
    

def model_study(scale_mode, feature_name, after_dense_F):
    file_name = datetime.now().strftime("%m%d%H%M%S") + feature_name

    X = np.load("../data_for_kernel/all"+scale_mode+"_X.npy")
    y = np.load("../data_for_kernel/all"+scale_mode+"_y.npy")
    easy_info = np.load("../data_for_kernel/all"+scale_mode+"_EasyInfo.npy")
    test = np.load("../data_for_kernel/all"+scale_mode+"_test.npy")

    print("Modeling Stage")

    series_input = Input(shape=(X.shape[1], X.shape[2]), dtype='float32', name="series_input")
    hidden_series1 = LSTM(16, return_sequences=True)(series_input)
    hidden_series1 = Dropout(0.5)(hidden_series1)
    hidden_series2 = LSTM(32, return_sequences=False)(hidden_series1)
    hidden_series2 = Dropout(0.5)(hidden_series2)

    item_input = Input(shape=(1,), dtype="float32", name="item_input")
    shop_input = Input(shape=(1,), dtype="float32", name="shop_input")
    concatenated_info = concatenate([item_input, shop_input], axis=-1)
    info_dense = Dense(8)(concatenated_info)
    info_dense = Dropout(0.5)(info_dense)
    concatenated_all = concatenate([hidden_series2, info_dense], axis=-1)
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
    print("\nFit Model")
    VALID = True
    LSTM_PARAM = {"batch_size":128,
                "verbose":1,
                "epochs":100}

    modelstart = time.time()
    if VALID is True:
        X_train, X_valid, y_train, y_valid, easy_info_train, easy_info_valid = train_test_split(X, y, easy_info, test_size=0.10, random_state=1, shuffle=False)
        # del X,y; gc.collect()
        print("X Train Shape: ",X_train.shape)
        print("X Valid Shape: ",X_valid.shape)
        print("y Train Shape: ",y_train.shape)
        print("y Valid Shape: ",y_valid.shape)
        print("easy_info Train Shape: ", easy_info_train.shape)
        print("easy_info Valid Shape: ", easy_info_valid.shape)

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

    set_kernel_numpy_data("Standard")
    set_kernel_numpy_data("MinMax")
