#%% [markdown]
# # LSTMで予測
#%% [markdown]
# 1. データをある店舗で売っているある商品(商品id×店舗id)に分類する
# 2. 2013/1～2015/9のデータを使って2015/10の売り上げを予測する
# 3. 説明変数は月売り上げ、
# 4. train/val/testの比率は

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#%%
from keras.models import Sequential
from keras.layers import Dense, Dropout, TimeDistributed
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.utils import plot_model

#%%
import json
import pickle


from common.prepro.downcast_dtypes import downcast_dtypes
from common.prepro.get_pivot_table import get_simple_pivot_table
from common.load.load_model import load_lstm_model, load_all_numpy_data



# sales = pd.read_csv('../../data/sales_train_v2.csv')

# #%%
# sales = downcast_dtypes(sales)
# print(sales.info())

# sales_by_item_id_shop_id = get_simple_pivot_table(sales, ["item_id", "shop_id"], ["date_block_num"], ["item_cnt_day"])

# print(sales_by_item_id_shop_id.head())


# #%%
# train, test = train_test_split(sales_by_item_id_shop_id, test_size=0.2, random_state=10)

# #%%
# train_x ,train_y, _, _, _ = encord_simple_data_for_study_data(train)
# test_x ,test_y, test_y_label, test_y_mean, test_y_std = encord_simple_data_for_study_data(test)

# #%%
# print(train_y.shape)
# print(train_x.shape)


#%%

def model_study(depth, layers, batch, activation, model_name, data_name, y_standard_F):

    # 各種パラメータ
    hidden_depth = depth
    hidden_layers = layers
    batch_size = batch
    hidden_activation = activation
    model_name = model_name
    data_name = data_name
    y_standard_F = y_standard_F

    train_x, train_y, test_x, test_y, test_y_label, test_y_mean, test_y_std, train_y_ori, test_y_ori, _, _, _, _ = load_all_numpy_data(data_name)
    file_name = data_name + "_" + model_name + "_" + str(hidden_depth)+ "depth_" + str(hidden_layers) + "layers_"  + hidden_activation + "-activation"
    if y_standard_F:
        pass
    else:
        file_name = file_name + "_y_ori"

    model = Sequential()
    if hidden_depth==1:
        model.add(LSTM(hidden_layers, input_shape=(train_x.shape[1], train_x.shape[2]), activation=hidden_activation, recurrent_dropout=0.5, return_sequences=False))
    else:
        model.add(LSTM(hidden_layers, input_shape=(train_x.shape[1], train_x.shape[2]), activation=hidden_activation, recurrent_dropout=0.5, return_sequences=True))
        while hidden_depth>2:
            model.add(LSTM(hidden_layers, activation=hidden_activation, recurrent_dropout=0.5, return_sequences=True))
            hidden_depth -= 1
        model.add(LSTM(hidden_layers, activation=hidden_activation, recurrent_dropout=0.5, return_sequences=False))

    model.add(Dense(1, activation=hidden_activation)) #yの標準化をやめたことにより活性化関数にrelu関数を通すことにした
    # model.add(TimeDistributed(Dense(1)))

    

    model.compile(loss="mse", optimizer="adam")

    model.summary()

    model_dir = '../model/' + file_name
    plot_model(model, to_file= model_dir + '.png')
    mc_cb = ModelCheckpoint(model_dir + ".hdf5", save_best_only=True)
    es_cb = EarlyStopping(monitor='val_loss' ,patience=7, verbose=1) # simple_dataではpatience=3だった
    if y_standard_F:
        history = model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=100, validation_split=0.2, callbacks=[mc_cb,es_cb])
    else:
        history = model.fit(x=train_x, y=train_y_ori, batch_size=batch_size, epochs=100, validation_split=0.2, callbacks=[mc_cb,es_cb])
    with open("../result/other/history/" + file_name + ".pickle", mode="wb") as f:
        pickle.dump(history.history, f)

    test_y_pre = model.predict(test_x)
    if y_standard_F:
        test_y_pre_fix = (test_y_pre + test_y_mean) * test_y_std
        result_check = pd.DataFrame(np.concatenate([test_y_label, test_y_pre_fix.reshape(-1,1)], axis=1))
    else:
        result_check = pd.DataFrame(np.concatenate([test_y_label, test_y_pre.reshape(-1,1)], axis=1))

    result_check.columns = ["item_id","shop_id","正解値","予測値"]
    result_dir = "../result/test/" + file_name + ".csv"
    result_check.to_csv(result_dir, encoding='cp932')
    if y_standard_F:
        rmse = np.sqrt(mean_squared_error(test_y,test_y_pre))
    else:
        rmse = np.sqrt(mean_squared_error(test_y_ori,test_y_pre))

    
    print('Val RMSE: %.3f' % rmse)


# hidden_depth = 1
# hidden_layers = 16
batch_size = 64 #simple_model(depth=1以外)はbatch_size=16なので注意
hidden_activation = "relu"
model_name = "simple_LSTM"
data_name = "simple_price_data"
y_standard_F = False


# for hidden_depth in [1,2]:
#     for hidden_layers in [16,32,64]:
#         for range_month in range(1,5):
#             data_name = "simple_data_" +  str(range_month) + "mfilter"
#             model_study(hidden_depth, hidden_layers, batch_size, hidden_activation, model_name, data_name, y_standard_F)

for hidden_depth in [1,2]:
    for hidden_layers in [16,32,64]:
        model_study(hidden_depth, hidden_layers, batch_size, hidden_activation, model_name, data_name, y_standard_F)

# model_study(1, 16, batch_size, hidden_activation, model_name, data_name, y_standard_F)