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

#%%
import json


#%%
sales = pd.read_csv('../../data/sales_train_v2.csv')

#%%
# データのバイト数を落としてメモリを節約する
def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df

sales = downcast_dtypes(sales)
print(sales.info())


#%%
# データをピボットテーブルで集計　※https://note.nkmk.me/python-pandas-pivot-table/
# 列：[item_id, shop_id]、行：date_block_num(月)、値：item_cnt_day(売れた商品の数)

sales_by_item_id_shop_id = sales.pivot_table(index=['item_id', 'shop_id'], values=['item_cnt_day'],
                                        columns='date_block_num', aggfunc=np.sum, fill_value=0).reset_index()
sales_by_item_id_shop_id.columns = sales_by_item_id_shop_id.columns.droplevel().map(str)
sales_by_item_id_shop_id = sales_by_item_id_shop_id.reset_index(drop=True).rename_axis(None, axis=1)
sales_by_item_id_shop_id.columns.values[0] = 'item_id'
sales_by_item_id_shop_id.columns.values[1] = 'shop_id'

print(sales_by_item_id_shop_id.head())


#%%
train, test = train_test_split(sales_by_item_id_shop_id, test_size=0.2, random_state=10)


#%%
# データを説明変数と目的変数に分割し、標準化する
def encord_data(data):    
    x_ori = data.iloc[:,2:-1].values
    x = (x_ori-x_ori.mean()/x_ori.std(ddof=1))  # 不偏標準偏差で標準化している
    x = x.reshape((-1,33,1))
    y_ori = data.iloc[:,-1].values
    y_mean = np.mean(y_ori)
    y_std = np.std(y_ori, ddof=1)
    y = (y_ori-y_ori.mean()/y_ori.std(ddof=1))  # 不偏標準偏差で標準化している
    y = y.reshape((-1,1))
    y_label = np.concatenate([data.iloc[:,0:2].values, y_ori.reshape([-1,1])], axis=1)
    return x, y, y_label, y_mean, y_std


#%%
train_x ,train_y, _, _, _ = encord_data(train)
test_x ,test_y, test_y_label, test_y_mean, test_y_std = encord_data(test)

#%%
def save_numpy_data(data_name, train_x, train_y, test_x, test_y, test_y_label, test_y_mean, test_y_std):
    np.save("../data_for_lstm/" + data_name + "_train_x.npy", train_x)
    np.save("../data_for_lstm/" + data_name + "_train_y.npy", train_y)
    np.save("../data_for_lstm/" + data_name + "_test_x.npy", test_x)
    np.save("../data_for_lstm/" + data_name + "_test_y.npy", test_y)
    np.save("../data_for_lstm/" + data_name + "_test_y_label.npy", test_y_label)
    with open("../data_for_lstm/" + data_name + "_test_y_summary.json", "w") as f:
        json.dump({"test_y_mean":test_y_mean, "test_y_std":test_y_std}, f, indent=4)

#%%
save_numpy_data("simple_data", train_x, train_y, test_x, test_y, test_y_label, test_y_mean, test_y_std)

#%%
print(train_y.shape)
print(train_x.shape)


#%%

def model_study(depth, layers, batch, activation, model_name, data_name):

    # 各種パラメータ
    hidden_depth = depth
    hidden_layers = layers
    batch_size = batch
    hidden_activation = activation
    model_name = model_name
    data_name = data_name

    file_name = data_name + "_" + model_name + "_" + str(hidden_depth)+ "depth_" + str(hidden_layers) + "layers_"  + hidden_activation + "-activation"

    #%%
    model = Sequential()
    if hidden_depth==1:
        model.add(LSTM(hidden_layers, input_shape=(train_x.shape[1], train_x.shape[2]), activation=hidden_activation, recurrent_dropout=0.5, return_sequences=False))
    else:
        model.add(LSTM(hidden_layers, input_shape=(train_x.shape[1], train_x.shape[2]), activation=hidden_activation, recurrent_dropout=0.5, return_sequences=True))
        while hidden_depth>2:
            model.add(LSTM(hidden_layers, activation=hidden_activation, recurrent_dropout=0.5, return_sequences=True))
            hidden_depth -= 1
        model.add(LSTM(hidden_layers, activation=hidden_activation, recurrent_dropout=0.5, return_sequences=False))

    model.add(Dense(1))
    # model.add(TimeDistributed(Dense(1)))
    model.compile(loss="mse", optimizer="adam")

    model.summary()

    model_dir = '../model/' + file_name + '.hdf5'
    mc_cb = ModelCheckpoint(model_dir, save_best_only=True)
    es_cb = EarlyStopping(monitor='val_loss' ,patience=3, verbose=1)
    history = model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=100, validation_split=0.2, callbacks=[mc_cb,es_cb])

    test_y_pre = model.predict(test_x)
    test_y_pre_fix = (test_y_pre + test_y_mean) * test_y_std
    result_check = pd.DataFrame(np.concatenate([test_y_label, test_y_pre_fix.reshape(-1,1)], axis=1))
    result_check.columns = ["item_id","shop_id","正解値","予測値"]
    result_dir = "../result/test/" + file_name + ".csv"
    result_check.to_csv(result_dir, encoding='cp932')

    rmse = np.sqrt(mean_squared_error(test_y,test_y_pre))
    print('Val RMSE: %.3f' % rmse)


# hidden_depth = 1
# hidden_layers = 16
batch_size = 16
hidden_activation = "relu"
model_name = "simple_LSTM"
data_name = "simple_data"


# for hidden_depth in [2,3,4,5,6,7,8]:
#     for hidden_layers in [8,16,32,64]:
#         model_study(hidden_depth, hidden_layers, batch_size, hidden_activation, model_name, data_name)

model_study(4, 64, batch_size, hidden_activation, model_name, data_name)