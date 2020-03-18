
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


#%%
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, concatenate, Embedding, Flatten
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.utils import plot_model

from keras import Input, Model


#%%
import json
import pickle


from common.prepro.downcast_dtypes import downcast_dtypes
from common.prepro.get_pivot_table import get_simple_pivot_table
from common.load.load_model import load_lstm_model, load_all_numpy_data



def model_study(hidden_depth, hidden_layers, batch_size, hidden_activation, model_name, data_name, y_standard_F):

    train_x, train_y, test_x, test_y, test_y_label, test_y_mean, test_y_std, train_y_ori, test_y_ori, _, _, train_easy_info, test_easy_info = load_all_numpy_data(data_name)
    train_item = train_easy_info[:,0]
    train_shop = train_easy_info[:,1]
    # train_item_category = train_info[:,2]
    test_item = test_easy_info[:,0]
    test_shop = test_easy_info[:,1]
    # test_item_category = train_info[:,2]
    

    file_name = data_name + "_" + model_name + "_" + str(hidden_depth)+ "depth_" + str(hidden_layers) + "layers_"  + hidden_activation + "-activation"
    if y_standard_F:
        pass
    else:
        file_name = file_name + "_y_ori"

    series_input = Input(shape=(train_x.shape[1], train_x.shape[2]), dtype='float32', name="series_input")
    hidden_series = LSTM(hidden_layers, return_sequences=False)(series_input)

    item_input = Input(shape=(1,), dtype="float32", name="item_input")
    # item_embedding = Embedding(output_dim=item_dim, input_dim=train_item.shape[0], input_length=None)(item_input)
    # item_flatten = Flatten()(item_embedding)
    shop_input = Input(shape=(1,), dtype="float32", name="shop_input")
    # shop_embedding = Embedding(output_dim=shop_dim, input_dim=train_shop.shape[0], input_length=None)(shop_input)
    # shop_flatten = Flatten()(shop_embedding)
    # item_category_input = Input(shape=(1,), dtype="int32", name="item_category_input")
    # item_category_embedding = Embedding(output_dim=item_category_dim, input_dim=train_item_category.shape[0], input_length=None)(item_category_input)
    # item_category_flatten = Flatten()(item_category_embedding)

    # concatenated_info = concatenate([item_embedding, shop_embedding, item_category_embedding], axis=2)
    # concatenated_info = concatenate([item_flatten, shop_flatten, item_category_flatten], axis=-1)
    concatenated_info = concatenate([item_input, shop_input], axis=-1)

    info_dense = Dense(4)(concatenated_info)
    # concatenated_info = Input(shape=(train_info.shape[1],),dtype='float32', name="info_input")

    concatenated_all = concatenate([hidden_series, info_dense], axis=-1)

    output = Dense(1, activation=hidden_activation, name="model_output")(concatenated_all)
    # dense_all = Dense(hidden_layers,)(concatenated_all)
    # dense_all = Dropout(0.5)(dense_all)

    # output = Dense(1, activation=hidden_activation, name="model_output")(dense_all)

    # model = Model([series_input, info_input], output)
    model = Model([series_input, item_input, shop_input], output)


    model.compile(loss="mse", optimizer="adam")
    print("data_name : " + data_name)
    print("model_name : ", model_name)
    model.summary()

    model_dir = '../model/' + file_name
    plot_model(model, to_file= model_dir + '.png', show_shapes=True, show_layer_names=True)
    mc_cb = ModelCheckpoint(model_dir + ".hdf5", save_best_only=True)
    es_cb = EarlyStopping(monitor='val_loss' ,patience=7, verbose=1) # simple_dataではpatience=3だった
    if y_standard_F:
        history = model.fit({"series_input":train_x, "item_input":train_item, "shop_input":train_shop}, {"model_output":train_y}, batch_size=batch_size, epochs=100, validation_split=0.2, callbacks=[mc_cb,es_cb])
    else:
        history = model.fit({"series_input":train_x, "item_input":train_item, "shop_input":train_shop}, {"model_output":train_y_ori}, batch_size=batch_size, epochs=100, validation_split=0.2, callbacks=[mc_cb,es_cb])
    with open("../result/other/history/" + file_name + ".pickle", mode="wb") as f:
        pickle.dump(history.history, f)

    test_y_pre = model.predict({"series_input":test_x, "item_input":test_item, "shop_input":test_shop})

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
model_name = "functional_easy_dense1depth_LSTM"
data_name = "simple_price_data"
y_standard_F = False


# for hidden_depth in [1,2,3]:
#     for hidden_layers in [16,32,64]:
#         model_study(hidden_depth, hidden_layers, batch_size, hidden_activation, model_name, data_name, y_standard_F)
# for hidden_layers in [16,32,64]:
#     model_study(1, hidden_layers, batch_size, hidden_activation, model_name, data_name, y_standard_F)

# for i in range(5):
#     data_name = "simple_data"
#     if i != 0:
#         data_name += "_" + str(i) + "mfilter"
#     model_study(1, 16, batch_size, hidden_activation, model_name, data_name, y_standard_F)

#     data_name = "basic_data"
#     if i != 0:
#         data_name += "_" + str(i) + "mfilter"
#     model_study(1, 16, batch_size, hidden_activation, model_name, data_name, y_standard_F)
for hidden_layers in [8,16,32]:
    model_study(1, hidden_layers, batch_size, hidden_activation, model_name, data_name, y_standard_F)

    # data_name = "basic_data"
    # if i != 0:
    #     data_name += "_" + str(i) + "mfilter"
    # model_study(1, hidden_layers, batch_size, hidden_activation, model_name, data_name, y_standard_F)