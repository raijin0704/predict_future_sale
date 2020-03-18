import numpy as np
import pandas as pd
from keras.models import load_model

def train_val_inference(model_name, scale_mode):

    X = np.load("../data_for_kernel/all"+scale_mode+"_X.npy")
    y = np.load("../data_for_kernel/all"+scale_mode+"_y.npy")
    easy_info = np.load("../data_for_kernel/all"+scale_mode+"_EasyInfo.npy")
    model = load_model("../model/" + model_name+".hdf5")

    prediction = model.predict({
                                "series_input":X,
                                "item_input":easy_info[:,0],
                                "shop_input":easy_info[:,1]
                                }, verbose=1)

    prediction_df = pd.DataFrame(prediction, columns=['item_cnt_month'])
    prediction_df.to_csv("../result/test/"+model_name+".csv", index_label="ID")



def for_morio_inference(model_name):

    target_data = pd.read_csv("../../data/feature_engineering/valid_ID.csv")
    prediction_df = pd.read_csv("../result/test/"+model_name+".csv")
    df = pd.merge(target_data,prediction_df, how="left", on="ID")
    df["item_cnt_month"].fillna(0, inplace=True)
    df.drop(df.columns[0], axis=1, inplace=True)

    df.to_csv("../for_morimori/"+model_name+".csv", index=False)





if __name__ == "__main__":
    # train_val_inference("0714163248all113216", "Standard")
    for_morio_inference("0714163248all113216")