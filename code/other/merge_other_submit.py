import pandas as pd

raijin_csv = "simple_data_5mfilter_simple_LSTM_2depth_16layers_relu-activation_y_ori.csv"

raijin_submit = pd.read_csv("../result/submit/" + raijin_csv)
morio_submit = pd.read_csv("../../inazawa/submission.csv")

merge_df = pd.merge(morio_submit, raijin_submit, on="ID")
id = merge_df["ID"]