import sys, os
sys.path.append(os.getcwd())
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main_func(model_name, mode):
    """item_cnt_monthの折れ線グラフを出力する関数
    
    Arguments:
        model_name {String} -- 予測モデルの名前(画像を保存するフォルダを作る際に使用)
        mode {Sting} -- 出力モードがsubmissionデータなら"s"、testデータなら"t"、両方なら"st"
    """

    # model_name = "simple_data_5mfilter_functional_easy_dense1depth_LSTM_1depth_64layers_relu-activation_y_ori"

    table = pd.read_csv("../../data/feature_engineering/item_cnt_month/table_item_shop_id.csv")
    test = pd.read_csv("../../data/test.csv")
    test_table = pd.merge(test, table, how="left", on=["shop_id", "item_id"])
    test_table.fillna(0, inplace=True)

    if "s" in mode:
        submission_path = "../result/submit/"+ model_name + ".csv"
        make_submission_plot(test_table, model_name, submission_path)

    if "t" in mode:
        test_path = "../result/test/"+ model_name + ".csv"
        make_test_plot(test_table, model_name, test_path)

    

def make_submission_plot(test_table, model_name, submission_path):
    submission = pd.read_csv(submission_path)
    df_sub = pd.merge(test_table, submission, how="left", on="ID")
    df_plot = df_sub[df_sub.iloc[:,3:].sum(axis=1)>500]
    df_plot.sort_values(by="item_id", inplace=True)

    plot_path = "../result/line_graph/submit/"+model_name+"/"
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    x_axis = np.arange(0, 35, 1)
    before_id = -1
    color_id = 0
    shop_id_list = []
    for _, row in df_plot.iterrows():
        # item_cnt_month_values = df_plot.iloc[0,3:].values
        item_cnt_month_values = row[3:].values
        if before_id == row[2]:
            color_id += 1
            plt.plot(x_axis, item_cnt_month_values, label=color_id, marker="o")
            shop_id_list.append("shop"+str(int(row[1])))
        else:
            if before_id!=-1:
                plt.legend(shop_id_list, loc="upper left")
                plt.title("item_id" + str(before_id) + "_ItemCntMonth")
                plt.xlabel("date_block_num")
                plt.ylabel("item_cnt_month")
                plt.savefig(plot_path + "item_id" + str(before_id) + "_ItemCntMonth.png")
            plt.clf()
            color_id = 0
            shop_id_list = []
            before_id = int(row[2])
            plt.plot(x_axis, item_cnt_month_values, label=color_id, marker="o")
            shop_id_list.append("shop"+str(int(row[1])))
    plt.legend(shop_id_list, loc="upper left")
    plt.title("item_id" + str(before_id) + "_ItemCntMonth")
    plt.xlabel("date_block_num")
    plt.ylabel("item_cnt_month")
    plt.savefig(plot_path + "item_id" + str(before_id) + "_ItemCntMonth.png")


def make_test_plot(test_table, model_name, test_path):
    test = pd.read_csv(test_path, encoding='cp932')
    df_test = pd.merge(test_table, test, how="left", on=["shop_id", "item_id"])
    df_plot = df_test[abs(df_test["正解値"]-df_test["予測値"])>10]
    df_plot.sort_values(by="item_id", inplace=True)

    plot_path = "../result/line_graph/test/"+model_name+"/"
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    x_axis = np.arange(0, 34, 1)
    for _, row in df_plot.iterrows():
        # item_cnt_month_values = df_plot.iloc[0,3:].values
        common_month = row[3:-4].values
        right_values = np.append(common_month, row[-2])
        prediction_values = np.append(common_month, row[-1])
        prediction_plot = plt.plot(x_axis, prediction_values, label=3, marker="o")
        right_plot = plt.plot(x_axis, right_values, label=0, marker="o")

        plt.legend((prediction_plot[0], right_plot[0]), ("prediction","correct"), loc="upper left")
        plt.title("shop_id" + str(int(row[1])) + "_item_id" + str(int(row[2])) + "_ItemCntMonth")
        plt.xlabel("date_block_num")
        plt.ylabel("item_cnt_month")
        plt.savefig(plot_path + "shop_id" + str(int(row[1])) + "_item_id" + str(int(row[2])) + "_ItemCntMonth.png")
        plt.clf()

if __name__ == "__main__":
    main_func("0713113424all1 submission", "s")