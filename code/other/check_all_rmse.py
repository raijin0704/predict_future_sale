import sys, os
sys.path.append(os.getcwd())
import glob
import pandas as pd
from common.check.check_rmse import check_rmse

def check_all_rmse(saveF):
    # checked_list = pd.read_csv("../result/test/all_rmse.csv")
    model_dir__list = glob.glob("../model/*.hdf5")
    model_name_list = [i.split("\\")[-1] for i in model_dir__list]

    rmse_result = []
    for model_name in model_name_list:
        # if model_name not in checked_list["model_name"]:
        if "simple_data" in model_name:
            if "filter" in model_name:
                data_name = "simple_data_5mfilter"
            else:
                data_name = "simple_data"
        elif "basic_data" in model_name:
            if "filter" in model_name:
                data_name = "basic_data_5mfilter"
            else:
                data_name = "basic_data"
        else:
            pass
        if "y_ori" in model_name:
            rmse = check_rmse(data_name, model_name, False)
        else:
            rmse = check_rmse(data_name, model_name, True)

    
        rmse_result.append([model_name,rmse])
    
    rmse_result_df_new = pd.DataFrame(rmse_result, columns=["model_name", "rmse"])
    rmse_result_df_new.sort_values("rmse", inplace=True)
    # rmse_result_df_all = pd.concat([checked_list, rmse_result_df_new])
    # rmse_result_df_all.sort_values("rmse", inplace=True)
    if saveF:
        # rmse_result_df.to_csv("../result/test/all_rmse.csv", index=False)
        rmse_result_df_new.to_csv("../result/test/all_rmse.csv", index=False)


    # return rmse_result_df
    return rmse_result_df_new

if __name__ == "__main__":
    check_all_rmse(True)