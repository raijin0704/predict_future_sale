from keras import models
import numpy as np

def get_predict_sales(model, test_x, test_y_mean, test_y_std, y_standard_F, model_name, test_info, test_easy_info):
    if "functional" in model_name:
        if "easy" in model_name:
            test_item = test_easy_info[:,0]
            test_shop = test_easy_info[:,1]
            test_y_pre = model.predict({"series_input":test_x, "item_input":test_item, "shop_input":test_shop}, verbose=1)
        else:
            test_item = test_info[:,0]
            test_shop = test_info[:,1]
            test_item_category = test_info[:,2]
            test_y_pre = model.predict({"series_input":test_x, "item_input":test_item, "shop_input":test_shop, "item_category_input":test_item_category}, verbose=1)
    else:
        test_y_pre = model.predict(test_x, verbose=1)
    if y_standard_F:
        test_y_pre_fix = (test_y_pre + test_y_mean) * test_y_std
        # 正確な四捨五入の式
        # test_y_pre_fix = np.round((test_y_pre_fix * 2 + 1)//2).astype(int)
        test_y_pre_fix = test_y_pre_fix.clip(0, None)
    else:
        # test_y_pre_fix = np.round((test_y_pre * 2 + 1)//2).astype(int)
        # test_y_pre_fix = test_y_pre_fix.clip(0, None)

        test_y_pre_fix = test_y_pre.clip(0, None)

    return test_y_pre, test_y_pre_fix