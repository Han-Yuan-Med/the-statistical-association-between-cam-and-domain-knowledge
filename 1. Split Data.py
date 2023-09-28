import pandas as pd
import numpy as np

metadata = pd.read_csv("D:\\Glaucoma Dataset\\metadata - standardized.csv")
data_full = metadata[["fundus", "types", "fundus_oc_seg", "fundus_od_seg", "bv_seg"]]

# Split according to anatomical structure annotations
data_full_op = data_full[data_full["fundus_oc_seg"].notna()]
data_full_op = data_full_op.iloc[np.where(data_full_op["fundus_oc_seg"] != "Not Visible")[0]]
data_full_bv = data_full[data_full["bv_seg"].notna()]

op_case = data_full_op.iloc[np.where(data_full_op["types"] == 1)[0]]
op_control = data_full_op.iloc[np.where(data_full_op["types"] == 0)[0]]

bv_case = data_full_bv.iloc[np.where(data_full_bv["types"] == 1)[0]]
bv_control = data_full_bv.iloc[np.where(data_full_bv["types"] == 0)[0]]


def data_split(sample_num):
    np.random.seed(0)
    idx_1 = np.random.choice(range(sample_num), round(sample_num / 3), replace=False)
    idx_l = np.setdiff1d(range(sample_num), idx_1)
    np.random.seed(1)
    idx_2 = np.random.choice(idx_l, round(sample_num / 3), replace=False)
    idx_3 = np.setdiff1d(idx_l, idx_2)
    return idx_1, idx_2, idx_3


op_case_index_1, op_case_index_2, op_case_index_3 = data_split(len(op_case))
op_control_index_1, op_control_index_2, op_control_index_3 = data_split(len(op_control))

bv_case_index_1, bv_case_index_2, bv_case_index_3 = data_split(len(bv_case))
bv_control_index_1, bv_control_index_2, bv_control_index_3 = data_split(len(bv_control))

pd.concat([op_case.iloc[op_case_index_1], op_control.iloc[op_control_index_1],
           bv_case.iloc[bv_case_index_1], bv_control.iloc[bv_control_index_1]]).to_csv("data_1.csv")

pd.concat([op_case.iloc[op_case_index_2], op_control.iloc[op_control_index_2],
           bv_case.iloc[bv_case_index_2], bv_control.iloc[bv_control_index_2]]).to_csv("data_2.csv")

pd.concat([op_case.iloc[op_case_index_3], op_control.iloc[op_control_index_3],
           bv_case.iloc[bv_case_index_3], bv_control.iloc[bv_control_index_3]]).to_csv("data_3.csv")
