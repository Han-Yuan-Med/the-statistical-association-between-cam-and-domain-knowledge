import torch
from pytorch_grad_cam import GradCAM, XGradCAM, ScoreCAM, EigenCAM, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import pandas as pd
from py_functions import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn import metrics
from hausdorff import hausdorff_distance
from scipy import stats

image_path = "D:\\Glaucoma Dataset"
# Instantiating CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

dataset_id = [[1, 2, 3], [2, 1, 3], [2, 3, 1], [3, 2, 1], [1, 3, 2], [3, 1, 2]]

########################################################################################################################
# Grad-CAM
########################################################################################################################

xai_oc_list_full = []
oc_fundus_list_full = []
iou_oc_list_full = []
dice_oc_list_full = []
hau_oc_list_full = []

xai_od_list_full = []
od_fundus_list_full = []
iou_od_list_full = []
dice_od_list_full = []
hau_od_list_full = []

xai_bv_list_full = []
bv_fundus_list_full = []
iou_bv_list_full = []
dice_bv_list_full = []
hau_bv_list_full = []

results_df = []


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


for id in range(len(dataset_id)):
    test_csv = pd.read_csv(f"data_{dataset_id[id][2]}.csv")

    test_op = Fundus_seg_2(csv_file=test_csv[test_csv["fundus_oc_seg"].notna()],
                           img_dir=image_path, label_id_1=3, label_id_2=4)
    test_bv = Fundus_seg_1(csv_file=test_csv[test_csv["bv_seg"].notna()],
                           img_dir=image_path, label_id=5)

    test_op_loader = DataLoader(test_op, batch_size=1, shuffle=False)
    test_bv_loader = DataLoader(test_bv, batch_size=1, shuffle=False)

    DeiTT_optimal = torch.load(f"DeiTT optimal {id}.pt")
    target_layers = [DeiTT_optimal.blocks[-1].norm1]
    targets = [ClassifierOutputTarget(1)]
    cam = GradCAM(model=DeiTT_optimal, target_layers=target_layers,
                  reshape_transform=reshape_transform, use_cuda=True)

    iou_oc_list = []
    dice_oc_list = []
    hau_oc_list = []
    xai_oc_list = []
    oc_fundus_list = []

    iou_od_list = []
    dice_od_list = []
    hau_od_list = []
    xai_od_list = []
    od_fundus_list = []

    # Test optic disk and optic cup
    for data_tmp in tqdm(test_op_loader):
        images_tmp, label_oc_tmp, label_od_tmp = data_tmp[0].float().to(device), \
                                                 data_tmp[1].numpy().flatten().astype("uint8"), \
                                                 data_tmp[2].numpy().flatten().astype("uint8")
        xai_tmp = cam(input_tensor=images_tmp, targets=targets).flatten()
        threshold_value = np.percentile(xai_tmp, 95)
        xai_tmp[np.where(xai_tmp >= threshold_value)] = 1
        xai_tmp[np.where(xai_tmp != 1)] = 0
        xai_tmp = xai_tmp.astype("uint8")

        xai_oc_tmp = (xai_tmp * label_oc_tmp).sum() / xai_tmp.sum()
        oc_fundus_tmp = label_oc_tmp.sum() / 50176
        iou_oc = format(metrics.jaccard_score(label_oc_tmp, xai_tmp), '.3f')
        dice_oc = format(metrics.f1_score(label_oc_tmp, xai_tmp), '.3f')
        hau_oc = format(hausdorff_distance(np.resize(xai_tmp, (224, 224)), np.resize(label_oc_tmp, (224, 224))), '.3f')

        xai_oc_list.append(xai_oc_tmp)
        oc_fundus_list.append(oc_fundus_tmp)
        iou_oc_list.append(iou_oc)
        dice_oc_list.append(dice_oc)
        hau_oc_list.append(hau_oc)

        xai_od_tmp = (xai_tmp * label_od_tmp).sum() / xai_tmp.sum()
        od_fundus_tmp = label_od_tmp.sum() / 50176
        iou_od = format(metrics.jaccard_score(label_od_tmp, xai_tmp), '.3f')
        dice_od = format(metrics.f1_score(label_od_tmp, xai_tmp), '.3f')
        hau_od = format(hausdorff_distance(np.resize(xai_tmp, (224, 224)), np.resize(label_od_tmp, (224, 224))), '.3f')

        xai_od_list.append(xai_od_tmp)
        od_fundus_list.append(od_fundus_tmp)
        iou_od_list.append(iou_od)
        dice_od_list.append(dice_od)
        hau_od_list.append(hau_od)

    xai_oc_mean, xai_oc_std = bootstrap_sample(value_list=np.array(xai_oc_list), times=100)
    oc_fundus_mean, oc_fundus_std = bootstrap_sample(value_list=np.array(oc_fundus_list), times=100)
    oc_difference_mean, oc_difference_std = bootstrap_sample(value_list=np.array(xai_oc_list) -
                                                                        np.array(oc_fundus_list), times=100)
    iou_oc_mean, iou_oc_std = bootstrap_sample(value_list=np.array(iou_oc_list), times=100)
    dice_oc_mean, dice_oc_std = bootstrap_sample(value_list=np.array(dice_oc_list), times=100)
    hau_oc_mean, hau_oc_std = bootstrap_sample(value_list=np.array(hau_oc_list), times=100)
    _, p_value = stats.ttest_rel(xai_oc_list, oc_fundus_list)

    results_df.append([f"DeiT-T", f"{id}", f"Grad-CAM", f"Optic cup",
                       f"{xai_oc_mean} ({xai_oc_std})", f"{oc_fundus_mean} ({oc_fundus_std})",
                       f"{oc_difference_mean} ({oc_difference_std})", f"{format(p_value, '.3e')}",
                       f"{iou_oc_mean} ({iou_oc_std})", f"{dice_oc_mean} ({dice_oc_std})",
                       f"{hau_oc_mean} ({hau_oc_std})"])

    xai_oc_list_full.append(xai_oc_list)
    oc_fundus_list_full.append(oc_fundus_list)
    iou_oc_list_full.append(iou_oc_list)
    dice_oc_list_full.append(dice_oc_list)
    hau_oc_list_full.append(hau_oc_list)

    xai_od_mean, xai_od_std = bootstrap_sample(value_list=np.array(xai_od_list), times=100)
    od_fundus_mean, od_fundus_std = bootstrap_sample(value_list=np.array(od_fundus_list), times=100)
    od_difference_mean, od_difference_std = bootstrap_sample(value_list=np.array(xai_od_list) -
                                                                        np.array(od_fundus_list), times=100)
    iou_od_mean, iou_od_std = bootstrap_sample(value_list=np.array(iou_od_list), times=100)
    dice_od_mean, dice_od_std = bootstrap_sample(value_list=np.array(dice_od_list), times=100)
    hau_od_mean, hau_od_std = bootstrap_sample(value_list=np.array(hau_od_list), times=100)
    _, p_value = stats.ttest_rel(xai_od_list, od_fundus_list)

    results_df.append([f"DeiT-T", f"{id}", f"Grad-CAM", f"Optic disk",
                       f"{xai_od_mean} ({xai_od_std})",
                       f"{od_fundus_mean} ({od_fundus_std})",
                       f"{od_difference_mean} ({od_difference_std})", f"{format(p_value, '.3e')}",
                       f"{iou_od_mean} ({iou_od_std})", f"{dice_od_mean} ({dice_od_std})",
                       f"{hau_od_mean} ({hau_od_std})"])

    xai_od_list_full.append(xai_od_list)
    od_fundus_list_full.append(od_fundus_list)
    iou_od_list_full.append(iou_od_list)
    dice_od_list_full.append(dice_od_list)
    hau_od_list_full.append(hau_od_list)

    iou_bv_list = []
    dice_bv_list = []
    hau_bv_list = []
    xai_bv_list = []
    bv_fundus_list = []

    # Test optic disk and optic cup
    for data_tmp in tqdm(test_bv_loader):
        images_tmp, label_bv_tmp = data_tmp[0].float().to(device), \
                                   data_tmp[1].numpy().flatten().astype("uint8")
        xai_tmp = cam(input_tensor=images_tmp, targets=targets).flatten()
        threshold_value = np.percentile(xai_tmp, 95)
        xai_tmp[np.where(xai_tmp >= threshold_value)] = 1
        xai_tmp[np.where(xai_tmp != 1)] = 0
        xai_tmp = xai_tmp.astype("uint8")

        xai_bv_tmp = (xai_tmp * label_bv_tmp).sum() / xai_tmp.sum()
        bv_fundus_tmp = label_bv_tmp.sum() / 50176
        iou_bv = format(metrics.jaccard_score(label_bv_tmp, xai_tmp), '.3f')
        dice_bv = format(metrics.f1_score(label_bv_tmp, xai_tmp), '.3f')
        hau_bv = format(hausdorff_distance(np.resize(xai_tmp, (224, 224)), np.resize(label_bv_tmp, (224, 224))), '.3f')

        xai_bv_list.append(xai_bv_tmp)
        bv_fundus_list.append(bv_fundus_tmp)
        iou_bv_list.append(iou_bv)
        dice_bv_list.append(dice_bv)
        hau_bv_list.append(hau_bv)

    xai_bv_mean, xai_bv_std = bootstrap_sample(value_list=np.array(xai_bv_list), times=100)
    bv_fundus_mean, bv_fundus_std = bootstrap_sample(value_list=np.array(bv_fundus_list), times=100)
    bv_difference_mean, bv_difference_std = bootstrap_sample(value_list=np.array(xai_bv_list) -
                                                                        np.array(bv_fundus_list), times=100)
    iou_bv_mean, iou_bv_std = bootstrap_sample(value_list=np.array(iou_bv_list), times=100)
    dice_bv_mean, dice_bv_std = bootstrap_sample(value_list=np.array(dice_bv_list), times=100)
    hau_bv_mean, hau_bv_std = bootstrap_sample(value_list=np.array(hau_bv_list), times=100)
    _, p_value = stats.ttest_rel(xai_bv_list, bv_fundus_list)

    results_df.append([f"DeiT-T", f"{id}", f"Grad-CAM", f"Blood vessels",
                       f"{xai_bv_mean} ({xai_bv_std})",
                       f"{bv_fundus_mean} ({bv_fundus_std})",
                       f"{bv_difference_mean} ({bv_difference_std})", f"{format(p_value, '.3e')}",
                       f"{iou_bv_mean} ({iou_bv_std})", f"{dice_bv_mean} ({dice_bv_std})",
                       f"{hau_bv_mean} ({hau_bv_std})"])

    xai_bv_list_full.append(xai_bv_list)
    bv_fundus_list_full.append(bv_fundus_list)
    iou_bv_list_full.append(iou_bv_list)
    dice_bv_list_full.append(dice_bv_list)
    hau_bv_list_full.append(hau_bv_list)

pd.DataFrame(results_df).to_csv("explanation_deit_t_grad_cam_cross_validation.csv", index=False, encoding="cp1252")

results_df_full = []

xai_oc_mean, xai_oc_std = bootstrap_sample(value_list=np.array(sum(xai_oc_list_full, [])), times=100)
oc_fundus_mean, oc_fundus_std = bootstrap_sample(value_list=np.array(sum(oc_fundus_list_full, [])), times=100)
oc_difference_mean, oc_difference_std = bootstrap_sample(value_list=np.array(sum(xai_oc_list_full, [])) -
                                                                    np.array(sum(oc_fundus_list_full, [])), times=100)
iou_oc_mean, iou_oc_std = bootstrap_sample(value_list=np.array(sum(iou_oc_list_full, [])), times=100)
dice_oc_mean, dice_oc_std = bootstrap_sample(value_list=np.array(sum(dice_oc_list_full, [])), times=100)
hau_oc_mean, hau_oc_std = bootstrap_sample(value_list=np.array(sum(hau_oc_list_full, [])), times=100)
_, p_value = stats.ttest_rel(sum(xai_oc_list_full, []), sum(oc_fundus_list_full, []))

results_df_full.append([f"DeiT-T", f"Grad-CAM", f"Optic cup",
                        f"{xai_oc_mean} ({xai_oc_std})", f"{oc_fundus_mean} ({oc_fundus_std})",
                        f"{oc_difference_mean} ({oc_difference_std})", f"{format(p_value, '.3e')}",
                        f"{iou_oc_mean} ({iou_oc_std})", f"{dice_oc_mean} ({dice_oc_std})",
                        f"{hau_oc_mean} ({hau_oc_std})"])

xai_od_mean, xai_od_std = bootstrap_sample(value_list=np.array(sum(xai_od_list_full, [])), times=100)
od_fundus_mean, od_fundus_std = bootstrap_sample(value_list=np.array(sum(od_fundus_list_full, [])), times=100)
od_difference_mean, od_difference_std = bootstrap_sample(value_list=np.array(sum(xai_od_list_full, [])) -
                                                                    np.array(sum(od_fundus_list_full, [])), times=100)
iou_od_mean, iou_od_std = bootstrap_sample(value_list=np.array(sum(iou_od_list_full, [])), times=100)
dice_od_mean, dice_od_std = bootstrap_sample(value_list=np.array(sum(dice_od_list_full, [])), times=100)
hau_od_mean, hau_od_std = bootstrap_sample(value_list=np.array(sum(hau_od_list_full, [])), times=100)
_, p_value = stats.ttest_rel(sum(xai_od_list_full, []), sum(od_fundus_list_full, []))

results_df_full.append([f"DeiT-T", f"Grad-CAM", f"Optic disk",
                        f"{xai_od_mean} ({xai_od_std})",
                        f"{od_fundus_mean} ({od_fundus_std})",
                        f"{od_difference_mean} ({od_difference_std})", f"{format(p_value, '.3e')}",
                        f"{iou_od_mean} ({iou_od_std})", f"{dice_od_mean} ({dice_od_std})",
                        f"{hau_od_mean} ({hau_od_std})"])

xai_bv_mean, xai_bv_std = bootstrap_sample(value_list=np.array(sum(xai_bv_list_full, [])), times=100)
bv_fundus_mean, bv_fundus_std = bootstrap_sample(value_list=np.array(sum(bv_fundus_list_full, [])), times=100)
bv_difference_mean, bv_difference_std = bootstrap_sample(value_list=np.array(sum(xai_bv_list_full, [])) -
                                                                    np.array(sum(bv_fundus_list_full, [])), times=100)
iou_bv_mean, iou_bv_std = bootstrap_sample(value_list=np.array(sum(iou_bv_list_full, [])), times=100)
dice_bv_mean, dice_bv_std = bootstrap_sample(value_list=np.array(sum(dice_bv_list_full, [])), times=100)
hau_bv_mean, hau_bv_std = bootstrap_sample(value_list=np.array(sum(hau_bv_list_full, [])), times=100)
_, p_value = stats.ttest_rel(sum(xai_bv_list_full, []), sum(bv_fundus_list_full, []))

results_df_full.append([f"DeiT-T", f"Grad-CAM", f"Blood vessels",
                        f"{xai_bv_mean} ({xai_bv_std})",
                        f"{bv_fundus_mean} ({bv_fundus_std})",
                        f"{bv_difference_mean} ({bv_difference_std})", f"{format(p_value, '.3e')}",
                        f"{iou_bv_mean} ({iou_bv_std})", f"{dice_bv_mean} ({dice_bv_std})",
                        f"{hau_bv_mean} ({hau_bv_std})"])

pd.DataFrame(results_df_full).to_csv("explanation_deit_t_grad_cam_aggregation.csv", index=False, encoding="cp1252")

########################################################################################################################
# XGrad-CAM
########################################################################################################################

xai_oc_list_full = []
oc_fundus_list_full = []
iou_oc_list_full = []
dice_oc_list_full = []
hau_oc_list_full = []

xai_od_list_full = []
od_fundus_list_full = []
iou_od_list_full = []
dice_od_list_full = []
hau_od_list_full = []

xai_bv_list_full = []
bv_fundus_list_full = []
iou_bv_list_full = []
dice_bv_list_full = []
hau_bv_list_full = []

results_df = []

for id in range(len(dataset_id)):
    test_csv = pd.read_csv(f"data_{dataset_id[id][2]}.csv")

    test_op = Fundus_seg_2(csv_file=test_csv[test_csv["fundus_oc_seg"].notna()],
                           img_dir=image_path, label_id_1=3, label_id_2=4)
    test_bv = Fundus_seg_1(csv_file=test_csv[test_csv["bv_seg"].notna()],
                           img_dir=image_path, label_id=5)

    test_op_loader = DataLoader(test_op, batch_size=1, shuffle=False)
    test_bv_loader = DataLoader(test_bv, batch_size=1, shuffle=False)

    DeiTT_optimal = torch.load(f"DeiTT optimal {id}.pt")
    target_layers = [DeiTT_optimal.blocks[-1].norm1]
    targets = [ClassifierOutputTarget(1)]
    cam = XGradCAM(model=DeiTT_optimal, target_layers=target_layers,
                   reshape_transform=reshape_transform, use_cuda=True)

    iou_oc_list = []
    dice_oc_list = []
    hau_oc_list = []
    xai_oc_list = []
    oc_fundus_list = []

    iou_od_list = []
    dice_od_list = []
    hau_od_list = []
    xai_od_list = []
    od_fundus_list = []

    # Test optic disk and optic cup
    for data_tmp in tqdm(test_op_loader):
        images_tmp, label_oc_tmp, label_od_tmp = data_tmp[0].float().to(device), \
                                                 data_tmp[1].numpy().flatten().astype("uint8"), \
                                                 data_tmp[2].numpy().flatten().astype("uint8")
        xai_tmp = cam(input_tensor=images_tmp, targets=targets).flatten()
        threshold_value = np.percentile(xai_tmp, 95)
        xai_tmp[np.where(xai_tmp >= threshold_value)] = 1
        xai_tmp[np.where(xai_tmp != 1)] = 0
        xai_tmp = xai_tmp.astype("uint8")

        xai_oc_tmp = (xai_tmp * label_oc_tmp).sum() / xai_tmp.sum()
        oc_fundus_tmp = label_oc_tmp.sum() / 50176
        iou_oc = format(metrics.jaccard_score(label_oc_tmp, xai_tmp), '.3f')
        dice_oc = format(metrics.f1_score(label_oc_tmp, xai_tmp), '.3f')
        hau_oc = format(hausdorff_distance(np.resize(xai_tmp, (224, 224)), np.resize(label_oc_tmp, (224, 224))), '.3f')

        xai_oc_list.append(xai_oc_tmp)
        oc_fundus_list.append(oc_fundus_tmp)
        iou_oc_list.append(iou_oc)
        dice_oc_list.append(dice_oc)
        hau_oc_list.append(hau_oc)

        xai_od_tmp = (xai_tmp * label_od_tmp).sum() / xai_tmp.sum()
        od_fundus_tmp = label_od_tmp.sum() / 50176
        iou_od = format(metrics.jaccard_score(label_od_tmp, xai_tmp), '.3f')
        dice_od = format(metrics.f1_score(label_od_tmp, xai_tmp), '.3f')
        hau_od = format(hausdorff_distance(np.resize(xai_tmp, (224, 224)), np.resize(label_od_tmp, (224, 224))), '.3f')

        xai_od_list.append(xai_od_tmp)
        od_fundus_list.append(od_fundus_tmp)
        iou_od_list.append(iou_od)
        dice_od_list.append(dice_od)
        hau_od_list.append(hau_od)

    xai_oc_mean, xai_oc_std = bootstrap_sample(value_list=np.array(xai_oc_list), times=100)
    oc_fundus_mean, oc_fundus_std = bootstrap_sample(value_list=np.array(oc_fundus_list), times=100)
    oc_difference_mean, oc_difference_std = bootstrap_sample(value_list=np.array(xai_oc_list) -
                                                                        np.array(oc_fundus_list), times=100)
    iou_oc_mean, iou_oc_std = bootstrap_sample(value_list=np.array(iou_oc_list), times=100)
    dice_oc_mean, dice_oc_std = bootstrap_sample(value_list=np.array(dice_oc_list), times=100)
    hau_oc_mean, hau_oc_std = bootstrap_sample(value_list=np.array(hau_oc_list), times=100)
    _, p_value = stats.ttest_rel(xai_oc_list, oc_fundus_list)

    results_df.append([f"DeiT-T", f"{id}", f"XGrad-CAM", f"Optic cup",
                       f"{xai_oc_mean} ({xai_oc_std})", f"{oc_fundus_mean} ({oc_fundus_std})",
                       f"{oc_difference_mean} ({oc_difference_std})", f"{format(p_value, '.3e')}",
                       f"{iou_oc_mean} ({iou_oc_std})", f"{dice_oc_mean} ({dice_oc_std})",
                       f"{hau_oc_mean} ({hau_oc_std})"])

    xai_oc_list_full.append(xai_oc_list)
    oc_fundus_list_full.append(oc_fundus_list)
    iou_oc_list_full.append(iou_oc_list)
    dice_oc_list_full.append(dice_oc_list)
    hau_oc_list_full.append(hau_oc_list)

    xai_od_mean, xai_od_std = bootstrap_sample(value_list=np.array(xai_od_list), times=100)
    od_fundus_mean, od_fundus_std = bootstrap_sample(value_list=np.array(od_fundus_list), times=100)
    od_difference_mean, od_difference_std = bootstrap_sample(value_list=np.array(xai_od_list) -
                                                                        np.array(od_fundus_list), times=100)
    iou_od_mean, iou_od_std = bootstrap_sample(value_list=np.array(iou_od_list), times=100)
    dice_od_mean, dice_od_std = bootstrap_sample(value_list=np.array(dice_od_list), times=100)
    hau_od_mean, hau_od_std = bootstrap_sample(value_list=np.array(hau_od_list), times=100)
    _, p_value = stats.ttest_rel(xai_od_list, od_fundus_list)

    results_df.append([f"DeiT-T", f"{id}", f"XGrad-CAM", f"Optic disk",
                       f"{xai_od_mean} ({xai_od_std})",
                       f"{od_fundus_mean} ({od_fundus_std})",
                       f"{od_difference_mean} ({od_difference_std})", f"{format(p_value, '.3e')}",
                       f"{iou_od_mean} ({iou_od_std})", f"{dice_od_mean} ({dice_od_std})",
                       f"{hau_od_mean} ({hau_od_std})"])

    xai_od_list_full.append(xai_od_list)
    od_fundus_list_full.append(od_fundus_list)
    iou_od_list_full.append(iou_od_list)
    dice_od_list_full.append(dice_od_list)
    hau_od_list_full.append(hau_od_list)

    iou_bv_list = []
    dice_bv_list = []
    hau_bv_list = []
    xai_bv_list = []
    bv_fundus_list = []

    # Test optic disk and optic cup
    for data_tmp in tqdm(test_bv_loader):
        images_tmp, label_bv_tmp = data_tmp[0].float().to(device), \
                                   data_tmp[1].numpy().flatten().astype("uint8")
        xai_tmp = cam(input_tensor=images_tmp, targets=targets).flatten()
        threshold_value = np.percentile(xai_tmp, 95)
        xai_tmp[np.where(xai_tmp >= threshold_value)] = 1
        xai_tmp[np.where(xai_tmp != 1)] = 0
        xai_tmp = xai_tmp.astype("uint8")

        xai_bv_tmp = (xai_tmp * label_bv_tmp).sum() / xai_tmp.sum()
        bv_fundus_tmp = label_bv_tmp.sum() / 50176
        iou_bv = format(metrics.jaccard_score(label_bv_tmp, xai_tmp), '.3f')
        dice_bv = format(metrics.f1_score(label_bv_tmp, xai_tmp), '.3f')
        hau_bv = format(hausdorff_distance(np.resize(xai_tmp, (224, 224)), np.resize(label_bv_tmp, (224, 224))), '.3f')

        xai_bv_list.append(xai_bv_tmp)
        bv_fundus_list.append(bv_fundus_tmp)
        iou_bv_list.append(iou_bv)
        dice_bv_list.append(dice_bv)
        hau_bv_list.append(hau_bv)

    xai_bv_mean, xai_bv_std = bootstrap_sample(value_list=np.array(xai_bv_list), times=100)
    bv_fundus_mean, bv_fundus_std = bootstrap_sample(value_list=np.array(bv_fundus_list), times=100)
    bv_difference_mean, bv_difference_std = bootstrap_sample(value_list=np.array(xai_bv_list) -
                                                                        np.array(bv_fundus_list), times=100)
    iou_bv_mean, iou_bv_std = bootstrap_sample(value_list=np.array(iou_bv_list), times=100)
    dice_bv_mean, dice_bv_std = bootstrap_sample(value_list=np.array(dice_bv_list), times=100)
    hau_bv_mean, hau_bv_std = bootstrap_sample(value_list=np.array(hau_bv_list), times=100)
    _, p_value = stats.ttest_rel(xai_bv_list, bv_fundus_list)

    results_df.append([f"DeiT-T", f"{id}", f"XGrad-CAM", f"Blood vessels",
                       f"{xai_bv_mean} ({xai_bv_std})",
                       f"{bv_fundus_mean} ({bv_fundus_std})",
                       f"{bv_difference_mean} ({bv_difference_std})", f"{format(p_value, '.3e')}",
                       f"{iou_bv_mean} ({iou_bv_std})", f"{dice_bv_mean} ({dice_bv_std})",
                       f"{hau_bv_mean} ({hau_bv_std})"])

    xai_bv_list_full.append(xai_bv_list)
    bv_fundus_list_full.append(bv_fundus_list)
    iou_bv_list_full.append(iou_bv_list)
    dice_bv_list_full.append(dice_bv_list)
    hau_bv_list_full.append(hau_bv_list)

pd.DataFrame(results_df).to_csv("explanation_deit_t_XGrad_cam_cross_validation.csv", index=False, encoding="cp1252")

results_df_full = []

xai_oc_mean, xai_oc_std = bootstrap_sample(value_list=np.array(sum(xai_oc_list_full, [])), times=100)
oc_fundus_mean, oc_fundus_std = bootstrap_sample(value_list=np.array(sum(oc_fundus_list_full, [])), times=100)
oc_difference_mean, oc_difference_std = bootstrap_sample(value_list=np.array(sum(xai_oc_list_full, [])) -
                                                                    np.array(sum(oc_fundus_list_full, [])), times=100)
iou_oc_mean, iou_oc_std = bootstrap_sample(value_list=np.array(sum(iou_oc_list_full, [])), times=100)
dice_oc_mean, dice_oc_std = bootstrap_sample(value_list=np.array(sum(dice_oc_list_full, [])), times=100)
hau_oc_mean, hau_oc_std = bootstrap_sample(value_list=np.array(sum(hau_oc_list_full, [])), times=100)
_, p_value = stats.ttest_rel(sum(xai_oc_list_full, []), sum(oc_fundus_list_full, []))

results_df_full.append([f"DeiT-T", f"XGrad-CAM", f"Optic cup",
                        f"{xai_oc_mean} ({xai_oc_std})", f"{oc_fundus_mean} ({oc_fundus_std})",
                        f"{oc_difference_mean} ({oc_difference_std})", f"{format(p_value, '.3e')}",
                        f"{iou_oc_mean} ({iou_oc_std})", f"{dice_oc_mean} ({dice_oc_std})",
                        f"{hau_oc_mean} ({hau_oc_std})"])

xai_od_mean, xai_od_std = bootstrap_sample(value_list=np.array(sum(xai_od_list_full, [])), times=100)
od_fundus_mean, od_fundus_std = bootstrap_sample(value_list=np.array(sum(od_fundus_list_full, [])), times=100)
od_difference_mean, od_difference_std = bootstrap_sample(value_list=np.array(sum(xai_od_list_full, [])) -
                                                                    np.array(sum(od_fundus_list_full, [])), times=100)
iou_od_mean, iou_od_std = bootstrap_sample(value_list=np.array(sum(iou_od_list_full, [])), times=100)
dice_od_mean, dice_od_std = bootstrap_sample(value_list=np.array(sum(dice_od_list_full, [])), times=100)
hau_od_mean, hau_od_std = bootstrap_sample(value_list=np.array(sum(hau_od_list_full, [])), times=100)
_, p_value = stats.ttest_rel(sum(xai_od_list_full, []), sum(od_fundus_list_full, []))

results_df_full.append([f"DeiT-T", f"XGrad-CAM", f"Optic disk",
                        f"{xai_od_mean} ({xai_od_std})",
                        f"{od_fundus_mean} ({od_fundus_std})",
                        f"{od_difference_mean} ({od_difference_std})", f"{format(p_value, '.3e')}",
                        f"{iou_od_mean} ({iou_od_std})", f"{dice_od_mean} ({dice_od_std})",
                        f"{hau_od_mean} ({hau_od_std})"])

xai_bv_mean, xai_bv_std = bootstrap_sample(value_list=np.array(sum(xai_bv_list_full, [])), times=100)
bv_fundus_mean, bv_fundus_std = bootstrap_sample(value_list=np.array(sum(bv_fundus_list_full, [])), times=100)
bv_difference_mean, bv_difference_std = bootstrap_sample(value_list=np.array(sum(xai_bv_list_full, [])) -
                                                                    np.array(sum(bv_fundus_list_full, [])), times=100)
iou_bv_mean, iou_bv_std = bootstrap_sample(value_list=np.array(sum(iou_bv_list_full, [])), times=100)
dice_bv_mean, dice_bv_std = bootstrap_sample(value_list=np.array(sum(dice_bv_list_full, [])), times=100)
hau_bv_mean, hau_bv_std = bootstrap_sample(value_list=np.array(sum(hau_bv_list_full, [])), times=100)
_, p_value = stats.ttest_rel(sum(xai_bv_list_full, []), sum(bv_fundus_list_full, []))

results_df_full.append([f"DeiT-T", f"XGrad-CAM", f"Blood vessels",
                        f"{xai_bv_mean} ({xai_bv_std})",
                        f"{bv_fundus_mean} ({bv_fundus_std})",
                        f"{bv_difference_mean} ({bv_difference_std})", f"{format(p_value, '.3e')}",
                        f"{iou_bv_mean} ({iou_bv_std})", f"{dice_bv_mean} ({dice_bv_std})",
                        f"{hau_bv_mean} ({hau_bv_std})"])

pd.DataFrame(results_df_full).to_csv("explanation_deit_t_XGrad_cam_aggregation.csv", index=False, encoding="cp1252")

########################################################################################################################
# Score-CAM
########################################################################################################################
xai_oc_list_full = []
oc_fundus_list_full = []
iou_oc_list_full = []
dice_oc_list_full = []
hau_oc_list_full = []

xai_od_list_full = []
od_fundus_list_full = []
iou_od_list_full = []
dice_od_list_full = []
hau_od_list_full = []

xai_bv_list_full = []
bv_fundus_list_full = []
iou_bv_list_full = []
dice_bv_list_full = []
hau_bv_list_full = []

results_df = []

for id in range(len(dataset_id)):
    test_csv = pd.read_csv(f"data_{dataset_id[id][2]}.csv")

    test_op = Fundus_seg_2(csv_file=test_csv[test_csv["fundus_oc_seg"].notna()],
                           img_dir=image_path, label_id_1=3, label_id_2=4)
    test_bv = Fundus_seg_1(csv_file=test_csv[test_csv["bv_seg"].notna()],
                           img_dir=image_path, label_id=5)

    test_op_loader = DataLoader(test_op, batch_size=1, shuffle=False)
    test_bv_loader = DataLoader(test_bv, batch_size=1, shuffle=False)

    DeiTT_optimal = torch.load(f"DeiTT optimal {id}.pt")
    target_layers = [DeiTT_optimal.blocks[-1].norm1]
    targets = [ClassifierOutputTarget(1)]
    cam = ScoreCAM(model=DeiTT_optimal, target_layers=target_layers,
                   reshape_transform=reshape_transform, use_cuda=True)

    iou_oc_list = []
    dice_oc_list = []
    hau_oc_list = []
    xai_oc_list = []
    oc_fundus_list = []

    iou_od_list = []
    dice_od_list = []
    hau_od_list = []
    xai_od_list = []
    od_fundus_list = []

    # Test optic disk and optic cup
    for data_tmp in tqdm(test_op_loader):
        images_tmp, label_oc_tmp, label_od_tmp = data_tmp[0].float().to(device), \
                                                 data_tmp[1].numpy().flatten().astype("uint8"), \
                                                 data_tmp[2].numpy().flatten().astype("uint8")
        xai_tmp = cam(input_tensor=images_tmp, targets=targets).flatten()
        threshold_value = np.percentile(xai_tmp, 95)
        xai_tmp[np.where(xai_tmp >= threshold_value)] = 1
        xai_tmp[np.where(xai_tmp != 1)] = 0
        xai_tmp = xai_tmp.astype("uint8")

        xai_oc_tmp = (xai_tmp * label_oc_tmp).sum() / xai_tmp.sum()
        oc_fundus_tmp = label_oc_tmp.sum() / 50176
        iou_oc = format(metrics.jaccard_score(label_oc_tmp, xai_tmp), '.3f')
        dice_oc = format(metrics.f1_score(label_oc_tmp, xai_tmp), '.3f')
        hau_oc = format(hausdorff_distance(np.resize(xai_tmp, (224, 224)), np.resize(label_oc_tmp, (224, 224))), '.3f')

        xai_oc_list.append(xai_oc_tmp)
        oc_fundus_list.append(oc_fundus_tmp)
        iou_oc_list.append(iou_oc)
        dice_oc_list.append(dice_oc)
        hau_oc_list.append(hau_oc)

        xai_od_tmp = (xai_tmp * label_od_tmp).sum() / xai_tmp.sum()
        od_fundus_tmp = label_od_tmp.sum() / 50176
        iou_od = format(metrics.jaccard_score(label_od_tmp, xai_tmp), '.3f')
        dice_od = format(metrics.f1_score(label_od_tmp, xai_tmp), '.3f')
        hau_od = format(hausdorff_distance(np.resize(xai_tmp, (224, 224)), np.resize(label_od_tmp, (224, 224))), '.3f')

        xai_od_list.append(xai_od_tmp)
        od_fundus_list.append(od_fundus_tmp)
        iou_od_list.append(iou_od)
        dice_od_list.append(dice_od)
        hau_od_list.append(hau_od)

    xai_oc_mean, xai_oc_std = bootstrap_sample(value_list=np.array(xai_oc_list), times=100)
    oc_fundus_mean, oc_fundus_std = bootstrap_sample(value_list=np.array(oc_fundus_list), times=100)
    oc_difference_mean, oc_difference_std = bootstrap_sample(value_list=np.array(xai_oc_list) -
                                                                        np.array(oc_fundus_list), times=100)
    iou_oc_mean, iou_oc_std = bootstrap_sample(value_list=np.array(iou_oc_list), times=100)
    dice_oc_mean, dice_oc_std = bootstrap_sample(value_list=np.array(dice_oc_list), times=100)
    hau_oc_mean, hau_oc_std = bootstrap_sample(value_list=np.array(hau_oc_list), times=100)
    _, p_value = stats.ttest_rel(xai_oc_list, oc_fundus_list)

    results_df.append([f"DeiT-T", f"{id}", f"Score-CAM", f"Optic cup",
                       f"{xai_oc_mean} ({xai_oc_std})", f"{oc_fundus_mean} ({oc_fundus_std})",
                       f"{oc_difference_mean} ({oc_difference_std})", f"{format(p_value, '.3e')}",
                       f"{iou_oc_mean} ({iou_oc_std})", f"{dice_oc_mean} ({dice_oc_std})",
                       f"{hau_oc_mean} ({hau_oc_std})"])

    xai_oc_list_full.append(xai_oc_list)
    oc_fundus_list_full.append(oc_fundus_list)
    iou_oc_list_full.append(iou_oc_list)
    dice_oc_list_full.append(dice_oc_list)
    hau_oc_list_full.append(hau_oc_list)

    xai_od_mean, xai_od_std = bootstrap_sample(value_list=np.array(xai_od_list), times=100)
    od_fundus_mean, od_fundus_std = bootstrap_sample(value_list=np.array(od_fundus_list), times=100)
    od_difference_mean, od_difference_std = bootstrap_sample(value_list=np.array(xai_od_list) -
                                                                        np.array(od_fundus_list), times=100)
    iou_od_mean, iou_od_std = bootstrap_sample(value_list=np.array(iou_od_list), times=100)
    dice_od_mean, dice_od_std = bootstrap_sample(value_list=np.array(dice_od_list), times=100)
    hau_od_mean, hau_od_std = bootstrap_sample(value_list=np.array(hau_od_list), times=100)
    _, p_value = stats.ttest_rel(xai_od_list, od_fundus_list)

    results_df.append([f"DeiT-T", f"{id}", f"Score-CAM", f"Optic disk",
                       f"{xai_od_mean} ({xai_od_std})",
                       f"{od_fundus_mean} ({od_fundus_std})",
                       f"{od_difference_mean} ({od_difference_std})", f"{format(p_value, '.3e')}",
                       f"{iou_od_mean} ({iou_od_std})", f"{dice_od_mean} ({dice_od_std})",
                       f"{hau_od_mean} ({hau_od_std})"])

    xai_od_list_full.append(xai_od_list)
    od_fundus_list_full.append(od_fundus_list)
    iou_od_list_full.append(iou_od_list)
    dice_od_list_full.append(dice_od_list)
    hau_od_list_full.append(hau_od_list)

    iou_bv_list = []
    dice_bv_list = []
    hau_bv_list = []
    xai_bv_list = []
    bv_fundus_list = []

    # Test optic disk and optic cup
    for data_tmp in tqdm(test_bv_loader):
        images_tmp, label_bv_tmp = data_tmp[0].float().to(device), \
                                   data_tmp[1].numpy().flatten().astype("uint8")
        xai_tmp = cam(input_tensor=images_tmp, targets=targets).flatten()
        threshold_value = np.percentile(xai_tmp, 95)
        xai_tmp[np.where(xai_tmp >= threshold_value)] = 1
        xai_tmp[np.where(xai_tmp != 1)] = 0
        xai_tmp = xai_tmp.astype("uint8")

        xai_bv_tmp = (xai_tmp * label_bv_tmp).sum() / xai_tmp.sum()
        bv_fundus_tmp = label_bv_tmp.sum() / 50176
        iou_bv = format(metrics.jaccard_score(label_bv_tmp, xai_tmp), '.3f')
        dice_bv = format(metrics.f1_score(label_bv_tmp, xai_tmp), '.3f')
        hau_bv = format(hausdorff_distance(np.resize(xai_tmp, (224, 224)), np.resize(label_bv_tmp, (224, 224))), '.3f')

        xai_bv_list.append(xai_bv_tmp)
        bv_fundus_list.append(bv_fundus_tmp)
        iou_bv_list.append(iou_bv)
        dice_bv_list.append(dice_bv)
        hau_bv_list.append(hau_bv)

    xai_bv_mean, xai_bv_std = bootstrap_sample(value_list=np.array(xai_bv_list), times=100)
    bv_fundus_mean, bv_fundus_std = bootstrap_sample(value_list=np.array(bv_fundus_list), times=100)
    bv_difference_mean, bv_difference_std = bootstrap_sample(value_list=np.array(xai_bv_list) -
                                                                        np.array(bv_fundus_list), times=100)
    iou_bv_mean, iou_bv_std = bootstrap_sample(value_list=np.array(iou_bv_list), times=100)
    dice_bv_mean, dice_bv_std = bootstrap_sample(value_list=np.array(dice_bv_list), times=100)
    hau_bv_mean, hau_bv_std = bootstrap_sample(value_list=np.array(hau_bv_list), times=100)
    _, p_value = stats.ttest_rel(xai_bv_list, bv_fundus_list)

    results_df.append([f"DeiT-T", f"{id}", f"Score-CAM", f"Blood vessels",
                       f"{xai_bv_mean} ({xai_bv_std})",
                       f"{bv_fundus_mean} ({bv_fundus_std})",
                       f"{bv_difference_mean} ({bv_difference_std})", f"{format(p_value, '.3e')}",
                       f"{iou_bv_mean} ({iou_bv_std})", f"{dice_bv_mean} ({dice_bv_std})",
                       f"{hau_bv_mean} ({hau_bv_std})"])

    xai_bv_list_full.append(xai_bv_list)
    bv_fundus_list_full.append(bv_fundus_list)
    iou_bv_list_full.append(iou_bv_list)
    dice_bv_list_full.append(dice_bv_list)
    hau_bv_list_full.append(hau_bv_list)

pd.DataFrame(results_df).to_csv("explanation_deit_t_Score_cam_cross_validation.csv", index=False, encoding="cp1252")

results_df_full = []

xai_oc_mean, xai_oc_std = bootstrap_sample(value_list=np.array(sum(xai_oc_list_full, [])), times=100)
oc_fundus_mean, oc_fundus_std = bootstrap_sample(value_list=np.array(sum(oc_fundus_list_full, [])), times=100)
oc_difference_mean, oc_difference_std = bootstrap_sample(value_list=np.array(sum(xai_oc_list_full, [])) -
                                                                    np.array(sum(oc_fundus_list_full, [])), times=100)
iou_oc_mean, iou_oc_std = bootstrap_sample(value_list=np.array(sum(iou_oc_list_full, [])), times=100)
dice_oc_mean, dice_oc_std = bootstrap_sample(value_list=np.array(sum(dice_oc_list_full, [])), times=100)
hau_oc_mean, hau_oc_std = bootstrap_sample(value_list=np.array(sum(hau_oc_list_full, [])), times=100)
_, p_value = stats.ttest_rel(sum(xai_oc_list_full, []), sum(oc_fundus_list_full, []))

results_df_full.append([f"DeiT-T", f"Score-CAM", f"Optic cup",
                        f"{xai_oc_mean} ({xai_oc_std})", f"{oc_fundus_mean} ({oc_fundus_std})",
                        f"{oc_difference_mean} ({oc_difference_std})", f"{format(p_value, '.3e')}",
                        f"{iou_oc_mean} ({iou_oc_std})", f"{dice_oc_mean} ({dice_oc_std})",
                        f"{hau_oc_mean} ({hau_oc_std})"])

xai_od_mean, xai_od_std = bootstrap_sample(value_list=np.array(sum(xai_od_list_full, [])), times=100)
od_fundus_mean, od_fundus_std = bootstrap_sample(value_list=np.array(sum(od_fundus_list_full, [])), times=100)
od_difference_mean, od_difference_std = bootstrap_sample(value_list=np.array(sum(xai_od_list_full, [])) -
                                                                    np.array(sum(od_fundus_list_full, [])), times=100)
iou_od_mean, iou_od_std = bootstrap_sample(value_list=np.array(sum(iou_od_list_full, [])), times=100)
dice_od_mean, dice_od_std = bootstrap_sample(value_list=np.array(sum(dice_od_list_full, [])), times=100)
hau_od_mean, hau_od_std = bootstrap_sample(value_list=np.array(sum(hau_od_list_full, [])), times=100)
_, p_value = stats.ttest_rel(sum(xai_od_list_full, []), sum(od_fundus_list_full, []))

results_df_full.append([f"DeiT-T", f"Score-CAM", f"Optic disk",
                        f"{xai_od_mean} ({xai_od_std})",
                        f"{od_fundus_mean} ({od_fundus_std})",
                        f"{od_difference_mean} ({od_difference_std})", f"{format(p_value, '.3e')}",
                        f"{iou_od_mean} ({iou_od_std})", f"{dice_od_mean} ({dice_od_std})",
                        f"{hau_od_mean} ({hau_od_std})"])

xai_bv_mean, xai_bv_std = bootstrap_sample(value_list=np.array(sum(xai_bv_list_full, [])), times=100)
bv_fundus_mean, bv_fundus_std = bootstrap_sample(value_list=np.array(sum(bv_fundus_list_full, [])), times=100)
bv_difference_mean, bv_difference_std = bootstrap_sample(value_list=np.array(sum(xai_bv_list_full, [])) -
                                                                    np.array(sum(bv_fundus_list_full, [])), times=100)
iou_bv_mean, iou_bv_std = bootstrap_sample(value_list=np.array(sum(iou_bv_list_full, [])), times=100)
dice_bv_mean, dice_bv_std = bootstrap_sample(value_list=np.array(sum(dice_bv_list_full, [])), times=100)
hau_bv_mean, hau_bv_std = bootstrap_sample(value_list=np.array(sum(hau_bv_list_full, [])), times=100)
_, p_value = stats.ttest_rel(sum(xai_bv_list_full, []), sum(bv_fundus_list_full, []))

results_df_full.append([f"DeiT-T", f"Score-CAM", f"Blood vessels",
                        f"{xai_bv_mean} ({xai_bv_std})",
                        f"{bv_fundus_mean} ({bv_fundus_std})",
                        f"{bv_difference_mean} ({bv_difference_std})", f"{format(p_value, '.3e')}",
                        f"{iou_bv_mean} ({iou_bv_std})", f"{dice_bv_mean} ({dice_bv_std})",
                        f"{hau_bv_mean} ({hau_bv_std})"])

pd.DataFrame(results_df_full).to_csv("explanation_deit_t_Score_cam_aggregation.csv", index=False, encoding="cp1252")



########################################################################################################################
# Eigen-CAM
########################################################################################################################
xai_oc_list_full = []
oc_fundus_list_full = []
iou_oc_list_full = []
dice_oc_list_full = []
hau_oc_list_full = []

xai_od_list_full = []
od_fundus_list_full = []
iou_od_list_full = []
dice_od_list_full = []
hau_od_list_full = []

xai_bv_list_full = []
bv_fundus_list_full = []
iou_bv_list_full = []
dice_bv_list_full = []
hau_bv_list_full = []

results_df = []

for id in range(len(dataset_id)):
    test_csv = pd.read_csv(f"data_{dataset_id[id][2]}.csv")

    test_op = Fundus_seg_2(csv_file=test_csv[test_csv["fundus_oc_seg"].notna()],
                           img_dir=image_path, label_id_1=3, label_id_2=4)
    test_bv = Fundus_seg_1(csv_file=test_csv[test_csv["bv_seg"].notna()],
                           img_dir=image_path, label_id=5)

    test_op_loader = DataLoader(test_op, batch_size=1, shuffle=False)
    test_bv_loader = DataLoader(test_bv, batch_size=1, shuffle=False)

    DeiTT_optimal = torch.load(f"DeiTT optimal {id}.pt")
    target_layers = [DeiTT_optimal.blocks[-1].norm1]
    targets = [ClassifierOutputTarget(1)]
    cam = EigenCAM(model=DeiTT_optimal, target_layers=target_layers,
                   reshape_transform=reshape_transform, use_cuda=True)

    iou_oc_list = []
    dice_oc_list = []
    hau_oc_list = []
    xai_oc_list = []
    oc_fundus_list = []

    iou_od_list = []
    dice_od_list = []
    hau_od_list = []
    xai_od_list = []
    od_fundus_list = []

    # Test optic disk and optic cup
    for data_tmp in tqdm(test_op_loader):
        images_tmp, label_oc_tmp, label_od_tmp = data_tmp[0].float().to(device), \
                                                 data_tmp[1].numpy().flatten().astype("uint8"), \
                                                 data_tmp[2].numpy().flatten().astype("uint8")
        xai_tmp = cam(input_tensor=images_tmp, targets=targets).flatten()
        threshold_value = np.percentile(xai_tmp, 95)
        xai_tmp[np.where(xai_tmp >= threshold_value)] = 1
        xai_tmp[np.where(xai_tmp != 1)] = 0
        xai_tmp = xai_tmp.astype("uint8")

        xai_oc_tmp = (xai_tmp * label_oc_tmp).sum() / xai_tmp.sum()
        oc_fundus_tmp = label_oc_tmp.sum() / 50176
        iou_oc = format(metrics.jaccard_score(label_oc_tmp, xai_tmp), '.3f')
        dice_oc = format(metrics.f1_score(label_oc_tmp, xai_tmp), '.3f')
        hau_oc = format(hausdorff_distance(np.resize(xai_tmp, (224, 224)), np.resize(label_oc_tmp, (224, 224))), '.3f')

        xai_oc_list.append(xai_oc_tmp)
        oc_fundus_list.append(oc_fundus_tmp)
        iou_oc_list.append(iou_oc)
        dice_oc_list.append(dice_oc)
        hau_oc_list.append(hau_oc)

        xai_od_tmp = (xai_tmp * label_od_tmp).sum() / xai_tmp.sum()
        od_fundus_tmp = label_od_tmp.sum() / 50176
        iou_od = format(metrics.jaccard_score(label_od_tmp, xai_tmp), '.3f')
        dice_od = format(metrics.f1_score(label_od_tmp, xai_tmp), '.3f')
        hau_od = format(hausdorff_distance(np.resize(xai_tmp, (224, 224)), np.resize(label_od_tmp, (224, 224))), '.3f')

        xai_od_list.append(xai_od_tmp)
        od_fundus_list.append(od_fundus_tmp)
        iou_od_list.append(iou_od)
        dice_od_list.append(dice_od)
        hau_od_list.append(hau_od)

    xai_oc_mean, xai_oc_std = bootstrap_sample(value_list=np.array(xai_oc_list), times=100)
    oc_fundus_mean, oc_fundus_std = bootstrap_sample(value_list=np.array(oc_fundus_list), times=100)
    oc_difference_mean, oc_difference_std = bootstrap_sample(value_list=np.array(xai_oc_list) -
                                                                        np.array(oc_fundus_list), times=100)
    iou_oc_mean, iou_oc_std = bootstrap_sample(value_list=np.array(iou_oc_list), times=100)
    dice_oc_mean, dice_oc_std = bootstrap_sample(value_list=np.array(dice_oc_list), times=100)
    hau_oc_mean, hau_oc_std = bootstrap_sample(value_list=np.array(hau_oc_list), times=100)
    _, p_value = stats.ttest_rel(xai_oc_list, oc_fundus_list)

    results_df.append([f"DeiT-T", f"{id}", f"Eigen-CAM", f"Optic cup",
                       f"{xai_oc_mean} ({xai_oc_std})", f"{oc_fundus_mean} ({oc_fundus_std})",
                       f"{oc_difference_mean} ({oc_difference_std})", f"{format(p_value, '.3e')}",
                       f"{iou_oc_mean} ({iou_oc_std})", f"{dice_oc_mean} ({dice_oc_std})",
                       f"{hau_oc_mean} ({hau_oc_std})"])

    xai_oc_list_full.append(xai_oc_list)
    oc_fundus_list_full.append(oc_fundus_list)
    iou_oc_list_full.append(iou_oc_list)
    dice_oc_list_full.append(dice_oc_list)
    hau_oc_list_full.append(hau_oc_list)

    xai_od_mean, xai_od_std = bootstrap_sample(value_list=np.array(xai_od_list), times=100)
    od_fundus_mean, od_fundus_std = bootstrap_sample(value_list=np.array(od_fundus_list), times=100)
    od_difference_mean, od_difference_std = bootstrap_sample(value_list=np.array(xai_od_list) -
                                                                        np.array(od_fundus_list), times=100)
    iou_od_mean, iou_od_std = bootstrap_sample(value_list=np.array(iou_od_list), times=100)
    dice_od_mean, dice_od_std = bootstrap_sample(value_list=np.array(dice_od_list), times=100)
    hau_od_mean, hau_od_std = bootstrap_sample(value_list=np.array(hau_od_list), times=100)
    _, p_value = stats.ttest_rel(xai_od_list, od_fundus_list)

    results_df.append([f"DeiT-T", f"{id}", f"Eigen-CAM", f"Optic disk",
                       f"{xai_od_mean} ({xai_od_std})",
                       f"{od_fundus_mean} ({od_fundus_std})",
                       f"{od_difference_mean} ({od_difference_std})", f"{format(p_value, '.3e')}",
                       f"{iou_od_mean} ({iou_od_std})", f"{dice_od_mean} ({dice_od_std})",
                       f"{hau_od_mean} ({hau_od_std})"])

    xai_od_list_full.append(xai_od_list)
    od_fundus_list_full.append(od_fundus_list)
    iou_od_list_full.append(iou_od_list)
    dice_od_list_full.append(dice_od_list)
    hau_od_list_full.append(hau_od_list)

    iou_bv_list = []
    dice_bv_list = []
    hau_bv_list = []
    xai_bv_list = []
    bv_fundus_list = []

    # Test optic disk and optic cup
    for data_tmp in tqdm(test_bv_loader):
        images_tmp, label_bv_tmp = data_tmp[0].float().to(device), \
                                   data_tmp[1].numpy().flatten().astype("uint8")
        xai_tmp = cam(input_tensor=images_tmp, targets=targets).flatten()
        threshold_value = np.percentile(xai_tmp, 95)
        xai_tmp[np.where(xai_tmp >= threshold_value)] = 1
        xai_tmp[np.where(xai_tmp != 1)] = 0
        xai_tmp = xai_tmp.astype("uint8")

        xai_bv_tmp = (xai_tmp * label_bv_tmp).sum() / xai_tmp.sum()
        bv_fundus_tmp = label_bv_tmp.sum() / 50176
        iou_bv = format(metrics.jaccard_score(label_bv_tmp, xai_tmp), '.3f')
        dice_bv = format(metrics.f1_score(label_bv_tmp, xai_tmp), '.3f')
        hau_bv = format(hausdorff_distance(np.resize(xai_tmp, (224, 224)), np.resize(label_bv_tmp, (224, 224))), '.3f')

        xai_bv_list.append(xai_bv_tmp)
        bv_fundus_list.append(bv_fundus_tmp)
        iou_bv_list.append(iou_bv)
        dice_bv_list.append(dice_bv)
        hau_bv_list.append(hau_bv)

    xai_bv_mean, xai_bv_std = bootstrap_sample(value_list=np.array(xai_bv_list), times=100)
    bv_fundus_mean, bv_fundus_std = bootstrap_sample(value_list=np.array(bv_fundus_list), times=100)
    bv_difference_mean, bv_difference_std = bootstrap_sample(value_list=np.array(xai_bv_list) -
                                                                        np.array(bv_fundus_list), times=100)
    iou_bv_mean, iou_bv_std = bootstrap_sample(value_list=np.array(iou_bv_list), times=100)
    dice_bv_mean, dice_bv_std = bootstrap_sample(value_list=np.array(dice_bv_list), times=100)
    hau_bv_mean, hau_bv_std = bootstrap_sample(value_list=np.array(hau_bv_list), times=100)
    _, p_value = stats.ttest_rel(xai_bv_list, bv_fundus_list)

    results_df.append([f"DeiT-T", f"{id}", f"Eigen-CAM", f"Blood vessels",
                       f"{xai_bv_mean} ({xai_bv_std})",
                       f"{bv_fundus_mean} ({bv_fundus_std})",
                       f"{bv_difference_mean} ({bv_difference_std})", f"{format(p_value, '.3e')}",
                       f"{iou_bv_mean} ({iou_bv_std})", f"{dice_bv_mean} ({dice_bv_std})",
                       f"{hau_bv_mean} ({hau_bv_std})"])

    xai_bv_list_full.append(xai_bv_list)
    bv_fundus_list_full.append(bv_fundus_list)
    iou_bv_list_full.append(iou_bv_list)
    dice_bv_list_full.append(dice_bv_list)
    hau_bv_list_full.append(hau_bv_list)

pd.DataFrame(results_df).to_csv("explanation_deit_t_Eigen_cam_cross_validation.csv", index=False, encoding="cp1252")

results_df_full = []

xai_oc_mean, xai_oc_std = bootstrap_sample(value_list=np.array(sum(xai_oc_list_full, [])), times=100)
oc_fundus_mean, oc_fundus_std = bootstrap_sample(value_list=np.array(sum(oc_fundus_list_full, [])), times=100)
oc_difference_mean, oc_difference_std = bootstrap_sample(value_list=np.array(sum(xai_oc_list_full, [])) -
                                                                    np.array(sum(oc_fundus_list_full, [])), times=100)
iou_oc_mean, iou_oc_std = bootstrap_sample(value_list=np.array(sum(iou_oc_list_full, [])), times=100)
dice_oc_mean, dice_oc_std = bootstrap_sample(value_list=np.array(sum(dice_oc_list_full, [])), times=100)
hau_oc_mean, hau_oc_std = bootstrap_sample(value_list=np.array(sum(hau_oc_list_full, [])), times=100)
_, p_value = stats.ttest_rel(sum(xai_oc_list_full, []), sum(oc_fundus_list_full, []))

results_df_full.append([f"DeiT-T", f"Eigen-CAM", f"Optic cup",
                        f"{xai_oc_mean} ({xai_oc_std})", f"{oc_fundus_mean} ({oc_fundus_std})",
                        f"{oc_difference_mean} ({oc_difference_std})", f"{format(p_value, '.3e')}",
                        f"{iou_oc_mean} ({iou_oc_std})", f"{dice_oc_mean} ({dice_oc_std})",
                        f"{hau_oc_mean} ({hau_oc_std})"])

xai_od_mean, xai_od_std = bootstrap_sample(value_list=np.array(sum(xai_od_list_full, [])), times=100)
od_fundus_mean, od_fundus_std = bootstrap_sample(value_list=np.array(sum(od_fundus_list_full, [])), times=100)
od_difference_mean, od_difference_std = bootstrap_sample(value_list=np.array(sum(xai_od_list_full, [])) -
                                                                    np.array(sum(od_fundus_list_full, [])), times=100)
iou_od_mean, iou_od_std = bootstrap_sample(value_list=np.array(sum(iou_od_list_full, [])), times=100)
dice_od_mean, dice_od_std = bootstrap_sample(value_list=np.array(sum(dice_od_list_full, [])), times=100)
hau_od_mean, hau_od_std = bootstrap_sample(value_list=np.array(sum(hau_od_list_full, [])), times=100)
_, p_value = stats.ttest_rel(sum(xai_od_list_full, []), sum(od_fundus_list_full, []))

results_df_full.append([f"DeiT-T", f"Eigen-CAM", f"Optic disk",
                        f"{xai_od_mean} ({xai_od_std})",
                        f"{od_fundus_mean} ({od_fundus_std})",
                        f"{od_difference_mean} ({od_difference_std})", f"{format(p_value, '.3e')}",
                        f"{iou_od_mean} ({iou_od_std})", f"{dice_od_mean} ({dice_od_std})",
                        f"{hau_od_mean} ({hau_od_std})"])

xai_bv_mean, xai_bv_std = bootstrap_sample(value_list=np.array(sum(xai_bv_list_full, [])), times=100)
bv_fundus_mean, bv_fundus_std = bootstrap_sample(value_list=np.array(sum(bv_fundus_list_full, [])), times=100)
bv_difference_mean, bv_difference_std = bootstrap_sample(value_list=np.array(sum(xai_bv_list_full, [])) -
                                                                    np.array(sum(bv_fundus_list_full, [])), times=100)
iou_bv_mean, iou_bv_std = bootstrap_sample(value_list=np.array(sum(iou_bv_list_full, [])), times=100)
dice_bv_mean, dice_bv_std = bootstrap_sample(value_list=np.array(sum(dice_bv_list_full, [])), times=100)
hau_bv_mean, hau_bv_std = bootstrap_sample(value_list=np.array(sum(hau_bv_list_full, [])), times=100)
_, p_value = stats.ttest_rel(sum(xai_bv_list_full, []), sum(bv_fundus_list_full, []))

results_df_full.append([f"DeiT-T", f"Eigen-CAM", f"Blood vessels",
                        f"{xai_bv_mean} ({xai_bv_std})",
                        f"{bv_fundus_mean} ({bv_fundus_std})",
                        f"{bv_difference_mean} ({bv_difference_std})", f"{format(p_value, '.3e')}",
                        f"{iou_bv_mean} ({iou_bv_std})", f"{dice_bv_mean} ({dice_bv_std})",
                        f"{hau_bv_mean} ({hau_bv_std})"])

pd.DataFrame(results_df_full).to_csv("explanation_deit_t_Eigen_cam_aggregation.csv", index=False, encoding="cp1252")

########################################################################################################################
# Layer-CAM
########################################################################################################################
xai_oc_list_full = []
oc_fundus_list_full = []
iou_oc_list_full = []
dice_oc_list_full = []
hau_oc_list_full = []

xai_od_list_full = []
od_fundus_list_full = []
iou_od_list_full = []
dice_od_list_full = []
hau_od_list_full = []

xai_bv_list_full = []
bv_fundus_list_full = []
iou_bv_list_full = []
dice_bv_list_full = []
hau_bv_list_full = []

results_df = []

for id in range(len(dataset_id)):
    test_csv = pd.read_csv(f"data_{dataset_id[id][2]}.csv")

    test_op = Fundus_seg_2(csv_file=test_csv[test_csv["fundus_oc_seg"].notna()],
                           img_dir=image_path, label_id_1=3, label_id_2=4)
    test_bv = Fundus_seg_1(csv_file=test_csv[test_csv["bv_seg"].notna()],
                           img_dir=image_path, label_id=5)

    test_op_loader = DataLoader(test_op, batch_size=1, shuffle=False)
    test_bv_loader = DataLoader(test_bv, batch_size=1, shuffle=False)

    DeiTT_optimal = torch.load(f"DeiTT optimal {id}.pt")
    target_layers = [DeiTT_optimal.blocks[-1].norm1]
    targets = [ClassifierOutputTarget(1)]
    cam = LayerCAM(model=DeiTT_optimal, target_layers=target_layers,
                   reshape_transform=reshape_transform, use_cuda=True)

    iou_oc_list = []
    dice_oc_list = []
    hau_oc_list = []
    xai_oc_list = []
    oc_fundus_list = []

    iou_od_list = []
    dice_od_list = []
    hau_od_list = []
    xai_od_list = []
    od_fundus_list = []

    # Test optic disk and optic cup
    for data_tmp in tqdm(test_op_loader):
        images_tmp, label_oc_tmp, label_od_tmp = data_tmp[0].float().to(device), \
                                                 data_tmp[1].numpy().flatten().astype("uint8"), \
                                                 data_tmp[2].numpy().flatten().astype("uint8")
        xai_tmp = cam(input_tensor=images_tmp, targets=targets).flatten()
        threshold_value = np.percentile(xai_tmp, 95)
        xai_tmp[np.where(xai_tmp >= threshold_value)] = 1
        xai_tmp[np.where(xai_tmp != 1)] = 0
        xai_tmp = xai_tmp.astype("uint8")

        xai_oc_tmp = (xai_tmp * label_oc_tmp).sum() / xai_tmp.sum()
        oc_fundus_tmp = label_oc_tmp.sum() / 50176
        iou_oc = format(metrics.jaccard_score(label_oc_tmp, xai_tmp), '.3f')
        dice_oc = format(metrics.f1_score(label_oc_tmp, xai_tmp), '.3f')
        hau_oc = format(hausdorff_distance(np.resize(xai_tmp, (224, 224)), np.resize(label_oc_tmp, (224, 224))), '.3f')

        xai_oc_list.append(xai_oc_tmp)
        oc_fundus_list.append(oc_fundus_tmp)
        iou_oc_list.append(iou_oc)
        dice_oc_list.append(dice_oc)
        hau_oc_list.append(hau_oc)

        xai_od_tmp = (xai_tmp * label_od_tmp).sum() / xai_tmp.sum()
        od_fundus_tmp = label_od_tmp.sum() / 50176
        iou_od = format(metrics.jaccard_score(label_od_tmp, xai_tmp), '.3f')
        dice_od = format(metrics.f1_score(label_od_tmp, xai_tmp), '.3f')
        hau_od = format(hausdorff_distance(np.resize(xai_tmp, (224, 224)), np.resize(label_od_tmp, (224, 224))), '.3f')

        xai_od_list.append(xai_od_tmp)
        od_fundus_list.append(od_fundus_tmp)
        iou_od_list.append(iou_od)
        dice_od_list.append(dice_od)
        hau_od_list.append(hau_od)

    xai_oc_mean, xai_oc_std = bootstrap_sample(value_list=np.array(xai_oc_list), times=100)
    oc_fundus_mean, oc_fundus_std = bootstrap_sample(value_list=np.array(oc_fundus_list), times=100)
    oc_difference_mean, oc_difference_std = bootstrap_sample(value_list=np.array(xai_oc_list) -
                                                                        np.array(oc_fundus_list), times=100)
    iou_oc_mean, iou_oc_std = bootstrap_sample(value_list=np.array(iou_oc_list), times=100)
    dice_oc_mean, dice_oc_std = bootstrap_sample(value_list=np.array(dice_oc_list), times=100)
    hau_oc_mean, hau_oc_std = bootstrap_sample(value_list=np.array(hau_oc_list), times=100)
    _, p_value = stats.ttest_rel(xai_oc_list, oc_fundus_list)

    results_df.append([f"DeiT-T", f"{id}", f"Layer-CAM", f"Optic cup",
                       f"{xai_oc_mean} ({xai_oc_std})", f"{oc_fundus_mean} ({oc_fundus_std})",
                       f"{oc_difference_mean} ({oc_difference_std})", f"{format(p_value, '.3e')}",
                       f"{iou_oc_mean} ({iou_oc_std})", f"{dice_oc_mean} ({dice_oc_std})",
                       f"{hau_oc_mean} ({hau_oc_std})"])

    xai_oc_list_full.append(xai_oc_list)
    oc_fundus_list_full.append(oc_fundus_list)
    iou_oc_list_full.append(iou_oc_list)
    dice_oc_list_full.append(dice_oc_list)
    hau_oc_list_full.append(hau_oc_list)

    xai_od_mean, xai_od_std = bootstrap_sample(value_list=np.array(xai_od_list), times=100)
    od_fundus_mean, od_fundus_std = bootstrap_sample(value_list=np.array(od_fundus_list), times=100)
    od_difference_mean, od_difference_std = bootstrap_sample(value_list=np.array(xai_od_list) -
                                                                        np.array(od_fundus_list), times=100)
    iou_od_mean, iou_od_std = bootstrap_sample(value_list=np.array(iou_od_list), times=100)
    dice_od_mean, dice_od_std = bootstrap_sample(value_list=np.array(dice_od_list), times=100)
    hau_od_mean, hau_od_std = bootstrap_sample(value_list=np.array(hau_od_list), times=100)
    _, p_value = stats.ttest_rel(xai_od_list, od_fundus_list)

    results_df.append([f"DeiT-T", f"{id}", f"Layer-CAM", f"Optic disk",
                       f"{xai_od_mean} ({xai_od_std})",
                       f"{od_fundus_mean} ({od_fundus_std})",
                       f"{od_difference_mean} ({od_difference_std})", f"{format(p_value, '.3e')}",
                       f"{iou_od_mean} ({iou_od_std})", f"{dice_od_mean} ({dice_od_std})",
                       f"{hau_od_mean} ({hau_od_std})"])

    xai_od_list_full.append(xai_od_list)
    od_fundus_list_full.append(od_fundus_list)
    iou_od_list_full.append(iou_od_list)
    dice_od_list_full.append(dice_od_list)
    hau_od_list_full.append(hau_od_list)

    iou_bv_list = []
    dice_bv_list = []
    hau_bv_list = []
    xai_bv_list = []
    bv_fundus_list = []

    # Test optic disk and optic cup
    for data_tmp in tqdm(test_bv_loader):
        images_tmp, label_bv_tmp = data_tmp[0].float().to(device), \
                                   data_tmp[1].numpy().flatten().astype("uint8")
        xai_tmp = cam(input_tensor=images_tmp, targets=targets).flatten()
        threshold_value = np.percentile(xai_tmp, 95)
        xai_tmp[np.where(xai_tmp >= threshold_value)] = 1
        xai_tmp[np.where(xai_tmp != 1)] = 0
        xai_tmp = xai_tmp.astype("uint8")

        xai_bv_tmp = (xai_tmp * label_bv_tmp).sum() / xai_tmp.sum()
        bv_fundus_tmp = label_bv_tmp.sum() / 50176
        iou_bv = format(metrics.jaccard_score(label_bv_tmp, xai_tmp), '.3f')
        dice_bv = format(metrics.f1_score(label_bv_tmp, xai_tmp), '.3f')
        hau_bv = format(hausdorff_distance(np.resize(xai_tmp, (224, 224)), np.resize(label_bv_tmp, (224, 224))), '.3f')

        xai_bv_list.append(xai_bv_tmp)
        bv_fundus_list.append(bv_fundus_tmp)
        iou_bv_list.append(iou_bv)
        dice_bv_list.append(dice_bv)
        hau_bv_list.append(hau_bv)

    xai_bv_mean, xai_bv_std = bootstrap_sample(value_list=np.array(xai_bv_list), times=100)
    bv_fundus_mean, bv_fundus_std = bootstrap_sample(value_list=np.array(bv_fundus_list), times=100)
    bv_difference_mean, bv_difference_std = bootstrap_sample(value_list=np.array(xai_bv_list) -
                                                                        np.array(bv_fundus_list), times=100)
    iou_bv_mean, iou_bv_std = bootstrap_sample(value_list=np.array(iou_bv_list), times=100)
    dice_bv_mean, dice_bv_std = bootstrap_sample(value_list=np.array(dice_bv_list), times=100)
    hau_bv_mean, hau_bv_std = bootstrap_sample(value_list=np.array(hau_bv_list), times=100)
    _, p_value = stats.ttest_rel(xai_bv_list, bv_fundus_list)

    results_df.append([f"DeiT-T", f"{id}", f"Layer-CAM", f"Blood vessels",
                       f"{xai_bv_mean} ({xai_bv_std})",
                       f"{bv_fundus_mean} ({bv_fundus_std})",
                       f"{bv_difference_mean} ({bv_difference_std})", f"{format(p_value, '.3e')}",
                       f"{iou_bv_mean} ({iou_bv_std})", f"{dice_bv_mean} ({dice_bv_std})",
                       f"{hau_bv_mean} ({hau_bv_std})"])

    xai_bv_list_full.append(xai_bv_list)
    bv_fundus_list_full.append(bv_fundus_list)
    iou_bv_list_full.append(iou_bv_list)
    dice_bv_list_full.append(dice_bv_list)
    hau_bv_list_full.append(hau_bv_list)

pd.DataFrame(results_df).to_csv("explanation_deit_t_Layer_cam_cross_validation.csv", index=False, encoding="cp1252")

results_df_full = []

xai_oc_mean, xai_oc_std = bootstrap_sample(value_list=np.array(sum(xai_oc_list_full, [])), times=100)
oc_fundus_mean, oc_fundus_std = bootstrap_sample(value_list=np.array(sum(oc_fundus_list_full, [])), times=100)
oc_difference_mean, oc_difference_std = bootstrap_sample(value_list=np.array(sum(xai_oc_list_full, [])) -
                                                                    np.array(sum(oc_fundus_list_full, [])), times=100)
iou_oc_mean, iou_oc_std = bootstrap_sample(value_list=np.array(sum(iou_oc_list_full, [])), times=100)
dice_oc_mean, dice_oc_std = bootstrap_sample(value_list=np.array(sum(dice_oc_list_full, [])), times=100)
hau_oc_mean, hau_oc_std = bootstrap_sample(value_list=np.array(sum(hau_oc_list_full, [])), times=100)
_, p_value = stats.ttest_rel(sum(xai_oc_list_full, []), sum(oc_fundus_list_full, []))

results_df_full.append([f"DeiT-T", f"Layer-CAM", f"Optic cup",
                        f"{xai_oc_mean} ({xai_oc_std})", f"{oc_fundus_mean} ({oc_fundus_std})",
                        f"{oc_difference_mean} ({oc_difference_std})", f"{format(p_value, '.3e')}",
                        f"{iou_oc_mean} ({iou_oc_std})", f"{dice_oc_mean} ({dice_oc_std})",
                        f"{hau_oc_mean} ({hau_oc_std})"])

xai_od_mean, xai_od_std = bootstrap_sample(value_list=np.array(sum(xai_od_list_full, [])), times=100)
od_fundus_mean, od_fundus_std = bootstrap_sample(value_list=np.array(sum(od_fundus_list_full, [])), times=100)
od_difference_mean, od_difference_std = bootstrap_sample(value_list=np.array(sum(xai_od_list_full, [])) -
                                                                    np.array(sum(od_fundus_list_full, [])), times=100)
iou_od_mean, iou_od_std = bootstrap_sample(value_list=np.array(sum(iou_od_list_full, [])), times=100)
dice_od_mean, dice_od_std = bootstrap_sample(value_list=np.array(sum(dice_od_list_full, [])), times=100)
hau_od_mean, hau_od_std = bootstrap_sample(value_list=np.array(sum(hau_od_list_full, [])), times=100)
_, p_value = stats.ttest_rel(sum(xai_od_list_full, []), sum(od_fundus_list_full, []))

results_df_full.append([f"DeiT-T", f"Layer-CAM", f"Optic disk",
                        f"{xai_od_mean} ({xai_od_std})",
                        f"{od_fundus_mean} ({od_fundus_std})",
                        f"{od_difference_mean} ({od_difference_std})", f"{format(p_value, '.3e')}",
                        f"{iou_od_mean} ({iou_od_std})", f"{dice_od_mean} ({dice_od_std})",
                        f"{hau_od_mean} ({hau_od_std})"])

xai_bv_mean, xai_bv_std = bootstrap_sample(value_list=np.array(sum(xai_bv_list_full, [])), times=100)
bv_fundus_mean, bv_fundus_std = bootstrap_sample(value_list=np.array(sum(bv_fundus_list_full, [])), times=100)
bv_difference_mean, bv_difference_std = bootstrap_sample(value_list=np.array(sum(xai_bv_list_full, [])) -
                                                                    np.array(sum(bv_fundus_list_full, [])), times=100)
iou_bv_mean, iou_bv_std = bootstrap_sample(value_list=np.array(sum(iou_bv_list_full, [])), times=100)
dice_bv_mean, dice_bv_std = bootstrap_sample(value_list=np.array(sum(dice_bv_list_full, [])), times=100)
hau_bv_mean, hau_bv_std = bootstrap_sample(value_list=np.array(sum(hau_bv_list_full, [])), times=100)
_, p_value = stats.ttest_rel(sum(xai_bv_list_full, []), sum(bv_fundus_list_full, []))

results_df_full.append([f"DeiT-T", f"Layer-CAM", f"Blood vessels",
                        f"{xai_bv_mean} ({xai_bv_std})",
                        f"{bv_fundus_mean} ({bv_fundus_std})",
                        f"{bv_difference_mean} ({bv_difference_std})", f"{format(p_value, '.3e')}",
                        f"{iou_bv_mean} ({iou_bv_std})", f"{dice_bv_mean} ({dice_bv_std})",
                        f"{hau_bv_mean} ({hau_bv_std})"])

pd.DataFrame(results_df_full).to_csv("explanation_deit_t_Layer_cam_aggregation.csv", index=False, encoding="cp1252")
