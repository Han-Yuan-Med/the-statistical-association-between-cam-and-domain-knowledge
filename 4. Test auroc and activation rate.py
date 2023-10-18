import pandas as pd
import numpy as np
import scipy.stats

auroc_list = np.zeros(4 * 6)
vgg_results = pd.read_csv("results_vgg_11_cls.csv")
res_results = pd.read_csv("results_res_18_cls.csv")
deit_results = pd.read_csv("results_deit_tiny_cls.csv")
swin_results = pd.read_csv("results_swin_tiny_cls.csv")

auroc_data = pd.concat(
    [vgg_results.iloc[:, 3], res_results.iloc[:, 3], deit_results.iloc[:, 3], swin_results.iloc[:, 3]])
for i in range(len(auroc_data)):
    auroc_list[i] = float(auroc_data.iloc[i].split(" ")[0])

# Grad CAM
activation_list = np.zeros(4 * 6 * 3)
vgg_grad_results = pd.read_csv("explanation_vgg_11_grad_cam_cross_validation.csv").iloc[:, 4]
res_grad_results = pd.read_csv("explanation_res_18_grad_cam_cross_validation.csv").iloc[:, 4]
deit_grad_results = pd.read_csv("explanation_deit_t_grad_cam_cross_validation.csv").iloc[:, 4]
swin_grad_results = pd.read_csv("explanation_swin_t_grad_cam_cross_validation.csv").iloc[:, 4]
activation_rate_data = pd.concat([vgg_grad_results, res_grad_results, deit_grad_results, swin_grad_results])
for i in range(len(activation_rate_data)):
    activation_list[i] = float(activation_rate_data.iloc[i].split(" ")[0])

activation_list_cup = []
for i in range(0, len(activation_list), 3):
    activation_list_cup.append(activation_list[i])

activation_list_disk = []
for i in range(1, len(activation_list), 3):
    activation_list_disk.append(activation_list[i])

activation_list_ves = []
for i in range(2, len(activation_list), 3):
    activation_list_ves.append(activation_list[i])

grad_cup_pearson = scipy.stats.pearsonr(np.asarray(activation_list_cup), auroc_list)
grad_cup_spearman = scipy.stats.spearmanr(np.asarray(activation_list_cup), auroc_list)

grad_disk_pearson = scipy.stats.pearsonr(np.asarray(activation_list_disk), auroc_list)
grad_disk_spearman = scipy.stats.spearmanr(np.asarray(activation_list_disk), auroc_list)

grad_ves_pearson = scipy.stats.pearsonr(np.asarray(activation_list_ves), auroc_list)
grad_ves_spearman = scipy.stats.spearmanr(np.asarray(activation_list_ves), auroc_list)

results_df = [["Grad-CAM", "Optic cup", f"{format(grad_cup_pearson[0], '.2f')}",
               f"{format(grad_cup_pearson[1], '.2e')}", f"{format(grad_cup_spearman[0], '.2f')}",
               f"{format(grad_cup_spearman[1], '.2e')}"],
              ["Grad-CAM", "Optic disk", f"{format(grad_disk_pearson[0], '.2f')}",
               f"{format(grad_disk_pearson[1], '.2e')}", f"{format(grad_disk_spearman[0], '.2f')}",
               f"{format(grad_disk_spearman[1], '.2e')}"],
              ["Grad-CAM", "Optic cup", f"{format(grad_ves_pearson[0], '.2f')}",
               f"{format(grad_ves_pearson[1], '.2e')}", f"{format(grad_ves_spearman[0], '.2f')}",
               f"{format(grad_ves_spearman[1], '.2e')}"]]

# XGrad CAM
activation_list = np.zeros(4 * 6 * 3)
vgg_XGrad_results = pd.read_csv("explanation_vgg_11_XGrad_cam_cross_validation.csv").iloc[:, 4]
res_XGrad_results = pd.read_csv("explanation_res_18_XGrad_cam_cross_validation.csv").iloc[:, 4]
deit_XGrad_results = pd.read_csv("explanation_deit_t_XGrad_cam_cross_validation.csv").iloc[:, 4]
swin_XGrad_results = pd.read_csv("explanation_swin_t_XGrad_cam_cross_validation.csv").iloc[:, 4]
activation_rate_data = pd.concat([vgg_XGrad_results, res_XGrad_results, deit_XGrad_results, swin_XGrad_results])
for i in range(len(activation_rate_data)):
    activation_list[i] = float(activation_rate_data.iloc[i].split(" ")[0])

activation_list_cup = []
for i in range(0, len(activation_list), 3):
    activation_list_cup.append(activation_list[i])

activation_list_disk = []
for i in range(1, len(activation_list), 3):
    activation_list_disk.append(activation_list[i])

activation_list_ves = []
for i in range(2, len(activation_list), 3):
    activation_list_ves.append(activation_list[i])

XGrad_cup_pearson = scipy.stats.pearsonr(np.asarray(activation_list_cup), auroc_list)
XGrad_cup_spearman = scipy.stats.spearmanr(np.asarray(activation_list_cup), auroc_list)

XGrad_disk_pearson = scipy.stats.pearsonr(np.asarray(activation_list_disk), auroc_list)
XGrad_disk_spearman = scipy.stats.spearmanr(np.asarray(activation_list_disk), auroc_list)

XGrad_ves_pearson = scipy.stats.pearsonr(np.asarray(activation_list_ves), auroc_list)
XGrad_ves_spearman = scipy.stats.spearmanr(np.asarray(activation_list_ves), auroc_list)

results_df.append(["XGrad-CAM", "Optic cup", f"{format(XGrad_cup_pearson[0], '.2f')}",
                   f"{format(XGrad_cup_pearson[1], '.2e')}", f"{format(XGrad_cup_spearman[0], '.2f')}",
                   f"{format(XGrad_cup_spearman[1], '.2e')}"])
results_df.append(["XGrad-CAM", "Optic disk", f"{format(XGrad_disk_pearson[0], '.2f')}",
                   f"{format(XGrad_disk_pearson[1], '.2e')}", f"{format(XGrad_disk_spearman[0], '.2f')}",
                   f"{format(XGrad_disk_spearman[1], '.2e')}"])
results_df.append(["XGrad-CAM", "Optic cup", f"{format(XGrad_ves_pearson[0], '.2f')}",
                   f"{format(XGrad_ves_pearson[1], '.2e')}", f"{format(XGrad_ves_spearman[0], '.2f')}",
                   f"{format(XGrad_ves_spearman[1], '.2e')}"])

# Score CAM
activation_list = np.zeros(4 * 6 * 3)
vgg_Score_results = pd.read_csv("explanation_vgg_11_Score_cam_cross_validation.csv").iloc[:, 4]
res_Score_results = pd.read_csv("explanation_res_18_Score_cam_cross_validation.csv").iloc[:, 4]
deit_Score_results = pd.read_csv("explanation_deit_t_Score_cam_cross_validation.csv").iloc[:, 4]
swin_Score_results = pd.read_csv("explanation_swin_t_Score_cam_cross_validation.csv").iloc[:, 4]
activation_rate_data = pd.concat([vgg_Score_results, res_Score_results, deit_Score_results, swin_Score_results])
for i in range(len(activation_rate_data)):
    activation_list[i] = float(activation_rate_data.iloc[i].split(" ")[0])

activation_list_cup = []
for i in range(0, len(activation_list), 3):
    activation_list_cup.append(activation_list[i])

activation_list_disk = []
for i in range(1, len(activation_list), 3):
    activation_list_disk.append(activation_list[i])

activation_list_ves = []
for i in range(2, len(activation_list), 3):
    activation_list_ves.append(activation_list[i])

Score_cup_pearson = scipy.stats.pearsonr(np.asarray(activation_list_cup), auroc_list)
Score_cup_spearman = scipy.stats.spearmanr(np.asarray(activation_list_cup), auroc_list)

Score_disk_pearson = scipy.stats.pearsonr(np.asarray(activation_list_disk), auroc_list)
Score_disk_spearman = scipy.stats.spearmanr(np.asarray(activation_list_disk), auroc_list)

Score_ves_pearson = scipy.stats.pearsonr(np.asarray(activation_list_ves), auroc_list)
Score_ves_spearman = scipy.stats.spearmanr(np.asarray(activation_list_ves), auroc_list)

results_df.append(["Score-CAM", "Optic cup", f"{format(Score_cup_pearson[0], '.2f')}",
                   f"{format(Score_cup_pearson[1], '.2e')}", f"{format(Score_cup_spearman[0], '.2f')}",
                   f"{format(Score_cup_spearman[1], '.2e')}"])
results_df.append(["Score-CAM", "Optic disk", f"{format(Score_disk_pearson[0], '.2f')}",
                   f"{format(Score_disk_pearson[1], '.2e')}", f"{format(Score_disk_spearman[0], '.2f')}",
                   f"{format(Score_disk_spearman[1], '.2e')}"])
results_df.append(["Score-CAM", "Optic cup", f"{format(Score_ves_pearson[0], '.2f')}",
                   f"{format(Score_ves_pearson[1], '.2e')}", f"{format(Score_ves_spearman[0], '.2f')}",
                   f"{format(Score_ves_spearman[1], '.2e')}"])

# Eigen CAM
activation_list = np.zeros(4 * 6 * 3)
vgg_Eigen_results = pd.read_csv("explanation_vgg_11_Eigen_cam_cross_validation.csv").iloc[:, 4]
res_Eigen_results = pd.read_csv("explanation_res_18_Eigen_cam_cross_validation.csv").iloc[:, 4]
deit_Eigen_results = pd.read_csv("explanation_deit_t_Eigen_cam_cross_validation.csv").iloc[:, 4]
swin_Eigen_results = pd.read_csv("explanation_swin_t_Eigen_cam_cross_validation.csv").iloc[:, 4]
activation_rate_data = pd.concat([vgg_Eigen_results, res_Eigen_results, deit_Eigen_results, swin_Eigen_results])
for i in range(len(activation_rate_data)):
    activation_list[i] = float(activation_rate_data.iloc[i].split(" ")[0])

activation_list_cup = []
for i in range(0, len(activation_list), 3):
    activation_list_cup.append(activation_list[i])

activation_list_disk = []
for i in range(1, len(activation_list), 3):
    activation_list_disk.append(activation_list[i])

activation_list_ves = []
for i in range(2, len(activation_list), 3):
    activation_list_ves.append(activation_list[i])

Eigen_cup_pearson = scipy.stats.pearsonr(np.asarray(activation_list_cup), auroc_list)
Eigen_cup_spearman = scipy.stats.spearmanr(np.asarray(activation_list_cup), auroc_list)

Eigen_disk_pearson = scipy.stats.pearsonr(np.asarray(activation_list_disk), auroc_list)
Eigen_disk_spearman = scipy.stats.spearmanr(np.asarray(activation_list_disk), auroc_list)

Eigen_ves_pearson = scipy.stats.pearsonr(np.asarray(activation_list_ves), auroc_list)
Eigen_ves_spearman = scipy.stats.spearmanr(np.asarray(activation_list_ves), auroc_list)

results_df.append(["Eigen-CAM", "Optic cup", f"{format(Eigen_cup_pearson[0], '.2f')}",
                   f"{format(Eigen_cup_pearson[1], '.2e')}", f"{format(Eigen_cup_spearman[0], '.2f')}",
                   f"{format(Eigen_cup_spearman[1], '.2e')}"])
results_df.append(["Eigen-CAM", "Optic disk", f"{format(Eigen_disk_pearson[0], '.2f')}",
                   f"{format(Eigen_disk_pearson[1], '.2e')}", f"{format(Eigen_disk_spearman[0], '.2f')}",
                   f"{format(Eigen_disk_spearman[1], '.2e')}"])
results_df.append(["Eigen-CAM", "Optic cup", f"{format(Eigen_ves_pearson[0], '.2f')}",
                   f"{format(Eigen_ves_pearson[1], '.2e')}", f"{format(Eigen_ves_spearman[0], '.2f')}",
                   f"{format(Eigen_ves_spearman[1], '.2e')}"])

# Layer CAM
activation_list = np.zeros(4 * 6 * 3)
vgg_Layer_results = pd.read_csv("explanation_vgg_11_Layer_cam_cross_validation.csv").iloc[:, 4]
res_Layer_results = pd.read_csv("explanation_res_18_Layer_cam_cross_validation.csv").iloc[:, 4]
deit_Layer_results = pd.read_csv("explanation_deit_t_Layer_cam_cross_validation.csv").iloc[:, 4]
swin_Layer_results = pd.read_csv("explanation_swin_t_Layer_cam_cross_validation.csv").iloc[:, 4]
activation_rate_data = pd.concat([vgg_Layer_results, res_Layer_results, deit_Layer_results, swin_Layer_results])
for i in range(len(activation_rate_data)):
    activation_list[i] = float(activation_rate_data.iloc[i].split(" ")[0])

activation_list_cup = []
for i in range(0, len(activation_list), 3):
    activation_list_cup.append(activation_list[i])

activation_list_disk = []
for i in range(1, len(activation_list), 3):
    activation_list_disk.append(activation_list[i])

activation_list_ves = []
for i in range(2, len(activation_list), 3):
    activation_list_ves.append(activation_list[i])

Layer_cup_pearson = scipy.stats.pearsonr(np.asarray(activation_list_cup), auroc_list)
Layer_cup_spearman = scipy.stats.spearmanr(np.asarray(activation_list_cup), auroc_list)

Layer_disk_pearson = scipy.stats.pearsonr(np.asarray(activation_list_disk), auroc_list)
Layer_disk_spearman = scipy.stats.spearmanr(np.asarray(activation_list_disk), auroc_list)

Layer_ves_pearson = scipy.stats.pearsonr(np.asarray(activation_list_ves), auroc_list)
Layer_ves_spearman = scipy.stats.spearmanr(np.asarray(activation_list_ves), auroc_list)

results_df.append(["Layer-CAM", "Optic cup", f"{format(Layer_cup_pearson[0], '.2f')}",
                   f"{format(Layer_cup_pearson[1], '.2e')}", f"{format(Layer_cup_spearman[0], '.2f')}",
                   f"{format(Layer_cup_spearman[1], '.2e')}"])
results_df.append(["Layer-CAM", "Optic disk", f"{format(Layer_disk_pearson[0], '.2f')}",
                   f"{format(Layer_disk_pearson[1], '.2e')}", f"{format(Layer_disk_spearman[0], '.2f')}",
                   f"{format(Layer_disk_spearman[1], '.2e')}"])
results_df.append(["Layer-CAM", "Optic cup", f"{format(Layer_ves_pearson[0], '.2f')}",
                   f"{format(Layer_ves_pearson[1], '.2e')}", f"{format(Layer_ves_spearman[0], '.2f')}",
                   f"{format(Layer_ves_spearman[1], '.2e')}"])

results_df = pd.DataFrame(results_df)
results_df.to_csv("correlation_results.csv", index=False, encoding="cp1252")
