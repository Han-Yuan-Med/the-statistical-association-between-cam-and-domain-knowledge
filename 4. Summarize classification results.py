import pandas as pd

vgg_results = pd.read_csv("results_vgg_11_cls.csv")
AUROC_mean = 0
AUROC_std = 0
AUPRC_mean = 0
AUPRC_std = 0
Accuracy_mean = 0
Accuracy_std = 0
Sensitivity_mean = 0
Sensitivity_std = 0
Specificity_mean = 0
Specificity_std = 0
PPV_mean = 0
PPV_std = 0
NPV_mean = 0
NPV_std = 0
for i in range(0, 6):
    AUROC_mean += float(vgg_results.iloc[i, 3].split(' (')[0])
    AUROC_std += float(vgg_results.iloc[i, 3].split(' (')[1].split(')')[0])

    AUPRC_mean += float(vgg_results.iloc[i, 4].split(' (')[0])
    AUPRC_std += float(vgg_results.iloc[i, 4].split(' (')[1].split(')')[0])

    Accuracy_mean += float(vgg_results.iloc[i, 5].split(' (')[0])
    Accuracy_std += float(vgg_results.iloc[i, 5].split(' (')[1].split(')')[0])

    Sensitivity_mean += float(vgg_results.iloc[i, 6].split(' (')[0])
    Sensitivity_std += float(vgg_results.iloc[i, 6].split(' (')[1].split(')')[0])

    Specificity_mean += float(vgg_results.iloc[i, 7].split(' (')[0])
    Specificity_std += float(vgg_results.iloc[i, 7].split(' (')[1].split(')')[0])

    PPV_mean += float(vgg_results.iloc[i, 8].split(' (')[0])
    PPV_std += float(vgg_results.iloc[i, 8].split(' (')[1].split(')')[0])

    NPV_mean += float(vgg_results.iloc[i, 9].split(' (')[0])
    NPV_std += float(vgg_results.iloc[i, 9].split(' (')[1].split(')')[0])

results_df = [[f"{format(AUROC_mean / 6, '.3f')} ({format(AUROC_std / 6, '.3f')})",
               f"{format(AUPRC_mean / 6, '.3f')} ({format(AUPRC_std / 6, '.3f')})",
               f"{format(Accuracy_mean / 6, '.3f')} ({format(Accuracy_std / 6, '.3f')})",
               f"{format(Sensitivity_mean / 6, '.3f')} ({format(Sensitivity_std / 6, '.3f')})",
               f"{format(Specificity_mean / 6, '.3f')} ({format(Specificity_std / 6, '.3f')})",
               f"{format(PPV_mean / 6, '.3f')} ({format(PPV_std / 6, '.3f')})",
               f"{format(NPV_mean / 6, '.3f')} ({format(NPV_std / 6, '.3f')})"]]
results_df = pd.DataFrame(results_df)
results_df.columns = ['AUROC', 'AUPRC', 'Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV']
results_df.to_csv("results_vgg_11_cls_aggregation.csv", index=False, encoding="cp1252")

res_results = pd.read_csv("results_res_18_cls.csv")
AUROC_mean = 0
AUROC_std = 0
AUPRC_mean = 0
AUPRC_std = 0
Accuracy_mean = 0
Accuracy_std = 0
Sensitivity_mean = 0
Sensitivity_std = 0
Specificity_mean = 0
Specificity_std = 0
PPV_mean = 0
PPV_std = 0
NPV_mean = 0
NPV_std = 0
for i in range(0, 6):
    AUROC_mean += float(res_results.iloc[i, 3].split(' (')[0])
    AUROC_std += float(res_results.iloc[i, 3].split(' (')[1].split(')')[0])

    AUPRC_mean += float(res_results.iloc[i, 4].split(' (')[0])
    AUPRC_std += float(res_results.iloc[i, 4].split(' (')[1].split(')')[0])

    Accuracy_mean += float(res_results.iloc[i, 5].split(' (')[0])
    Accuracy_std += float(res_results.iloc[i, 5].split(' (')[1].split(')')[0])

    Sensitivity_mean += float(res_results.iloc[i, 6].split(' (')[0])
    Sensitivity_std += float(res_results.iloc[i, 6].split(' (')[1].split(')')[0])

    Specificity_mean += float(res_results.iloc[i, 7].split(' (')[0])
    Specificity_std += float(res_results.iloc[i, 7].split(' (')[1].split(')')[0])

    PPV_mean += float(res_results.iloc[i, 8].split(' (')[0])
    PPV_std += float(res_results.iloc[i, 8].split(' (')[1].split(')')[0])

    NPV_mean += float(res_results.iloc[i, 9].split(' (')[0])
    NPV_std += float(res_results.iloc[i, 9].split(' (')[1].split(')')[0])

results_df = [[f"{format(AUROC_mean / 6, '.3f')} ({format(AUROC_std / 6, '.3f')})",
               f"{format(AUPRC_mean / 6, '.3f')} ({format(AUPRC_std / 6, '.3f')})",
               f"{format(Accuracy_mean / 6, '.3f')} ({format(Accuracy_std / 6, '.3f')})",
               f"{format(Sensitivity_mean / 6, '.3f')} ({format(Sensitivity_std / 6, '.3f')})",
               f"{format(Specificity_mean / 6, '.3f')} ({format(Specificity_std / 6, '.3f')})",
               f"{format(PPV_mean / 6, '.3f')} ({format(PPV_std / 6, '.3f')})",
               f"{format(NPV_mean / 6, '.3f')} ({format(NPV_std / 6, '.3f')})"]]
results_df = pd.DataFrame(results_df)
results_df.columns = ['AUROC', 'AUPRC', 'Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV']
results_df.to_csv("results_res_18_cls_aggregation.csv", index=False, encoding="cp1252")

deit_results = pd.read_csv("results_deit_tiny_cls.csv")
AUROC_mean = 0
AUROC_std = 0
AUPRC_mean = 0
AUPRC_std = 0
Accuracy_mean = 0
Accuracy_std = 0
Sensitivity_mean = 0
Sensitivity_std = 0
Specificity_mean = 0
Specificity_std = 0
PPV_mean = 0
PPV_std = 0
NPV_mean = 0
NPV_std = 0
for i in range(0, 6):
    AUROC_mean += float(deit_results.iloc[i, 3].split(' (')[0])
    AUROC_std += float(deit_results.iloc[i, 3].split(' (')[1].split(')')[0])

    AUPRC_mean += float(deit_results.iloc[i, 4].split(' (')[0])
    AUPRC_std += float(deit_results.iloc[i, 4].split(' (')[1].split(')')[0])

    Accuracy_mean += float(deit_results.iloc[i, 5].split(' (')[0])
    Accuracy_std += float(deit_results.iloc[i, 5].split(' (')[1].split(')')[0])

    Sensitivity_mean += float(deit_results.iloc[i, 6].split(' (')[0])
    Sensitivity_std += float(deit_results.iloc[i, 6].split(' (')[1].split(')')[0])

    Specificity_mean += float(deit_results.iloc[i, 7].split(' (')[0])
    Specificity_std += float(deit_results.iloc[i, 7].split(' (')[1].split(')')[0])

    PPV_mean += float(deit_results.iloc[i, 8].split(' (')[0])
    PPV_std += float(deit_results.iloc[i, 8].split(' (')[1].split(')')[0])

    NPV_mean += float(deit_results.iloc[i, 9].split(' (')[0])
    NPV_std += float(deit_results.iloc[i, 9].split(' (')[1].split(')')[0])

results_df = [[f"{format(AUROC_mean / 6, '.3f')} ({format(AUROC_std / 6, '.3f')})",
               f"{format(AUPRC_mean / 6, '.3f')} ({format(AUPRC_std / 6, '.3f')})",
               f"{format(Accuracy_mean / 6, '.3f')} ({format(Accuracy_std / 6, '.3f')})",
               f"{format(Sensitivity_mean / 6, '.3f')} ({format(Sensitivity_std / 6, '.3f')})",
               f"{format(Specificity_mean / 6, '.3f')} ({format(Specificity_std / 6, '.3f')})",
               f"{format(PPV_mean / 6, '.3f')} ({format(PPV_std / 6, '.3f')})",
               f"{format(NPV_mean / 6, '.3f')} ({format(NPV_std / 6, '.3f')})"]]
results_df = pd.DataFrame(results_df)
results_df.columns = ['AUROC', 'AUPRC', 'Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV']
results_df.to_csv("results_deit_tiny_cls_aggregation.csv", index=False, encoding="cp1252")

swin_results = pd.read_csv("results_swin_tiny_cls.csv")
AUROC_mean = 0
AUROC_std = 0
AUPRC_mean = 0
AUPRC_std = 0
Accuracy_mean = 0
Accuracy_std = 0
Sensitivity_mean = 0
Sensitivity_std = 0
Specificity_mean = 0
Specificity_std = 0
PPV_mean = 0
PPV_std = 0
NPV_mean = 0
NPV_std = 0
for i in range(0, 6):
    AUROC_mean += float(swin_results.iloc[i, 3].split(' (')[0])
    AUROC_std += float(swin_results.iloc[i, 3].split(' (')[1].split(')')[0])

    AUPRC_mean += float(swin_results.iloc[i, 4].split(' (')[0])
    AUPRC_std += float(swin_results.iloc[i, 4].split(' (')[1].split(')')[0])

    Accuracy_mean += float(swin_results.iloc[i, 5].split(' (')[0])
    Accuracy_std += float(swin_results.iloc[i, 5].split(' (')[1].split(')')[0])

    Sensitivity_mean += float(swin_results.iloc[i, 6].split(' (')[0])
    Sensitivity_std += float(swin_results.iloc[i, 6].split(' (')[1].split(')')[0])

    Specificity_mean += float(swin_results.iloc[i, 7].split(' (')[0])
    Specificity_std += float(swin_results.iloc[i, 7].split(' (')[1].split(')')[0])

    PPV_mean += float(swin_results.iloc[i, 8].split(' (')[0])
    PPV_std += float(swin_results.iloc[i, 8].split(' (')[1].split(')')[0])

    NPV_mean += float(swin_results.iloc[i, 9].split(' (')[0])
    NPV_std += float(swin_results.iloc[i, 9].split(' (')[1].split(')')[0])

results_df = [[f"{format(AUROC_mean / 6, '.3f')} ({format(AUROC_std / 6, '.3f')})",
               f"{format(AUPRC_mean / 6, '.3f')} ({format(AUPRC_std / 6, '.3f')})",
               f"{format(Accuracy_mean / 6, '.3f')} ({format(Accuracy_std / 6, '.3f')})",
               f"{format(Sensitivity_mean / 6, '.3f')} ({format(Sensitivity_std / 6, '.3f')})",
               f"{format(Specificity_mean / 6, '.3f')} ({format(Specificity_std / 6, '.3f')})",
               f"{format(PPV_mean / 6, '.3f')} ({format(PPV_std / 6, '.3f')})",
               f"{format(NPV_mean / 6, '.3f')} ({format(NPV_std / 6, '.3f')})"]]
results_df = pd.DataFrame(results_df)
results_df.columns = ['AUROC', 'AUPRC', 'Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV']
results_df.to_csv("results_swin_tiny_cls_aggregation.csv", index=False, encoding="cp1252")