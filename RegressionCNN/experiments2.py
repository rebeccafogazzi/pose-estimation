import os
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.getcwd()

import numpy as np
from torch import nn
from torch.utils.data import random_split  # to load dataset into batches and to split it randomly

from util import *
from tools import *
from main import *

# -----------------------------------------------------------------------------

def experiment_2(case, plane_type):

    key_values = case_dictionary[case]
    data_type = key_values[0]
    saved_model = key_values[1]
    data_folder = key_values[2]
    conversion = key_values[3]

    num_subsets = None

    figures_path = rf'output/{data_folder}/figures/experiment2/case{case}'
    output_path = rf'output/{data_folder}/files/experiment2/case{case}'

    isExist_figs = os.path.exists(figures_path)
    isExist_files = os.path.exists(output_path)
    if not isExist_figs:  
        os.makedirs(figures_path)
    if not isExist_files:  
        os.makedirs(output_path)

    labels_train, labels_test, scaler = prepare_labels(
        rf'data/{data_folder}/labels/unity/train{case}.csv',
        rf'data/{data_folder}/labels/train{case}.csv',
        rf'data/{data_folder}/labels/unity/test{case}_{plane_type}.csv',
        rf'data/{data_folder}/labels/test{case}_{plane_type}.csv')

    if case == '2':
        labels_train, scaler_train = prepare_labels_train(
            rf'data/{data_folder}/labels/unity/train{case}.csv',
            rf'data/{data_folder}/labels/train{case}.csv')
        _, scaler_test = prepare_labels_train(
            rf'data/{data_folder}/labels/unity/train5.csv',
            rf'data/{data_folder}/labels/train5.csv')
        labels_test = prepare_labels_test(
            rf'data/{data_folder}/labels/unity/test{case}_{plane_type}.csv',
            rf'data/{data_folder}/labels/test{case}_{plane_type}.csv',
            scaler_test)
        scaler = scaler_test

    if data_type == 'phantom':
        print('Experiments on phantom data\n')
    else:
        print('Experiments on real data\n')
        
    dataset_train = FetalDataset(labels_train, rf'data/{data_folder}/planes/train{case}/', data_transforms[f'{data_type}'])
    dataset_test = FetalDataset(labels_test, rf'data/{data_folder}/planes/test{case}_{plane_type}', data_transforms[f'{data_type}'])

    train_dataset, valid_dataset, test_dataset = split(dataset_train, dataset_test)

    train_loader, _, test_loader = data_loaders(train_dataset, valid_dataset, test_dataset, subset=num_subsets)

    print(f"\n>>> Data loaded: case {case}\n")

    transl_train, transl_test, geod_train, geod_test = evaluate(train_loader, test_loader, scaler, figures_path, output_path, saved_model, gt_train_filename=f'gt_train_case{case}_{plane_type}.csv',pred_train_filename=f'pred_train_case{case}_{plane_type}.csv',gt_test_filename=f'gt_test_case{case}_{plane_type}.csv',pred_test_filename=f'pred_test_case{case}_{plane_type}.csv',res_train_filename=f'res_train_case{case}_{plane_type}.txt',res_test_filename=f'res_test_case{case}_{plane_type}.txt',data_type=data_type)

    return transl_train, transl_test, geod_train, geod_test, figures_path

# -----------------------------------------------------------------------------

case_dictionary = {
    # '1': ['phantom', 'resnet18_phantom_joint_0.01.pt', 'ipcai', 'false'],
    # '1': ['phantom', 'resnet18_ijcars_phantom.pt', 'ipcai', 'false'],
    # '2': ['real', 'resnet18_phantom_joint_0.01.pt', 'ipcai', 'false'],
    # '3': ['real', 'resnet18_real3_joint_0.01.pt', 'ipcai', 'true'],
    # '4': ['real', 'resnet18_real3_joint_0.01.pt', 'ipcai', 'true'],
    # '5': ['real', 'resnet18_ijcars_0.01.pt', 'ijcars', 'false'],
    # '1': ['phantom', 'resnet18_ijcars_phantom.pt', 'ijcars', 'false'],
    # '5': ['real', 'resnet18_ijcars_real.pt', 'ijcars', 'false'],
    '1': ['phantom', 'resnet50_phantom.pt', 'ijcars', 'false'],
    '5': ['real', 'resnet50_real5.pt', 'ijcars', 'false'],
}

for case in case_dictionary.keys():

    transl_train_rp, transl_test_rp, geod_train_rp, geod_test_rp, figures_path = experiment_2(case, 'rp')
    transl_train_sp, transl_test_sp, geod_train_sp, geod_test_sp, figures_path = experiment_2(case, 'sp')

    boxplot(data_to_plot_list=[transl_train_rp, transl_test_rp, transl_test_sp], boxColors_list=[color_train,color_test,color_test_sp], labels_list=['Train', 'Test RP', 'Test SP'], label_y='Euclidean error [mm]', figtitle='Translation errors - Boxplots', figname='transl_boxplot.png', figpath=figures_path, type='translation')

    boxplot(data_to_plot_list=[geod_train_rp, geod_test_rp, geod_test_sp],boxColors_list=[color_train, color_test, color_test_sp], labels_list=['Train', 'Test RP', 'Test SP'], label_y='Geodesic error [degrees]', figtitle='Rotation errors - Boxplots', figname='rot_boxplot.png', figpath=figures_path, type='rotation')


# -----------------------------------------------------------------------------
# Weekly Evaluation
# -----------------------------------------------------------------------------

def experiment_2_weekly(week, saved_model, plane_type):

    key_value = week_dictionary[week]
    data_type = key_value[0]
    data_folder = key_value[1]
    
    out_folder = 'resnet50'

    num_subsets = None

    figures_path = rf'output/{out_folder}/figures/experiment2/weekly/{week}w'
    output_path = rf'output/{out_folder}/files/experiment2/weekly/{week}w'

    print('Experiments on real data\n')
    
    labels_train_week, labels_test_week, scaler_week = prepare_labels(rf'data/{data_folder}/labels/unity/train5.csv', rf'data/{data_folder}/labels/train5.csv', rf'data/{data_folder}/labels/unity/{week}w_{plane_type}.csv',  rf'data/{data_folder}/labels/{week}w_{plane_type}.csv')

    dataset_train_week = FetalDataset(labels_train_week, rf'data/{data_folder}/planes/train5/', data_transforms[f'{data_type}'])

    dataset_test_week = FetalDataset(labels_test_week, rf'data/{data_folder}/planes/{week}w_{plane_type}/', data_transforms[f'{data_type}'])

    print(rf'data/{data_folder}/planes/{week}w_{plane_type}/')

    train_dataset_week, valid_dataset_week, test_dataset_week = split(dataset_train_week, dataset_test_week)

    train_loader_week, _, test_loader_week = data_loaders(dataset_train=train_dataset_week, dataset_valid=valid_dataset_week,dataset_test=test_dataset_week, subset=num_subsets)

    transl_train_week, transl_test_week, geod_train_week, geod_test_week = evaluate(train_loader=train_loader_week, test_loader=test_loader_week,scaler=scaler_week, figures_path=figures_path, output_path=output_path,saved_model=saved_model, gt_train_filename=f'gt_train_{week}w_{plane_type}.csv', pred_train_filename=f'pred_train_{week}w_{plane_type}.csv', gt_test_filename=f'gt_test_{week}w_{plane_type}.csv', pred_test_filename=f'pred_test_{week}w_{plane_type}.csv',res_train_filename=f'res_train_{week}w_{plane_type}.txt',res_test_filename=f'res_test_{week}w_{plane_type}.txt',data_type=data_type)

    pd.DataFrame(transl_train_week).to_csv(os.path.join(output_path, f'transl_train_err_{week}w.csv'), index=False, header=False)
    pd.DataFrame(geod_train_week).to_csv(os.path.join(output_path, f'geod_train_err_{week}w.csv'), index=False, header=False)
    pd.DataFrame(transl_test_week).to_csv(os.path.join(output_path, f'transl_test_{week}w_{plane_type}_err.csv'), index=False, header=False)
    pd.DataFrame(geod_test_week).to_csv(os.path.join(output_path, f'geod_test_{week}w_{plane_type}_err.csv'), index=False, header=False)

    return transl_train_week, transl_test_week, geod_train_week, geod_test_week, figures_path

# -----------------------------------------------------------------------------

week_dictionary = {
    # '21': ['real','ipcai'],
    # '22': ['real','ipcai'],
    # '23': ['real','ipcai'],
    # '24': ['real','ipcai'],
    # '26': ['real','ipcai'],
    # '39': ['real','ipcai'],
    '21': ['real','ijcars'],
    '22': ['real','ijcars'],
    '23': ['real','ijcars'],
    '24': ['real','ijcars'],
    '25': ['real','ijcars'],
    '39': ['real','ijcars']
}

# saved_model = 'resnet18_real5_joint_0.01.pt'
# # saved_model = 'resnet18_ipcai.pt'
# fig_path = r'output/ipcai/figures/experiment2/weekly/360'
# file_path = r'output/ipcai/files/experiment2/weekly/360'

# saved_model = 'resnet18_ijcars.pt'
# saved_model = 'resnet18_da_real.pt'
# fig_path = r'output/da/figures/experiment2/weekly'
# file_path = r'output/da/files/experiment2/weekly'

saved_model = 'resnet50_real5.pt'
fig_path = r'output/resnet50/figures/experiment2/weekly'
file_path = r'output/resnet50/files/experiment2/weekly'

geod_test_week_list_rp = []
transl_test_week_list_rp = []
geod_test_week_list_sp = []
transl_test_week_list_sp = []

for week in week_dictionary.keys():

    transl_train_week_rp, transl_test_week_rp, geod_train_week_rp, geod_test_week_rp, figures_path = experiment_2_weekly(week, saved_model, 'rp')

    boxplot(data_to_plot_list=[transl_train_week_rp, transl_test_week_rp],boxColors_list=[color_train, color_test], labels_list=['Train', 'Test'], label_y='Euclidean error [mm]', figtitle='Translation errors - Boxplots', figname='transl_boxplot.png', figpath=figures_path,type='translation')

    boxplot(data_to_plot_list=[geod_train_week_rp, geod_test_week_rp],boxColors_list=[color_train, color_test], labels_list=['Train', 'Test'], label_y='Geodesic error [degrees]', figtitle='Rotation errors - Boxplots', figname='rot_boxplot.png', figpath=figures_path,type='rotation')

    # if week == '23':

    #     transl_train_week_sp, transl_test_week_sp, geod_train_week_sp, geod_test_week_sp, figures_path = experiment_2_weekly(week, saved_model, 'sp')
    
    #     boxplot(data_to_plot_list=[transl_train_week_rp, transl_test_week_rp, transl_test_week_sp], boxColors_list=[color_train, color_test, color_test_sp], labels_list=['Train', 'Test RP', 'Test SP'], label_y='Euclidean error [mm]', figtitle='Translation errors - Boxplots', figname='transl_boxplot_complete.png',figpath=figures_path, type='translation')

    #     boxplot(data_to_plot_list=[geod_train_week_rp, geod_test_week_rp, geod_test_week_sp], boxColors_list=[color_train, color_test, color_test_sp], labels_list=['Train', 'Test RP', 'Test SP'],label_y='Geodesic error [degrees]', figtitle='Rotation errors - Boxplots', figname='rot_boxplot_complete.png', figpath=figures_path, type='rotation')

    transl_test_week_list_rp.append(transl_test_week_rp)
    geod_test_week_list_rp.append(geod_test_week_rp)
    # transl_test_week_list_sp.append(transl_test_week_sp)
    # geod_test_week_list_sp.append(geod_test_week_sp)


boxplot(
    data_to_plot_list=[transl_train_week_rp] + transl_test_week_list_rp,
    boxColors_list=[color_train, color_test, color_test, color_test, color_test,color_test,color_test],
    labels_list=['23w', '21w', '22w', '23w', '24w','25w','39w'],
    label_y='Euclidean error [mm]',
    figtitle='Translation errors - Boxplots',
    figname='transl_boxplot_weekly_rp.png',
    figpath=fig_path,
    type='translation')

boxplot(
    data_to_plot_list=[geod_train_week_rp] + geod_test_week_list_rp,
    boxColors_list=[color_train, color_test, color_test, color_test, color_test,color_test,color_test],
    labels_list=['23w', '21w', '22w', '23w', '24w','25w','39w'],
    label_y='Geodesic error [degrees]',
    figtitle='Rotation errors - Boxplots',
    figname='rot_boxplot_weekly_rp.png',
    figpath=fig_path,
    type='rotation')

# # boxplot(
# #     data_to_plot_list=[transl_train_week_rp] + transl_test_week_list_rp +
# #     transl_test_week_list_sp,
# #     # boxColors_list=[color_train, color_test, color_test, color_test, color_test, color_test, color_test_sp, color_test_sp, color_test_sp, color_test_sp, color_test_sp],
# #     boxColors_list=[
# #         color_train, color_test, color_test, color_test, color_test,
# #         color_test, color_test_sp, color_test_sp, color_test_sp, color_test_sp,
# #         color_test_sp],
# #     # labels_list=['23w', '21w', '22w', '23w', '26w', '39w', '21w', '22w', '23w', '26w', '39w'],
# #     labels_list=[
# #         '23w', '21w', '22w', '23w', '24w', '26w', '21w', '22w', '23w', '24w',
# #         '26w'
# #     ],
# #     label_y='Euclidean error [mm]',
# #     figtitle='Translation errors - Boxplots',
# #     figname='transl_boxplot_weekly_rp.png',
# #     figpath=fig_path,
# #     type='translation')

# # boxplot(
# #     data_to_plot_list=[geod_train_week_rp] + geod_test_week_list_rp +
# #     geod_test_week_list_sp,
# #     # boxColors_list=[color_train, color_test, color_test, color_test, color_test, color_test, color_test_sp, color_test_sp, color_test_sp, color_test_sp, color_test_sp],
# #     boxColors_list=[color_train, color_test, color_test, color_test,color_test, color_test,
# #                     color_test_sp, color_test_sp, color_test_sp, color_test_sp,color_test_sp],
# #     # labels_list=['23w', '21w', '22w', '23w', '26w', '39w', '21w', '22w', '23w', '26w', '39w'],
# #     labels_list=['23w', '21w', '22w', '23w', '24w', '26w', '21w', '22w', '23w', '24w', '26w'],
# #     label_y='Geodesic error [degrees]',
# #     figtitle='Rotation errors - Boxplots',
# #     figname='rot_boxplot_weekly.png',
# #     figpath=fig_path,
# #     type='rotation')

#     # dictionary = {f'transl_train_week_rp_{week}': transl_train_week_rp,
#     #               f'transl_test_week_rp_{week}': transl_test_week_rp,
#     #               f'geod_train_week_rp_{week}': geod_train_week_rp,
#     #             #   f'geod_test_week_rp_{week}': geod_test_week_rp,
#     #             #   f'transl_train_week_sp_{week}': transl_train_week_sp,
#     #             #   f'transl_test_week_sp_{week}': transl_test_week_sp,
#     #             #   f'geod_train_week_sp_{week}': geod_train_week_sp,
#     #             #   f'geod_test_week_sp_{week}': geod_test_week_sp
#     # }

#     # # df = pd.DataFrame.from_dict(dictionary, orient='index')
#     # # df = df.transpose()

#     # # df.to_csv(os.path.join(file_path,f'debug_prints_{week}.csv'), index=False)