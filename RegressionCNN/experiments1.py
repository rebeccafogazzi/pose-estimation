import os
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.getcwd()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split  # to load dataset into batches and to split it randomly
from PIL import Image

from util import *
from tools import *
from main import *

# -----------------------------------------------------------------------------

case_dictionary = {
    # '1': ['phantom', 'train', 'ipcai', 'resnet18_phantom_joint_0.01.pt'],
    # '2': ['real', 'infer', 'ipcai', 'resnet18_phantom_joint_0.01.pt'],
    # '3': ['real', 'train', 'ipcai', 'resnet18_real3_joint_0.01.pt'],
    # '3': ['real', 'infer', 'ipcai', 'resnet18_real3_joint_0.01.pt'],
    # '4': ['real', 'infer', 'ipcai', 'resnet18_real3_joint_0.01.pt'],
    # '5': ['real', 'train', 'ipcai', 'resnet18_real5_joint_0.01.pt'],
    # '5': ['real', 'infer', 'ipcai', 'resnet18_real5_joint_0.01.pt'],
    # '5': ['real', 'train', 'ipcai', 'resnet18_ipcai.pt'],
    # '5': ['real', 'infer', 'ipcai', 'resnet18_ipcai.pt'],
    # '1': ['phantom', 'train', 'ijcars', 'resnet18_ijcars_phantom.pt'],
    # '5': ['real', 'train', 'ijcars', 'resnet18_ijcars_real.pt'],
    # '5': ['real', 'infer', 'ijcars', 'resnet18_ijcars_real.pt'],
    # '1': ['phantom', 'train', 'ijcars', 'resnet18_da_phantom.pt'],
    # '5': ['real', 'train', 'ijcars', 'resnet18_da_real.pt'],
    # '1': ['phantom', 'train', 'ijcars', 'resnet50_phantom.pt'],
    # '2': ['real', 'infer', 'ijcars', 'resnet50_phantom.pt'],
    # '3': ['real', 'train', 'ijcars', 'resnet50_real3.pt'],
    # '4': ['real', 'infer', 'ijcars', 'resnet50_real3.pt'],
    '5': ['real', 'train', 'ijcars', 'resnet50_real5.pt'],
}


def experiment_1(case, gt_train_filename, pred_train_filename,
                 gt_test_filename, pred_test_filename, res_train_filename,
                 res_test_filename):

    num_subsets=None

    key_values = case_dictionary[case]
    data_type = key_values[0]
    training_mode = key_values[1]
    data_folder = key_values[2]
    out_model = key_values[3]

    out_folder = 'resnet50'
    
    lambda_val = 0.01

    figures_path = rf'output/{out_folder}/figures/experiment1/case{case}'
    output_path = rf'output/{out_folder}/files/experiment1/case{case}'

    isExist_figs = os.path.exists(figures_path)
    isExist_files = os.path.exists(output_path)
    if not isExist_figs:  
        os.makedirs(figures_path)
    if not isExist_files:  
        os.makedirs(output_path)
        
    labels_train, labels_test, scaler = prepare_labels(
        rf'data/{data_folder}/labels/unity/train{case}.csv',
        rf'data/{data_folder}/labels/train{case}.csv',
        rf'data/{data_folder}/labels/unity/test{case}.csv',
        rf'data/{data_folder}/labels/test{case}.csv')

    if case == '2':
        labels_train, scaler_train = prepare_labels_train(
            rf'data/{data_folder}/labels/unity/train{case}.csv',
            rf'data/{data_folder}/labels/train2.csv')
        _, scaler_test = prepare_labels_train(
            rf'data/{data_folder}/labels/unity/train{case}.csv',
            rf'data/{data_folder}/labels/train{case}.csv')
        labels_test = prepare_labels_test(
            rf'data/{data_folder}/labels/unity/test2.csv',
            rf'data/{data_folder}/labels/test2.csv', scaler_test)
        scaler = scaler_test

    if data_type == 'phantom':
        print('Experiments on phantom data\n')
    else:
        print('Experiments on real data\n')
    
    dataset_train = FetalDataset(labels_train, 
        rf'data/{data_folder}/planes/train{case}/', 
        data_transforms[f'{data_type}'])
    dataset_test = FetalDataset(labels_test, 
        rf'data/{data_folder}/planes/test{case}/', 
        data_transforms[f'{data_type}'])

    train_dataset, valid_dataset, test_dataset = split(dataset_train, dataset_test)

    train_loader, valid_loader, test_loader = data_loaders(dataset_train=train_dataset,dataset_valid=valid_dataset,dataset_test=test_dataset,subset=num_subsets)

    print(f"\n>>> Data loaded: case {case}\n")

    if training_mode == 'train':

        print('Training new model\n')

        if data_type == 'phantom':
            input_model = None
            print('No model loaded')
            output_model = out_model
        else:
            # input_model = 'resnet18_phantom_joint_0.01.pt'
            # input_model = 'resnet18_ijcars_phantom.pt'
            input_model = 'resnet50_phantom.pt'
            print(f'Loaded {input_model} model')
            output_model = out_model
            # output_model = f'resnet18_real{case}_joint_0.01.pt'
        
        saved_model = train(train_loader, valid_loader, figures_path, output_path, input_model, output_model, weighting_factor=lambda_val)
        transl_train, transl_test, geod_train, geod_test = evaluate(train_loader, test_loader, scaler, figures_path, output_path,saved_model, gt_train_filename, pred_train_filename,gt_test_filename, pred_test_filename, res_train_filename,res_test_filename, data_type)
    else:
        saved_model = out_model
        print(f'Loading existing model: {saved_model}\n')
        transl_train, transl_test, geod_train, geod_test = evaluate(train_loader, test_loader, scaler, figures_path, output_path,saved_model, gt_train_filename, pred_train_filename, gt_test_filename, pred_test_filename, res_train_filename, res_test_filename, data_type)

    # if case == '1' or case == '5':

    #     labels = prepare_labels_test(
    #         rf'data/{data_folder}/labels/unity/{data_type}_eval.csv',
    #         rf'data/{data_folder}/labels/{data_type}_eval.csv', scaler)

    #     data_path = rf'data/{data_folder}/planes/evaluation_{data_type}/'
    #     out_path = rf'output/{data_folder}/files/experiment3/case{case}'
    #     svd_model = f'resnet18_{data_type}_0.01.pt'
        
    #     dataset = FetalDataset(labels, data_path, data_transforms[f'{data_type}'])
    #     data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    #     print(f"\n>>> Data loaded: case {case}, exp 3\n")

    #     start = time.time()

    #     inference_sp(data_loader, scaler, out_path, svd_model, 'gt_eval.csv','pred_eval.csv', 'res_eval.txt', data_type)

    #     end = time.time()
    #     elapsed_time = (end - start) / 60

    #     print('Elapsed time per plane: ', elapsed_time / 2)

    return  transl_train, transl_test, geod_train, geod_test, figures_path


for case in case_dictionary.keys():
    transl_train, transl_test, geod_train, geod_test, figures_path = experiment_1(case, 'gt_train.csv', 'pred_train.csv', 'gt_test.csv', 'pred_test.csv', 'res_train.txt', 'res_test.txt')
    save_plots(transl_train, transl_test, geod_train, geod_test, figures_path)