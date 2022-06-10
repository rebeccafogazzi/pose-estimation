from __future__ import print_function, division, absolute_import
import os
import argparse
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torchsummary import summary
from tqdm import tqdm
import time
import hiddenlayer as hl
from torchviz import make_dot
from scipy.spatial.transform import Rotation as R

from util import *
from model import *
from tools import *

# -----------------------------------------------------------------------------

def train(train_loader, valid_loader, figures_path, output_path, input_model, output_model, weighting_factor):

    # -------------------------------------------------------------------------
    # CNN
    # -------------------------------------------------------------------------

    model = ResNet().to(device)
    # print(model)
    # summary(model, input_size=(3, img_size[0], img_size[1]))

    if input_model is not None:
        model.load_state_dict(torch.load(input_model))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1,last_epoch=-1)

    criterion = nn.MSELoss().to(device)

    # Removes Constant nodes from graph
    # transforms = [hl.transforms.Prune('Constant')]  
    # graph = hl.build_graph(model, train_images.to(device), transforms=transforms)
    # graph.theme = hl.graph.THEMES['blue'].copy()
    # graph.save(os.path.join(figures_path, 'cnn_hiddenlayer'), format='png')

    # out = model(train_images.to(device))
    # make_dot(out).render(os.path.join(figures_path, 'cnn_blocks'), format='png')
    
    # -------------------------------------------------------------------------
    
    def iter_dataloader(data_loader,model,training):
    
        running_loss_transl = 0.0
        running_err_transl = np.array([])
        running_loss_rot = 0.0
        running_err_rot = 0.0
        running_loss = 0.0

        for image_batch, label_batch in data_loader:

            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            label_batch_transl = label_batch[:, :3]
            label_batch_rot = label_batch[:, -3:]

            label_batch_rotmat = R.from_euler('xyz', label_batch_rot.detach().cpu())
            label_batch_rotmat = label_batch_rotmat.as_matrix()

            if training == True:
                optimizer.zero_grad()

            output_batch_transl, output_batch_rotmat = model(image_batch)

            loss_transl = criterion(output_batch_transl.float(), label_batch_transl.float())
            running_loss_transl += loss_transl.item() * image_batch.size(0)
            error_transl = ((output_batch_transl.float().detach().cpu()-label_batch_transl.float().detach().cpu())**2).mean(axis=1)
            running_err_transl = np.append(running_err_transl, error_transl)

            loss_rot = model.compute_rotation_matrix_l2_loss(output_batch_rotmat.float(), torch.Tensor(label_batch_rotmat).to(device))
            running_loss_rot += loss_rot.item() * image_batch.size(0)
            running_err_rot += model.compute_rotation_matrix_geodesic_loss(output_batch_rotmat.float(), torch.Tensor(label_batch_rotmat).to(device))

            loss_tot = loss_rot + weighting_factor * loss_transl
            running_loss += loss_tot.item() * image_batch.size(0)

            if training == True:
                loss_tot.backward()
                optimizer.step()

        # average loss over a single set of inputs (accumulated loss divided by the total number of images in the training dataset)
        loss_transl = running_loss_transl / len(data_loader.dataset)
        err_transl = running_err_transl / len(data_loader)
        loss_rot = running_loss_rot / len(data_loader.dataset)
        err_rot = running_err_rot / len(data_loader)
        loss = running_loss / len(data_loader.dataset)

        return loss_transl, err_transl, loss_rot, err_rot, loss
    
    # -------------------------------------------------------------------------

    def training(model, train_loader):

        model.train()

        loss_transl, err_transl, loss_rot, err_rot, loss_train = iter_dataloader(train_loader, model, training=True)

        return loss_transl, err_transl, loss_rot, err_rot, loss_train

    # -------------------------------------------------------------------------

    def testing(model, valid_loader):

        model.eval()

        with torch.no_grad():

            loss_transl, err_transl, loss_rot, err_rot, loss_valid = iter_dataloader(valid_loader, model, training=False)

        return loss_transl, err_transl, loss_rot, err_rot, loss_valid

    # -------------------------------------------------------------------------

    train_losses_transl = []
    train_errors_transl = []
    train_losses_rot = []
    train_errors_rot = []
    train_losses = []
    valid_losses_transl = []
    valid_errors_transl = []
    valid_losses_rot = []
    valid_errors_rot = []
    valid_losses = []
    valid_loss_min = np.inf

    # -------------------------------------------------------------------------

    print('\n>>> Training\n')

    start = time.time()
    model_idx = 0

    for epoch in range(1, num_epochs + 1):

        # scheduler.step()
        # print(scheduler.get_lr())

        loss_transl_train, err_transl_train, loss_rot_train, err_rot_train, loss_epoch_train = training(model, train_loader)
        loss_transl_valid, err_transl_valid, loss_rot_valid, err_rot_valid, loss_epoch_valid = testing(model, valid_loader)

        train_losses_transl.append(loss_transl_train)
        train_errors_transl.append(err_transl_train)
        train_losses_rot.append(loss_rot_train)
        train_errors_rot.append(err_rot_train)
        train_losses.append(loss_epoch_train)

        valid_losses_transl.append(loss_transl_valid)
        valid_errors_transl.append(err_transl_valid)
        valid_losses_rot.append(loss_rot_valid)
        valid_errors_rot.append(err_rot_valid)
        valid_losses.append(loss_epoch_valid)

        print()
        print('Translation')
        print(f'Epoch: {epoch}/{num_epochs} \tTrain Loss: {loss_transl_train:.6f} \tValid Loss: {loss_transl_valid:.6f}')
        print()
        print('Rotation')
        print(f'Epoch: {epoch}/{num_epochs} \tTrain Loss: {loss_rot_train:.6f} \tTrain Error: {err_rot_train * 180 / np.pi:.6f} \tValid Loss: {loss_rot_valid:.6f} \tValid Error: {err_rot_valid * 180 / np.pi:.6f}')
        print()

        if loss_epoch_valid < valid_loss_min:
            valid_loss_min = loss_epoch_valid
            torch.save(model.state_dict(), output_model)
            model_idx += 1
            print(f'Minimum Validation Loss of {valid_loss_min:.6f} at epoch {epoch}/{num_epochs}\n')

        print('-------------------------------------------------------------------------------\n')

    end = time.time()
    elapsed_time = (end - start) / 60

    print(f'>>> Training Complete: {elapsed_time:.2f} minutes\n')

    visualize_losses(loss_train=train_losses_transl,loss_valid=valid_losses_transl, label_y='Mean Squared Error',figure_path=figures_path, figtitle = 'Translation Loss',figname='translation_loss.png')

    visualize_losses(loss_train=train_losses_rot, loss_valid=valid_losses_rot,label_y='Mean Squared Error', figure_path=figures_path,figtitle='Rotation Loss', figname='rotation_loss.png')

    visualize_losses(loss_train=train_losses, loss_valid=valid_losses,label_y='Mean Squared Error', figure_path=figures_path,figtitle='Total Loss', figname='total_loss.png')

    visualize_losses_comparison(train_losses_transl, train_losses_rot, valid_losses_transl, valid_losses_rot, figures_path,'losses_comparisons.png')
    
    # -------------------------------------------------------------------------

    print('\n>>> Completed')
    print(f'Last saved model at epoch {model_idx}')

    return output_model

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def compute_errors(data_loader, model, scaler, data_type):
    
    labels_transl_list, labels_rot_list = [], []
    out_transl_list, out_angles_list = [], []
    geodesic_errors_list = np.array([])
    transl_errors_list = np.array([])
    
    model.eval()
    
    with torch.no_grad():

        # for image_batch, label_batch in tqdm(data_loader):
        for image_batch, label_batch in data_loader:

            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            label_batch_transl = label_batch[:, :3]
            label_batch_rot = label_batch[:, -3:]
            
            label_batch_rotmat = R.from_euler('xyz', label_batch_rot.detach().cpu())
            label_batch_rotmat = label_batch_rotmat.as_matrix()
            
            output_batch_transl, output_batch_rotmat = model(image_batch)

            out_angles = R.from_matrix(output_batch_rotmat.detach().cpu())
            out_angles = out_angles.as_euler('xyz')

            labels_transl_list.append(label_batch_transl.detach().cpu().numpy())
            out_transl_list.append(output_batch_transl.detach().cpu().numpy())
            labels_rot_list.append(label_batch_rot.detach().cpu().numpy())
            out_angles_list.append(out_angles)

            _, _, err_mm = compute_metrics(labels_list=label_batch_transl.detach().cpu().numpy(), predictions_list=output_batch_transl.detach().cpu().numpy(), scaler=scaler, data_type = data_type)

            geodesic_errors = np.array(compute_geodesic_distance_from_two_matrices(output_batch_rotmat.float(), torch.Tensor(label_batch_rotmat).to(device)).data.tolist())
            geodesic_errors = geodesic_errors * 180 / np.pi 
            
            transl_errors_list = np.append(transl_errors_list, err_mm)
            geodesic_errors_list = np.append(geodesic_errors_list, geodesic_errors)

            # geodesic_error_order = np.argsort(np.array(geodesic_errors_train))  # indexs of to sort the array
            # geodesic_error_order_list = np.append(geodesic_error_order_list, geodesic_error_order)

    return labels_transl_list, out_transl_list, labels_rot_list, out_angles_list, transl_errors_list, geodesic_errors_list


def evaluate(train_loader, test_loader, scaler, figures_path, output_path, saved_model, gt_train_filename, pred_train_filename, gt_test_filename, pred_test_filename, res_train_filename, res_test_filename, data_type):

    train_res = open(os.path.join(output_path, res_train_filename), 'w')
    test_res = open(os.path.join(output_path, res_test_filename), 'w')

    model = ResNet().to(device)

    if saved_model is not None:
        model.load_state_dict(torch.load(saved_model))
        print(f'Loaded {saved_model} model')
    else:
        print('Error: model not loaded')

    # print('------------------------------------')
    # print('>>> Starting Evaluation on Train set')
    # print('------------------------------------\n')

    labels_transl_list_train, out_transl_list_train, labels_rot_list_train, out_angles_list_train, transl_errors_list_train, geodesic_errors_list_train = compute_errors(train_loader, model, scaler, data_type)

    save_results(transl_errors_list_train, geodesic_errors_list_train, train_res)
    save_files(labels_transl_list_train, out_transl_list_train, scaler, labels_rot_list_train, out_angles_list_train, gt_train_filename, pred_train_filename, output_path)
    save_errors(transl_errors_list_train, geodesic_errors_list_train, 'transl_err_train.csv', 'rot_err_train.csv', output_path)
    
    print('>>> Evaluation on Train set completed')

    # -------------------------------------------------------------------------

    # print('\n-----------------------------------')
    # print('>>> Starting Evaluation on Test set')
    # print('-----------------------------------\n')

    labels_transl_list_test, out_transl_list_test, labels_rot_list_test, out_angles_list_test, transl_errors_list_test, geodesic_errors_list_test = compute_errors(test_loader, model, scaler, data_type)

    save_results(transl_errors_list_test, geodesic_errors_list_test, test_res)
    save_files(labels_transl_list_test, out_transl_list_test, scaler, labels_rot_list_test, out_angles_list_test, gt_test_filename, pred_test_filename, output_path)
    save_errors(transl_errors_list_test, geodesic_errors_list_test, 'transl_err_test.csv', 'rot_err_test.csv', output_path)
    
    print('>>> Evaluation on Test set completed')

    print('\n>>> Evaluation completed\n')

    return transl_errors_list_train, transl_errors_list_test, geodesic_errors_list_train, geodesic_errors_list_test

# -----------------------------------------------------------------------------


def save_plots(transl_err_train, transl_err_test, geod_err_train, geod_err_test, figures_path):
    
    boxplot(data_to_plot_list=[transl_err_train, transl_err_test], 
            boxColors_list=[color_train, color_test], 
            labels_list=['Train', 'Test'], 
            label_y='Euclidean error [mm]', figtitle='Translation errors - Boxplots', 
            figname='transl_boxplot.png', figpath=figures_path, type='translation')
    
    boxplot(data_to_plot_list=[geod_err_train, geod_err_test], 
            boxColors_list=[color_train, color_test],
            labels_list=['Train', 'Test'],
            label_y='Geodesic error [degrees]', figtitle='Rotation errors - Boxplots', 
            figname='rot_boxplot.png', figpath=figures_path, type='rotation')


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def inference_generalisation(dataset_loader, output_path, filename, saved_model):

    model = ResNet().to(device)

    if saved_model is not None:
        model.load_state_dict(torch.load(saved_model))
        print(f'Loaded {saved_model} model')
    else:
        print('Error: model not found')

    with torch.no_grad():

        out_transl_list_eval, out_angles_list_eval = [], []

        model.eval()

        for image_batch, label_batch in tqdm(dataset_loader):

            image_batch = image_batch.to(device)
            output_batch_transl, output_batch_rot = model(image_batch)

            out_angles_eval = R.from_matrix(output_batch_rot.detach().cpu())
            out_angles_eval = out_angles_eval.as_euler('xyz')

            out_transl_list_eval.append(output_batch_transl.detach().cpu().numpy())
            out_angles_list_eval.append(out_angles_eval)

    save_files_generalization(out_transl_list_eval, out_angles_list_eval, filename, output_path)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def inference_sp(dataset_loader, scaler, output_path, saved_model, gt_filename, pred_filename, err_filename, data_type):
    
    evaluation_res = open(os.path.join(output_path, err_filename), 'w')

    model = ResNet().to(device)

    if saved_model is not None:
        model.load_state_dict(torch.load(saved_model))
        print(f'Loaded {saved_model} model')
    else:
        print('Error: model not found')
        
    model.eval()

    with torch.no_grad():

        labels_transl_list_eval, out_transl_list_eval, labels_rot_list_eval, out_angles_list_eval, transl_errors_list_eval, geodesic_errors_list_eval = compute_errors(dataset_loader, model, scaler, data_type)

    save_results(transl_errors_list_eval, geodesic_errors_list_eval, evaluation_res)
    save_files(labels_transl_list_eval, out_transl_list_eval, scaler, labels_rot_list_eval, out_angles_list_eval, gt_filename, pred_filename, output_path)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# def save_model(weights, whole_model):

#     model = ResNet().to(device)
#     model.load_state_dict(torch.load(weights))
#     torch.save(model, whole_model)

# def load_model(whole_model):

#     model = torch.load(whole_model)
#     model.eval()