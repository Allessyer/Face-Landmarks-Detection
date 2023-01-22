from tqdm import tqdm
from tqdm import trange
import sys
from itertools import starmap
import gc

import numpy as np
import torch
import torchlm
import dlib

# my modules
from face_landmarks.metric import CED
from face_landmarks.transforms import *

def train_model(model, 
                num_epochs, 
                train_dataloader, 
                val_dataloader,
                criterion,
                optimizer,
                scheduler,
                device, 
                path2results, 
                model_name,
                exp_name):

    logs = {}

    logs['train_loss'] = []
    logs['train_fr_0_08'] = []
    logs['train_auc_0_08'] = []
    logs['train_fr_1_0'] = []
    logs['train_auc_1_0'] = []

    logs['val_loss'] = []
    logs['val_fr_0_08'] = []
    logs['val_auc_0_08'] = []
    logs['val_fr_1_0'] = []
    logs['val_auc_1_0'] = []

    logs['best_auc'] = 0.0

    for epoch in trange(num_epochs, desc="Epoch",file=sys.stdout):
        
        model, train_avg_loss, train_fr_0_08, train_auc_0_08, train_fr_1_0, train_auc_1_0 = train_singe_epoch(model,
                        train_dataloader, 
                        criterion,
                        optimizer,
                        epoch,
                        device)

        model, val_avg_loss, val_fr_0_08, val_auc_0_08, val_fr_1_0, val_auc_1_0 = evaluate_single_epoch(model,
                        val_dataloader,
                        criterion,
                        epoch, 
                        device)
        
        if scheduler:
            scheduler.step()

        logs['train_loss'].append(train_avg_loss)
        logs['train_fr_0_08'].append(train_fr_0_08)
        logs['train_auc_0_08'].append(train_auc_0_08)
        logs['train_fr_1_0'].append(train_fr_1_0)
        logs['train_auc_1_0'].append(train_auc_1_0)

        logs['val_loss'].append(val_avg_loss)
        logs['val_fr_0_08'].append(val_fr_0_08)
        logs['val_auc_0_08'].append(val_auc_0_08)
        logs['val_fr_1_0'].append(val_fr_1_0)
        logs['val_auc_1_0'].append(val_auc_1_0)

        if logs['best_auc'] < val_auc_1_0:
            logs['best_auc'] = val_auc_1_0
            torch.save(model.state_dict(), f"{path2results}/{model_name}_{exp_name}_model_best_auc.pth")
            print("Best auc model saved at epoch {}".format(epoch))

        torch.save(logs,f'{path2results}/{model_name}_{exp_name}_logs')
        torch.save(model.state_dict(), f"{path2results}/{model_name}_{exp_name}_model_current.pth")

    del logs
    gc.collect()

    return model


def train_singe_epoch(model,
                      train_dataloader, 
                      criterion,
                      optimizer,
                      epoch,
                      device):
    model.train()
    pbar = tqdm(train_dataloader, desc=f'Train (epoch = {epoch})')

    total_loss = 0
    NME = []

    metric = CED()

    for batch in pbar:
        image, true_landmarks, rect = batch

        image = image.to(device)
        true_landmarks = true_landmarks.to(device)

        pred_landmarks = model(image)
        pred_landmarks = pred_landmarks.reshape(-1,68,2)

        loss = criterion(pred_landmarks, true_landmarks)
        total_loss += loss

        image = image.cpu()
        true_landmarks = true_landmarks.cpu()
        pred_landmarks = pred_landmarks.cpu().detach()

        orig_true_landmarks = untransform_batch(image, true_landmarks, rect)
        orig_pred_landmarks = untransform_batch(image, pred_landmarks, rect)

        nme = metric.nme_batch(orig_true_landmarks, 
                        orig_pred_landmarks, 
                        rect.numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        NME = np.concatenate((NME, nme))

    fr_0_08, auc_0_08 = metric.fr_and_auc(NME, thres=0.08)
    fr_1_0, auc_1_0 = metric.fr_and_auc(NME, thres=1.0)

    avg_loss = total_loss / len(train_dataloader)
    print("\nTrain AUC 0.08: {}".format(auc_0_08))
    print("\nTrain FR 0.08: {}".format(fr_0_08))
    print("\nTrain AUC 1.0: {}".format(auc_1_0))
    print("\nTrain FR 1.0: {}".format(fr_1_0))
    print("\nTrain loss: {}".format(avg_loss))

    return model, avg_loss, fr_0_08, auc_0_08, fr_1_0, auc_1_0


def evaluate_single_epoch(model,
                          val_dataloader,
                          criterion,
                          epoch, 
                          device):
    
    model.eval()
    pbar = tqdm(val_dataloader, desc=f'Eval (epoch = {epoch})')

    NME = []
    total_loss = 0

    metric = CED()

    for batch in pbar:
        image, true_landmarks, rect = batch

        image = image.to(device)
        true_landmarks = true_landmarks.to(device)

        with torch.no_grad():
            pred_landmarks = model(image)
            pred_landmarks = pred_landmarks.reshape(-1,68,2)

            loss = criterion(pred_landmarks, true_landmarks)
            total_loss += loss

            image = image.cpu()
            true_landmarks = true_landmarks.cpu()
            pred_landmarks = pred_landmarks.cpu().detach()

            orig_true_landmarks = untransform_batch(image, true_landmarks, rect)
            orig_pred_landmarks = untransform_batch(image, pred_landmarks, rect)

            nme = metric.nme_batch(orig_true_landmarks, 
                            orig_pred_landmarks, 
                            rect.numpy())

            NME = np.concatenate((NME, nme))

    fr_0_08, auc_0_08 = metric.fr_and_auc(NME, thres=0.08)
    fr_1_0, auc_1_0 = metric.fr_and_auc(NME, thres=1.0)

    avg_loss = total_loss / len(val_dataloader)
    print("\nValidation AUC 0.08: {}".format(auc_0_08))
    print("\nValidation FR 0.08: {}".format(fr_0_08))
    print("\nValidation AUC 1.0: {}".format(auc_1_0))
    print("\nValidation FR 1.0: {}".format(fr_1_0))
    print("\nValidation loss: {}".format(avg_loss))

    return model, avg_loss, fr_0_08, auc_0_08, fr_1_0, auc_1_0