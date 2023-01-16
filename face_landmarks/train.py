from tqdm import tqdm
from tqdm import trange
import sys
from itertools import starmap
import gc

import numpy as np
import torch
import torchlm


# my modules
from face_landmarks.metric import CED

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
    logs['train_auc'] = []

    logs['val_loss'] = []
    logs['val_auc'] = []
    logs['best_auc'] = 0.0


    for epoch in trange(num_epochs, desc="Epoch",file=sys.stdout):
        
        model, train_loss, train_auc = train_singe_epoch(model,
                        train_dataloader, 
                        criterion,
                        optimizer,
                        epoch,
                        device)

        model, val_loss, val_auc = evaluate_single_epoch(model,
                        val_dataloader,
                        criterion,
                        epoch, 
                        device)
            
        scheduler.step()

        logs['train_loss'].append(train_loss)
        logs['val_loss'].append(val_loss)

        logs['train_auc'].append(train_auc)
        logs['val_auc'].append(val_auc)

        if logs['best_auc'] < val_auc:
            logs['best_auc'] = val_auc
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
    pbar = tqdm(train_dataloader, desc=f'Train (epoch = {epoch})',leave=False)  

    total_loss = 0
    NME = []

    metric = CED()

    for batch in pbar:
      images, true_landmarks, image_orig_shape, face_crop_shape = batch

      images = images.to(device)
      true_landmarks = true_landmarks.to(device)

      predicted_landmarks = model(images)
      predicted_landmarks = predicted_landmarks.reshape(-1,68,2)

      loss = criterion(predicted_landmarks, true_landmarks)
      total_loss += loss

      orig_size_pred_landmarks = list(starmap(get_original_landmarks, zip(images.cpu(), predicted_landmarks.cpu(), image_orig_shape)))
      orig_size_pred_landmarks = torch.stack(orig_size_pred_landmarks, axis=0)

      orig_size_true_landmarks = list(starmap(get_original_landmarks, zip(images.cpu(), true_landmarks.cpu(), image_orig_shape)))
      orig_size_true_landmarks = torch.stack(orig_size_true_landmarks, axis=0)
      
      nme = metric.get_NME(orig_size_pred_landmarks, 
                       orig_size_true_landmarks, 
                       face_crop_shape)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      NME = np.concatenate((NME, nme))

    auc_metric = metric.get_AUC(NME, plot_ced=False)

    avg_loss = total_loss / len(train_dataloader)
    print("\nTrain AUC: {}".format(auc_metric))
    print("\nTrain loss: {}".format(avg_loss))

    return model, avg_loss, auc_metric

def get_original_landmarks(image, landmarks, original_coords, with_image=False):

  resize_after = torchlm.LandmarksResize((int(original_coords[1]),int(original_coords[0])))
  untransform_image, untransform_landmarks = resize_after(image.permute(1,2,0).numpy().astype(np.uint8), landmarks)

  if with_image:
    return untransform_image, untransform_landmarks
  else:
    return untransform_landmarks

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
      images, true_landmarks, image_orig_shape, face_crop_shape = batch

      images = images.to(device)
      true_landmarks = true_landmarks.to(device)

      with torch.no_grad():
        predicted_landmarks = model(images)
        predicted_landmarks = predicted_landmarks.reshape(-1,68,2)

        loss = criterion(predicted_landmarks, true_landmarks)
        total_loss += loss

        orig_size_pred_landmarks = list(starmap(get_original_landmarks, zip(images.cpu(), predicted_landmarks.cpu(), image_orig_shape)))
        orig_size_pred_landmarks = torch.stack(orig_size_pred_landmarks, axis=0)

        orig_size_true_landmarks = list(starmap(get_original_landmarks, zip(images.cpu(), true_landmarks.cpu(), image_orig_shape)))
        orig_size_true_landmarks = torch.stack(orig_size_true_landmarks, axis=0)
        
        nme = metric.get_NME(orig_size_pred_landmarks, 
                             orig_size_true_landmarks, 
                             face_crop_shape)
        
      NME = np.concatenate((NME, nme))

    auc_metric = metric.get_AUC(NME, plot_ced=False)
    avg_loss = total_loss / len(val_dataloader)
    print("\nValidation AUC: {}".format(auc_metric))
    print("\nValidation loss: {}".format(avg_loss))

    return model, avg_loss, auc_metric