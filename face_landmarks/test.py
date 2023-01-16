import matplotlib.pyplot as plt
import torch
import torchlm
from tqdm import tqdm
from itertools import starmap
import numpy as np

# face landmark library
from face_landmarks.metric import CED
from face_landmarks.dataset import LandmarksDataset
from face_landmarks.model import ONet
from face_landmarks.model import ResNet18


def test(model_name,
        exp_name,
        path2dir,
        path2results,
        path2weights,
        device,
        dataset_name="Menpo"):

    test_Menpo_dataset = LandmarksDataset(path2dir, dataset_name=dataset_name, 
                                     train_type="test")

    if model_name == "ONet":
        resize = 48
    else:
        resize = 128

    test_transform = torchlm.LandmarksCompose([
                torchlm.LandmarksResize((resize, resize)),
                torchlm.LandmarksToTensor(),
            ])
            
    test_Menpo_dataset.transforms = test_transform
    test_dataloader = torch.utils.data.DataLoader(test_Menpo_dataset, 
                                                shuffle=True, 
                                                batch_size=64)
    if model_name == "ONet":
        model = ONet()
    elif model_name == "ResNet18":
        model = ResNet18()

    model_weights = f"{model_name}_{exp_name}_model_best_auc.pth"
    model.load_state_dict(torch.load(f"{path2weights}/{model_weights}"))

    model = model.to(device)

    auc = test_model(model,
              test_dataloader,
              model_name,
              dataset_name,
              exp_name,
              path2results,
              device)

def test_model(model,
              test_dataloader,
              model_name,
              dataset_name,
              exp_name,
              path2results,
              device):
    
    model.eval()

    NME = []
    if dataset_name=="joint":
      dataset_name = "300W+Menpo"

    metric = CED(model_name=f"{model_name}_{exp_name}_{dataset_name}",path2results=path2results)

    for batch in tqdm(test_dataloader):
      images, true_landmarks, image_orig_shape, face_crop_shape = batch

      images = images.to(device)
      true_landmarks = true_landmarks.to(device)

      with torch.no_grad():
        predicted_landmarks = model(images)
        predicted_landmarks = predicted_landmarks.reshape(-1,68,2)

        orig_size_pred_landmarks = list(starmap(get_original_landmarks, zip(images.cpu(), predicted_landmarks.cpu(), image_orig_shape)))
        orig_size_pred_landmarks = torch.stack(orig_size_pred_landmarks, axis=0)

        orig_size_true_landmarks = list(starmap(get_original_landmarks, zip(images.cpu(), true_landmarks.cpu(), image_orig_shape)))
        orig_size_true_landmarks = torch.stack(orig_size_true_landmarks, axis=0)
        
        nme = metric.get_NME(orig_size_pred_landmarks, 
                             orig_size_true_landmarks, 
                             face_crop_shape)
        
      NME = np.concatenate((NME, nme))

    auc_metric = metric.get_AUC(NME, plot_ced=True)
    print("\nValidation AUC: {}".format(auc_metric))

    plot_sample(images[0].cpu(), true_landmarks[0].cpu(), predicted_landmarks[0].cpu(), image_name=f"{model_name}_{exp_name}_{dataset_name}")

    return auc_metric


def get_original_landmarks(image, landmarks, original_coords, with_image=False):

  resize_after = torchlm.LandmarksResize((int(original_coords[1]),int(original_coords[0])))
  untransform_image, untransform_landmarks = resize_after(image.permute(1,2,0).numpy().astype(np.uint8), landmarks)

  if with_image:
    return untransform_image, untransform_landmarks
  else:
    return untransform_landmarks

def plot_sample(image, true, pred, image_name):
    true_land = true.cpu()
    pred_land = pred.cpu().detach().numpy()

    plt.imshow(image.permute(1,2,0).to(torch.uint8).cpu().numpy())
    plt.scatter(true_land[:,0],true_land[:,1], s=5, c = 'b')
    plt.scatter(pred_land[:,0],pred_land[:,1], s=5, c = 'r')

    plt.savefig(f"/workdir/results/{image_name}_sample.png")
