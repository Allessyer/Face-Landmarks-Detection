
import dlib
from face_landmarks.utils import shape_to_np
from face_landmarks.utils import read_pts
from face_landmarks.train import evaluate_single_epoch
from face_landmarks.dataset import LandmarksDataset
from face_landmarks.model import ONet
from face_landmarks.model import YinNet
from face_landmarks.model import ResNet18
from face_landmarks.transforms import Transforms
from face_landmarks.transforms import untransform_batch
from face_landmarks.metric import CED
from face_landmarks.utils import find_face


import torch
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2




def get_landmarks(device,
                  path2weights,
                  path2image):

    model = YinNet()
    model.load_state_dict(torch.load(f"{path2weights}", map_location=device))

    model = model.to(device)
    resize_size = 128

    model.eval()
    orig_image = dlib.load_rgb_image(path2image)
    points = np.zeros((68,2))

    if device.type == 'cuda':
        path2cnn_weights = "/workdir/mmod_human_face_detector.dat"
        face_detector = dlib.cnn_face_detection_model_v1(path2cnn_weights)
    else:
        face_detector = dlib.get_frontal_face_detector()
    
    faces = face_detector(orig_image, 1)

    if len(faces) == 0:
        print("Couldn't find any face.")
    elif len(faces) >= 1:
        face_coords = []
        for face in faces: 
            if device.type == "cuda":
                rect = face.rect
            else:
                rect = face
            
            x1 = rect.left()
            y1 = rect.top()
            x2 = rect.right()
            y2 = rect.bottom()
            
            face_coords.append([x1,y1,x2,y2])
        
        all_pred_landmarks = []
        for rect in face_coords:
            pred_landmarks = find_landmarks(orig_image, 
                                            rect,
                                            model,
                                            device)
        all_pred_landmarks.append(pred_landmarks)


    return np.array(all_pred_landmarks)


def find_landmarks(orig_image, 
                   rect,
                   model,device):
    
    path2cnn_weights = "/workdir/mmod_human_face_detector.dat"
    face_detector = dlib.cnn_face_detection_model_v1(path2cnn_weights)
    
    points = np.zeros((68,2))
    image, landmarks, rect = Transforms(augment=False).transform(orig_image, points, rect, 128)
    image = image.unsqueeze(0)
    rect = rect.unsqueeze(0)
    image = image.to(device)

    pred_landmarks = model(image)
    pred_landmarks = pred_landmarks.reshape(-1,68,2)

    pred_landmarks = pred_landmarks.detach().cpu()
    image = image.cpu()
    orig_pred_landmarks = untransform_batch(image, pred_landmarks, rect)[0]
    
    return orig_pred_landmarks
