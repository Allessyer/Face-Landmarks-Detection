from os import listdir
from os.path import isfile, join
import dlib
import numpy as np
from tqdm import tqdm
import pandas as pd

from face_landmarks.utils import read_pts
from face_landmarks.utils import find_face


def create_annotations_file(datasets_dir, dataset_name, train_type, device):
    
    path2data = f"{datasets_dir}/{dataset_name}/{train_type}"
    onlyfiles = [f for f in listdir(path2data) if isfile(join(path2data, f))]
    pts_files = sorted([file for file in onlyfiles if file.split(".")[-1] == "pts"])
    img_files = sorted([file for file in onlyfiles if file.split(".")[-1] != "pts"])

    image_name_array = []

    face_rect_x1 = []
    face_rect_y1 = []
    face_rect_x2 = []
    face_rect_y2 = []
    landmarks_in_bbox = []
    n_faces_detected = []

    dataset_type_array = []
    train_type_array = []

    if device.type == 'cuda':
        path2cnn_weights = "/workdir/mmod_human_face_detector.dat"
        face_detector = dlib.cnn_face_detection_model_v1(path2cnn_weights)
    else:
        face_detector = dlib.get_frontal_face_detector()

    N = len(img_files)
    for index in tqdm(range(N)):

      image_name = img_files[index]
      image = dlib.load_rgb_image(f'{path2data}/{image_name}')

      points_file = f'{image_name.split(".")[0]}.pts'
      points = read_pts(f'{path2data}/{points_file}')

      rect, in_bbox, n_faces = find_face(image, points, face_detector, device)
      
      image_name_array.append(image_name)
      face_rect_x1.append(rect[0])
      face_rect_y1.append(rect[1])
      face_rect_x2.append(rect[2])
      face_rect_y2.append(rect[3])
      landmarks_in_bbox.append(in_bbox)
      n_faces_detected.append(n_faces)

    dataset_type_array = np.concatenate((dataset_type_array,np.full(N, dataset_name)))
    train_type_array = np.concatenate((train_type_array,np.full(N, train_type)))


    df = pd.DataFrame()

    df['image_name'] = image_name_array
    df['face_rect_x1'] = face_rect_x1
    df['face_rect_y1'] = face_rect_y1
    df['face_rect_x2'] = face_rect_x2
    df['face_rect_y2'] = face_rect_y2
    df['landmarks_in_bbox'] = landmarks_in_bbox
    df['n_faces_detected'] = n_faces_detected

    df['dataset_type'] = dataset_type_array
    df['train_type'] = train_type_array

    return df