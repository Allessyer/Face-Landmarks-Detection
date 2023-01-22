import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

import dlib

# read pts
from pathlib import Path
from typing import Union

from face_landmarks.utils import read_pts

class LandmarksDataset(Dataset):
    def __init__(self, datasets_dir, 
                       annotations_file,
                       dataset_type, 
                       train_type,
                       postprocess=False,
                       transform=None,
                       resize_size=48
                ):
        super(LandmarksDataset, self).__init__()
        '''
        Parameters:
          datasets_dir: str
                path to directory, where all datasets are located.
          annotations_file: str
          dataset_type: str
                300W or Menpo or joint (300W+Menpo)
          train_type: str
                train or test or joint (train+test)
          postprocess: bool
          transform: None or class Transforms()
          resize_size: int
        '''
        self.datasets_dir = datasets_dir
        self.dataset_type = dataset_type
        self.train_type = train_type
        self.postprocess = postprocess
        self.transform = transform
        self.resize_size = resize_size

        df = pd.read_csv(annotations_file)
        if dataset_type != "joint":
            df = df.loc[(df['dataset_type'] == dataset_type),:].reset_index(drop=True)
        if train_type != "joint":
            df = df.loc[(df['train_type'] == train_type),:].reset_index(drop=True)
        
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        image, landmarks, rect, landmarks_in_bbox = self._get_sample(index)

        if self.postprocess:
            if landmarks_in_bbox != 68:
                rect = self.expand_rect(rect, image.shape, percentage=0.3)
                landmarks_in_bbox = self.points_in_bbox(landmarks, rect)

        if self.transform:
            image, landmarks, rect = self.transform.transform(image, landmarks, rect, self.resize_size)
            
        return image, landmarks, rect


    def _get_sample(self, index):

        ''' 
            Return:
                image: np.array
                points: np.array shape (68,2)
                rect: np.array [x1,y1,x2,y2]
                landmarks_in_bbox: int
        '''

        dataset_type = self.df["dataset_type"][index]
        train_type = self.df["train_type"][index]
        path2data = f"{self.datasets_dir}/{dataset_type}/{train_type}"

        image_name = self.df["image_name"][index]
        image = dlib.load_rgb_image(f'{path2data}/{image_name}')

        points_file = f'{image_name.split(".")[0]}.pts'
        points = read_pts(f'{path2data}/{points_file}')

        x1 = int(self.df["face_rect_x1"][index])
        y1 = int(self.df["face_rect_y1"][index])
        x2 = int(self.df["face_rect_x2"][index])
        y2 = int(self.df["face_rect_y2"][index])

        rect = np.array([x1,y1,x2,y2])

        landmarks_in_bbox = self.df["landmarks_in_bbox"][index]

        return image, points, rect, landmarks_in_bbox


    def expand_rect(self, rect, image_shape, percentage=0.30):
        '''
        Expand given rectangle.
        Parameters:
            rect: np.array [x1,y1,x2,y2]
            image_shape:
            percentage: how much you want to expand rectangle.
                        if percentage=0.30, it means rectangle expand by 30% of the total width and height.

        Return: 
            new_rect: np.array [x1,y1,x2,y2] 
                      New coordinates of bounding box.
        '''

        x1 = rect[0]
        y1 = rect[1]
        x2 = rect[2]
        y2 = rect[3]

        h = y2 - y1
        w = x2 - x1

        new_x1 = int(x1 - percentage*w)
        new_x2 = int(x2 + percentage*w)
        new_y1 = int(y1 - percentage*h)
        new_y2 = int(y2 + percentage*h)

        if new_x1 < 0:
            new_x1 = 0
        if new_x2 > image_shape[1]:
            new_x2 = image_shape[1]
        if new_y1 < 0:
            new_y1 = 0
        if new_y2 > image_shape[0]:
            new_y2 = image_shape[0]

        new_rect = np.array([new_x1, new_y1, new_x2, new_y2])

        return new_rect
    
    def points_in_bbox(self, landmarks, rect):

        '''
        Parameters:
            landmarks: np.array with shape (68,2)
            rect: np.array [x1,y1,x2,y2]
        Returns: 
            in_bbox: int
                how many landmarks are in the face rectangular.
        '''
        in_bbox = 0

        x1 = rect[0]
        y1 = rect[1]
        x2 = rect[2]
        y2 = rect[3]

        for point_coords in landmarks: 
            x, y = point_coords
            if (x > x1 and x < x2) and (y < y2 and y > y1):
                in_bbox += 1

        return in_bbox