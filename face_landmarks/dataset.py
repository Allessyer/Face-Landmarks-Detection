import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

import dlib
import albumentations as A

# read pts
from pathlib import Path
from typing import Union


class LandmarksDataset(Dataset):
    def __init__(self, path2dir, dataset_name, train_type, transforms=None):
        super(LandmarksDataset, self).__init__()
        '''
        Parameters:
          path2dir: path to directory, where all datasets are located.
          dataset_type: 300W or Menpo or joint
          train_type: train or test
        '''

        df = pd.read_csv(f"{path2dir}/annotations_file_all_clean.csv")
        if dataset_name != "joint":
          df = df[df["dataset_type"] == dataset_name]

        df = df[df["train_type"] == train_type]
        df = df.reset_index()
        df.drop('index', axis=1, inplace=True)

        self.df = df
        self.path2dir = path2dir
        self.dataset_name = dataset_name
        self.train_type = train_type

        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        image, landmarks, rect = self._get_sample(index)
        image_orig_shape = torch.tensor(image.shape)
        image, landmarks = self._FaceCrop(rect, image, landmarks)
        face_crop_shape = torch.tensor(image.shape) # face_crop_shape

        if self.transforms:
          image, landmarks = self.transforms(image, landmarks)
          image = image.type(torch.FloatTensor)

        return image, landmarks, image_orig_shape, face_crop_shape

    def _get_sample(self, index):

      image_name = self.df["image_file"][index]
      if self.dataset_name == "joint":
        dataset_type = self.df["dataset_type"][index]
      else:
        dataset_type = self.dataset_name
      path2data = f'{self.path2dir}/{dataset_type}/{self.train_type}'

      image = dlib.load_rgb_image(f'{path2data}/{image_name}')
      points_file = f'{image_name.split(".")[0]}.pts'
      landmarks = read_pts(f'{path2data}/{points_file}')

      rect = np.array(self.df.iloc[index,5:9].values, dtype=int)

      return image, landmarks, rect

    def _FaceCrop(self, rect, image, landmarks):
      x1,y1,x2,y2 = rect[0],rect[1],rect[2],rect[3]

      if x1 < 0:
        x1 = 0
      if x2 > image.shape[1]:
        x2 = image.shape[1]
      if y1 > image.shape[0]:
        y1 = image.shape[0]
      if y2 < 0:
        y2 = 0

      # check landmarks in the image
      # landmarks are inside x-axis frame
      cond1 = landmarks[:,0].min() > 0 and landmarks[:,0].max() < image.shape[1]
      # landmarks are inside y-axis frame
      cond2 = landmarks[:,1].min() > 0 and landmarks[:,1].max() < image.shape[0]

      if cond1 and cond2:
        crop = A.Compose(
          [A.augmentations.crops.transforms.Crop (x_min=x1, y_min=y2, x_max=x2, y_max=y1)], 
          keypoint_params=A.KeypointParams(format='xy')
        )

        transformed = crop(image=image, keypoints=landmarks)
        image = transformed['image']
        landmarks = np.array([[x,y] for x,y in transformed['keypoints']])

      return image, landmarks

def read_pts(filename: Union[str, bytes, Path]) -> np.ndarray:
    """
    Read a .PTS landmarks file into a numpy array
    This function was taken from this resource: 
    https://stackoverflow.com/questions/59591181/does-python-have-a-standard-pts-reader-or-parser
    """
    with open(filename, 'rb') as f:
        # process the PTS header for n_rows and version information
        rows = version = None
        for line in f:
            if line.startswith(b"//"):  # comment line, skip
                continue
            header, _, value = line.strip().partition(b':')
            if not value:
                if header != b'{':
                    raise ValueError("Not a valid pts file")
                if version != 1:
                    raise ValueError(f"Not a supported PTS version: {version}")
                break
            try:
                if header == b"n_points":
                    rows = int(value)
                elif header == b"version":
                    version = float(value)  # version: 1 or version: 1.0
                elif not header.startswith(b"image_size_"):
                    # returning the image_size_* data is left as an excercise
                    # for the reader.
                    raise ValueError
            except ValueError:
                raise ValueError("Not a valid pts file")

        # if there was no n_points line, make sure the closing } line
        # is not going to trip up the numpy reader by marking it as a comment
        points = np.loadtxt(f, max_rows=rows, comments="}")

    if rows is not None and len(points) < rows:
        raise ValueError(f"Failed to load all {rows} points")
    return points