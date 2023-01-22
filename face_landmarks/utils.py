from os import listdir
from os.path import isfile, join
from pathlib import Path
from typing import Union

import os
import os.path

import sys
from tqdm import tqdm
from tqdm import trange
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2
from skimage import io
import dlib
import torch
from torch import linalg as LA
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import crop
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_number = 42
torch.manual_seed(seed_number)
torch.cuda.manual_seed(seed_number)
np.random.seed(seed_number)
# random.seed(seed_number)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True


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

def points_in_bbox(landmarks, rect):

    '''
    Returns how many landmarks are in the face rectangular.
    '''
    in_bbox = 0
    x1 = rect[0]
    y1 = rect[1]
    x2 = rect[2]
    y2 = rect[3]

    for point_coords in landmarks: 
      x, y = point_coords
      if (x > x1 and x < x2) and (y > y2 and y < y1):
        in_bbox += 1

    return in_bbox

def expand_rect(rect,percentage=0.15):
    '''
    Expand the rectangle by 15% of the total width and height.

    Return new coordinates of bounding box.
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

    new_rect = [new_x1, new_y1, new_x2, new_y2]
    return new_rect # new_x1, new_y1, new_x2, new_y2

def find_face(image, landmarks, face_detector, device):
  faces = face_detector(image, 1)

  if len(faces) == 0:
    rect = [None, None, None, None]
    in_bbox = None
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
    
    if len(faces) == 1:
      rect = face_coords[0]
    else:
      in_bboxs = []
      for rect_ in face_coords:
        in_bbox = points_in_bbox(landmarks, rect_)
        in_bboxs.append(in_bbox)

      rect = face_coords[np.argmax(in_bboxs)]

    # rect = expand_rect(rect, percentage=0.3)
    in_bbox = points_in_bbox(landmarks, rect)
    # if in_bbox != 68:
    #   print(f"{in_bbox} landmarks are in bbox.")

  return rect, in_bbox, len(faces)

def shape_to_np(shape):
    coords = np.zeros((68, 2))
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def plot_sample(image, points):
    plt.imshow(image)
    plt.scatter(points[:,0], points[:,1],s=5)
    plt.show()