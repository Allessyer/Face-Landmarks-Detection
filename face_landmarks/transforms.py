import numpy as np
import cv2
import dlib
from torchvision import transforms
import torch
from itertools import starmap


def untransform_batch(image, landmarks, rect):
    '''
    Parameters:
        image: torch.tensor [batch_size, channels, height, width]
        landmarks: torch.tensor [batch_size, 68, 2]
        rect: torch.tensor [batch_size, 4]
    Return:
        Untransformed landmarks: np.array [batch_size,  68, 2]
    '''

    T = Transforms()

    image = image.permute(0,2,3,1).numpy()
    landmarks = landmarks.numpy()
    rect = rect.numpy()

    landmarks = list(starmap(T.untransform, zip(image, landmarks, rect)))
    landmarks = np.array(landmarks)
    
    return landmarks

class Transforms():
    def __init__(self, augment=False):
        self.augment = augment

    def transform(self, image, landmarks, rect, resize_size):

        image, landmarks, rect = self.crop(image, landmarks, rect)
        image, landmarks = self.resize(image, landmarks, (resize_size, resize_size))

        if self.augment:
            torchvision_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=0.3, 
                                                contrast=0.3,
                                                saturation=0.3, 
                                                hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )])
        else:
            torchvision_transform = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.ToTensor()])

        image = torchvision_transform(image)
        landmarks = torch.tensor(landmarks, dtype=torch.float)
        rect = torch.tensor(rect)

        return image, landmarks, rect

    def untransform(self, image, landmarks, rect):

        '''
        Parameters:
            image: np.array [H,W,RGB]
            landmarks: np.array (68,2)
            rect: np.array [x1,y1,x2,y2]
        Return:
            cropped_image: np.array
            cropped_landmarks: np.array
            new_rect: np.array
        '''

        image_size =(rect[3]-rect[1],rect[2]-rect[0]) 
        unresized_image, unresized_landmarks = self.unresize(image, landmarks, image_size)
        _, uncropped_landmarks, _ = self.uncrop(unresized_image, unresized_landmarks, rect)

        return uncropped_landmarks

    def crop(self, image, landmarks, rect):

        '''
        Parameters:
            image: np.array [H,W,RGB]
            landmarks: np.array (68,2)
            rect: np.array [x1,y1,x2,y2]
        Return:
            cropped_image: np.array
            cropped_landmarks: np.array
            new_rect: np.array
        '''

        # check что rect не за пределами изображения
        x1 = rect[0] # left
        y1 = rect[1] # top
        x2 = rect[2] # right
        y2 = rect[3] # bottom

        if x1 < 0:
            x1 = 0
        if x2 > image.shape[1]:
            x2 = image.shape[1]
        if y1 < 0:
            y1 = 0
        if y2 > image.shape[0]:
            y2 = image.shape[0]

        cropped_image = image[y1:y2,x1:x2,:]

        cropped_landmarks = np.zeros((68,2))
        cropped_landmarks[:,0] = landmarks[:,0] - np.full(len(cropped_landmarks), x1)
        cropped_landmarks[:,1] = landmarks[:,1] - np.full(len(cropped_landmarks), y1)

        new_rect = np.array([x1,y1,x2,y2])

        return cropped_image, cropped_landmarks, new_rect


    def uncrop(self, image, cropped_landmarks, rect):
        '''
        Parameters:
            image: np.array
            cropped_landmarks: np.array 
                Landmarks coordinates
            rect: np.array [x1,y1,x2,y2]
        Return:
            resized_image: np.array
            resized_landmarks: np.array
        '''
    
        uncropped_landmarks = np.zeros((68,2))
        uncropped_landmarks[:,0] = cropped_landmarks[:,0] + np.full(len(uncropped_landmarks), rect[0]) # left
        uncropped_landmarks[:,1] = cropped_landmarks[:,1] + np.full(len(uncropped_landmarks), rect[1]) # top

        return image, uncropped_landmarks, rect


    def resize(self, image, landmarks, image_size):
        '''
        Parameters:
            image: np.array
            landmarks: np.array 
                Landmarks coordinates
            image_size: (H,W), because usually np arrays print image shape as (H,W)
                Desired image size
        Return:
            resized_image: np.array
            resized_landmarks: np.array
        '''

        resized_image = cv2.resize(image, (image_size[1],image_size[0]), interpolation = cv2.INTER_AREA) # here W,H
        
        scale_y = image_size[0] / image.shape[0]
        scale_x = image_size[1] / image.shape[1]

        resized_landmarks = np.zeros((68,2))

        resized_landmarks[:,0] = landmarks[:,0] * scale_x
        resized_landmarks[:,1] = landmarks[:,1] * scale_y
        
        return resized_image, resized_landmarks

    def unresize(self, image, landmarks, image_size):
        '''
        Parameters:
            image: np.array
            points: np.array 
            image_size: (H,W), because usually np arrays print image shape as (H,W)
        Return:
            unresized_image: np.array
            unresized_landmarks: np.array
        '''

        unresized_image = cv2.resize(image, (image_size[1],image_size[0]), interpolation = cv2.INTER_AREA) # W,H

        scale_y = image.shape[0] / image_size[0]
        scale_x = image.shape[1] / image_size[1]

        unresized_landmarks = np.zeros((68,2))

        unresized_landmarks[:,0] = landmarks[:,0] / scale_x
        unresized_landmarks[:,1] = landmarks[:,1] / scale_y

        return unresized_image, unresized_landmarks
