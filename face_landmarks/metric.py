from itertools import starmap
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
import torch

class CED:

  def __init__(self, model_name=None, path2results=None):
    self.path2results = path2results
    self.model_name = model_name

  def get_AUC(self, nmes, plot_ced=False):

    fr, auc = self.fr_and_auc(nmes,plot=plot_ced)

    return auc

  def get_NME(self, predicted_points, true_points, face_crop_shape):

    NME = list(starmap(self.nme, zip(predicted_points, true_points, face_crop_shape)))
    NME = torch.stack(NME, axis=0).numpy()

    return NME


  def fr_and_auc(self, nmes,thres=0.08,step=0.0001,plot=False):
    num_data = len(nmes)
    xs = np.arange(0, thres + step, step)
    ys = np.array([np.count_nonzero(nmes <= x) for x in xs]) / float(num_data)
    fr = 1.0 - ys[-1]
    auc = simps(ys, x=xs) / thres
    print(f"auc with thresh: {auc}")
    print(f"auc without thresh: {simps(ys, x=xs)}")
    if plot:
        plt.plot(xs,ys)
        plt.xlabel("NME")
        plt.ylabel("Images Proportion")
        plt.title(f"{self.model_name} CED: AUC = {float('{:.4f}'.format(auc))} ")
        plt.grid()

        plt.savefig(f"{self.path2results}/{self.model_name}_CED.png")
    
    return fr, auc

  def nme(self, lms_pred, lms_gt, face_crop_shape):
    """
    :param lms_pred: (n,2) predicted landmarks.
    :param lms_gt: (n,2) ground truth landmarks.
    :param norm: normalize value, the distance between two eyeballs.
    :return: nme value.
    """
    lms_pred = lms_pred.reshape((-1, 2))
    lms_gt = lms_gt.reshape((-1, 2))

    W = face_crop_shape[1]
    H = face_crop_shape[0]
    
    norm = np.sqrt(H*W)

    nme_ = np.mean(np.linalg.norm(lms_pred - lms_gt, axis=1)) / norm
    return nme_
  