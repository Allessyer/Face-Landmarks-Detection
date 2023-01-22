import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from itertools import starmap
import torch

class CED:
  
  def fr_and_auc(self, nmes, method="simpson", thres=0.08, step=0.0001, 
                      image_name=None,
                      path2saveimage=None,
                      plot_ced=False):
    
    num_data = len(nmes)
    xs = np.arange(0, thres + step, step)
    ys = np.array([np.count_nonzero(nmes <= x) for x in xs]) / float(num_data)
    fr = 1.0 - ys[-1]
    
    if method == "simpson":
        auc = integrate.simpson(ys, x=xs) / thres
    elif method == "trapez":
        auc = integrate.trapezoid(ys, x=xs) / thres
    
    if plot_ced:
        if image_name is None:
          plt.plot(xs,ys)
        else:
          plt.plot(xs,ys,label=f"{image_name}")
        plt.xlabel("NME")
        plt.ylabel("Images Proportion")
        plt.title(f"CED: AUC = {float('{:.4f}'.format(auc))}")
        plt.grid()

        if path2saveimage is not None:
          plt.savefig(f"{path2saveimage}/{image_name}_ced.png")

    return fr, auc

  def nme_batch(self, pred_landmarks, true_landmarks, rect):
    '''
    Parameters:
        pred_landmarks: np.array shape [batch_size, 68, 2]
        true_landmarks: np.array shape [batch_size, 68, 2]
        rect: np.array [x1,y1,x2,y2]
    Return:
        NME: np.array [batch_size]
    '''

    NME = np.array(list(starmap(self.nme_image, zip(pred_landmarks, true_landmarks, rect))))

    return NME

  def nme_image(self, pred_landmarks, true_landmarks, rect):

    '''
      Parameters:
        pred_landmarks: np.array shape: (68,2)
        true_landmarks: np.array shape: (68,2)
        rect: np.array [x1,y1,x2,y2]
      Return:
        nme: float
    '''

    W = abs(rect[2] - rect[0])
    H = abs(rect[3] - rect[1])

    # version 1
    # eucledian_distance = np.sqrt((true_points - predicted_points).pow(2).sum(dim=1))
    # version 2
    eucledian_distance = np.linalg.norm(true_landmarks - pred_landmarks, axis=1)

    d = np.sqrt(H*W)
    N = true_landmarks.shape[0]

    NME = eucledian_distance.sum() / (N*d)

    return NME
  