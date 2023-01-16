from face_landmarks.dataset import LandmarksDataset
from face_landmarks.metric import CED
import dlib
import torchlm
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def _shape_to_np(shape):
        coords = np.zeros((68, 2))
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

def get_original_landmarks(image, landmarks, image_orig_shape):

  resize_after = torchlm.LandmarksResize((int(image_orig_shape[1]),int(image_orig_shape[0])))
  untransform_image, untransform_landmarks = resize_after(image, landmarks)

  return untransform_landmarks

def main():
    path2dir = "/workdir/landmarks_task"

    test_Menpo_dataset = LandmarksDataset(path2dir, dataset_name="Menpo", 
                                     train_type="test")
    
    root_dir = "/workdir"
    predictor = dlib.shape_predictor(f'{root_dir}/shape_predictor_68_face_landmarks.dat')

    NME = []
    metric = CED(model_name="DLIB",path2results="/workdir/results")
    for i in tqdm(range(len(test_Menpo_dataset))):

        image, true_landmarks, image_orig_shape, face_crop_shape = test_Menpo_dataset[i]
        
        rect = dlib.rectangle(left=0, top=0, right=image.shape[1], bottom=image.shape[0])
        pred_landmarks = predictor(image, rect)
        pred_landmarks = _shape_to_np(pred_landmarks)

        true_landmarks = get_original_landmarks(image, true_landmarks, image_orig_shape)
        pred_landmarks = get_original_landmarks(image, pred_landmarks, image_orig_shape)

        nme_score = metric.nme(pred_landmarks, true_landmarks, face_crop_shape)
        NME.append(nme_score)
        
    auc = metric.get_AUC(NME, plot_ced=True)

    plt.clf()

    plt.imshow(image)
    plt.scatter(true_landmarks[:,0], true_landmarks[:,1], s=5, c='b')
    plt.scatter(pred_landmarks[:,0], pred_landmarks[:,1],s=5, c='r')

    plt.savefig("/workdir/results/DLIB_sample.png")

    print(f"DLIB AUC = {auc}")


if __name__== "__main__":
  main()