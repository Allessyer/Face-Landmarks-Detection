from face_landmarks.dataset import LandmarksDataset
from face_landmarks.metric import CED
from face_landmarks.utils import shape_to_np

import dlib
import torchlm
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch



def test_dlib(datasets_dir, 
            annotations_file,
            dataset_type="Menpo", 
            train_type="test"):

    predictor = dlib.shape_predictor(f'/workdir/shape_predictor_68_face_landmarks.dat')
    test_dataset = LandmarksDataset(datasets_dir, 
                       annotations_file,
                       dataset_type=dataset_type, 
                       train_type=train_type,
                       postprocess=False,
                       transform=None,
                       resize_size=None
                    )
                    
    metric = CED()

    NME = []
    for index in tqdm(range(len(test_dataset))):
        image, points, rect = test_dataset[index]
        dlib_rect = dlib.rectangle(left=rect[0], top=rect[1], right=rect[2], bottom=rect[3]) 
        pred_landmarks = predictor(image, dlib_rect)
        pred_landmarks = shape_to_np(pred_landmarks)

        nme_score = metric.nme_image(pred_landmarks, points, rect)
        NME.append(nme_score)
    NME = np.sort(NME)

    fr_0_08, auc_0_08 = metric.fr_and_auc(NME,method="simpson",thres=0.08)
    fr_1_0, auc_1_0 = metric.fr_and_auc(NME,method="simpson",thres=1.0)

    print("\nDLIB AUC 0.08: {}".format(auc_0_08))
    print("\nDLIB FR 0.08: {}".format(fr_0_08))
    print("\nDLIB AUC 1.0: {}".format(auc_1_0))
    print("\nDLIB FR 1.0: {}".format(fr_1_0))

    metrics = {}

    metrics["auc_0_08"] = auc_0_08
    metrics["fr_0_08"] = fr_0_08
    metrics["auc_1_0"] = auc_1_0
    metrics["fr_1_0"] = fr_1_0

    return metrics



if __name__== "__main__":
    
    path2results = "/workdir/results"
    datasets_dir = "/workdir/landmarks_task"
    annotations_file = "/workdir/annotation_files/annotations_file_cleaned.csv"

    results = {}

    dataset_type = "Menpo"
    train_type = "test"
    results[f"{dataset_type}_{train_type}"] = test_dlib(datasets_dir, 
                annotations_file,
                dataset_type=dataset_type, 
                train_type=train_type)

    dataset_type = "Menpo"
    train_type = "train"
    results[f"{dataset_type}_{train_type}"] = test_dlib(datasets_dir, 
                annotations_file,
                dataset_type=dataset_type, 
                train_type=train_type)

    dataset_type = "Menpo"
    train_type = "joint"
    results[f"{dataset_type}_{train_type}"] = test_dlib(datasets_dir, 
                annotations_file,
                dataset_type=dataset_type, 
                train_type=train_type)

    dataset_type = "300W"
    train_type = "test"
    results[f"{dataset_type}_{train_type}"] = test_dlib(datasets_dir, 
                annotations_file,
                dataset_type=dataset_type, 
                train_type=train_type)

    dataset_type = "joint"
    train_type = "test"
    results[f"{dataset_type}_{train_type}"] = test_dlib(datasets_dir, 
                annotations_file,
                dataset_type=dataset_type, 
                train_type=train_type)

    torch.save(results, f"{path2results}/dlib_results_cleaned_more40")

    
  
  


