from face_landmarks.test_image import get_landmarks
from face_landmarks.parser import createParser

import torch
import numpy as np

if __name__== "__main__":

    parser = createParser(train=False)
    namespace = parser.parse_args()

    path2weights = namespace.path2weights
    path2image = namespace.path2image
    path2results = namespace.save_dir

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    
    image_name = path2image.split("/")[-1].split(".")[0]
    path2results = "/workdir/results"

    print("Start finding landmarks...")
    all_pred_landmarks = get_landmarks(device,
                  path2weights,
                  path2image)

    print(f"Check {path2results} directory, file with saved landmarks coordinates is there.")
    np.save(f"{path2results}/{image_name}_predicted_landmarks", all_pred_landmarks)