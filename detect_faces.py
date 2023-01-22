import torch
import numpy as np
import sys

from face_landmarks.create_annotations_file import create_annotations_file

if __name__== "__main__":

    datasets_dir = "/workdir/landmarks_task"
    dataset_name =  sys.argv[1] #"Menpo"
    train_type = sys.argv[2]  #"test"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_number = 42
    torch.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    np.random.seed(seed_number)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True

    df = create_annotations_file(datasets_dir, dataset_name, train_type, device)

    path2annot_dir = "/workdir/annotation_files"
    df.to_csv(f'{path2annot_dir}/annotations_file_{dataset_name}_{train_type}.csv',index=False)

