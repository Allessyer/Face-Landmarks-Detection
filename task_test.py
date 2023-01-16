import torch

# face landmark library
from face_landmarks.parser import createParser
from face_landmarks.test import test

def main():

    parser = createParser(train=False)
    namespace = parser.parse_args()

    n_gpu = namespace.n_gpu
    model_name = namespace.model_name
    exp_name = namespace.exp_name
    path2dir = namespace.dataset_dir
    path2results = namespace.save_dir
    path2weights = namespace.path2weights
    dataset_name = namespace.dataset_name

    device = torch.device(f"cuda:{str(n_gpu)}" if torch.cuda.is_available() else "cpu")

    dataset_name = "Menpo"

    test(model_name,
        exp_name,
        path2dir,
        path2results,
        path2weights,
        device,
        dataset_name)

if __name__== "__main__":
  main()
