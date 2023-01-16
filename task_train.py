import torch
import numpy as np
import random
import torchlm

# face landmark library
from face_landmarks.parser import createParser
from face_landmarks.dataset import LandmarksDataset
from face_landmarks.model import ONet
from face_landmarks.model import ResNet18
from face_landmarks.train import train_model

def main():
    parser = createParser()
    namespace = parser.parse_args()

    # device
    n_gpu = namespace.n_gpu
    seed_number = namespace.seed_number

    # dataset
    path2datasetdir = namespace.dataset_dir
    dataset_name = namespace.dataset_name

    # transforms
    resize = namespace.resize
    augment = namespace.augment

    # model
    model_name = namespace.model_name

    # train
    num_epochs = namespace.num_epochs
    path2results = namespace.save_dir
    exp_name = namespace.exp_name


    device = torch.device(f"cuda:{str(n_gpu)}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    np.random.seed(seed_number)
    random.seed(seed_number)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True

    # dataset
    dataset = LandmarksDataset(path2dir = path2datasetdir, dataset_name=dataset_name, 
                                     train_type="train")

    # transforms
    if augment:
        train_transform = torchlm.LandmarksCompose([
            # use native torchlm transforms
            torchlm.LandmarksResize((resize, resize)),
            torchlm.LandmarksRandomTranslate(prob=0.5),
            torchlm.LandmarksRandomBlur(kernel_range=(5, 25), prob=0.5),
            torchlm.LandmarksRandomBrightness(prob=0.),
            torchlm.LandmarksRandomRotate(40, prob=0.5, bins=8),
            torchlm.LandmarksNormalize(),
            torchlm.LandmarksToTensor(),
        ])

        val_transform = torchlm.LandmarksCompose([
            # use native torchlm transforms
            torchlm.LandmarksResize((resize, resize)),
            torchlm.LandmarksToTensor(),
        ])
    else:
        train_transform = torchlm.LandmarksCompose([
            # use native torchlm transforms
            torchlm.LandmarksResize((resize, resize)),
            torchlm.LandmarksToTensor(),
        ])

        val_transform = torchlm.LandmarksCompose([
            # use native torchlm transforms
            torchlm.LandmarksResize((resize, resize)),
            torchlm.LandmarksToTensor(),
        ])
    
    train_size = int(len(dataset) * 0.85)
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_set.dataset.transforms = train_transform
    val_set.dataset.transforms = val_transform

    train_dataloader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=64)
    val_dataloader = torch.utils.data.DataLoader(val_set, shuffle=True, batch_size=64)

    if model_name == "ONet":
        model = ONet()
    elif model_name == "ResNet18":
        model = ResNet18()
    
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    if model_name == "ResNet18":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    criterion = torch.nn.MSELoss()
    criterion = criterion.to(device)

    model = train_model(model, 
                        num_epochs, 
                        train_dataloader, 
                        val_dataloader,
                        criterion,
                        optimizer,
                        scheduler,
                        device, 
                        path2results, 
                        model_name,
                        exp_name)

if __name__== "__main__":
  main()
