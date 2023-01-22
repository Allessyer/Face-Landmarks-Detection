from face_landmarks.train import train_model
from face_landmarks.dataset import LandmarksDataset

from face_landmarks.model import ONet
from face_landmarks.model import ResNet18
from face_landmarks.model import YinNet
from face_landmarks.parser import createParser
from face_landmarks.transforms import Transforms

import torch
import numpy as np
import random

if __name__== "__main__":

    parser = createParser()
    namespace = parser.parse_args()

    # device
    n_gpu = namespace.n_gpu
    seed_number = namespace.seed_number

    # dataset
    datasets_dir = namespace.dataset_dir
    dataset_type = namespace.dataset_name
    annotations_file = namespace.annotations_file
    augment = namespace.augment

    # model
    model_name = namespace.model_name

    # optimizer + scheduler
    optimizer_name = namespace.optimizer_name
    lr = namespace.learning_rate
    weight_decay = namespace.weight_decay

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

    # model
    if model_name == "ONet":
        resize_size = 48
        model = ONet()
    elif model_name == "ResNet18":
        resize_size = 224
        model = ResNet18()
    elif model_name == 'YinNet':
        resize_size = 128
        model = YinNet()

    model = model.to(device)

    # dataset
    train_dataset = LandmarksDataset(datasets_dir, 
                       annotations_file,
                       dataset_type=dataset_type, 
                       train_type="train",
                       postprocess=False,
                       transform=Transforms(augment=augment),
                       resize_size=resize_size
                )

    train_size = int(len(train_dataset) * 0.85)
    val_size = len(train_dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_dataloader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=64)
    val_dataloader = torch.utils.data.DataLoader(val_set, shuffle=True, batch_size=64)

    # optimizer + scheduler
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(),lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(),lr=lr, weight_decay=weight_decay)
        
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.1)

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



