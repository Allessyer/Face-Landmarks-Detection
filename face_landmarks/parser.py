import argparse

def createParser(train=True):

    if train:
        parser = argparse.ArgumentParser(description='Input parameters to train the model.')

        # device
        parser.add_argument ('-n_gpu', '--n_gpu', type=int, default=0,
                            help='GPU core number.')
        parser.add_argument ('-seed_number', '--seed_number', type=int, default=42,
                            help='Seed number.')

        # dataset
        parser.add_argument ('-ddir', '--dataset_dir', default="/workdir/landmarks_task",
                            help='Path to the dataset directory.')
        parser.add_argument ('-ann_file', '--annotations_file', 
                            default="/workdir/annotation_files/annotations_file_cleaned.csv",
                            help='Path to annotations file.')
        parser.add_argument ('-dataset', '--dataset_name', choices=("300W","Menpo","joint"), 
                            default="300W", help='Choose, which dataset to use: 300W, Menpo or both.')
        parser.add_argument ('-augment', '--augment', type=int, choices=(1, 0), 
                            default=1, help='Choose, 0 - no augmentation, 1 - do augmentation.')

        # model
        parser.add_argument ('-model', '--model_name', choices=("ONet", "ResNet18", "YinNet"), 
                            default="ONet", help='Choose, which model to use: ONet, ResNet18 or YinNet.')

        # optimizer + scheduler
        parser.add_argument ('-optimizer', '--optimizer_name', choices=("Adam", "AdamW"), 
                            default="Adam", help='Choose, which optimizer to use: Adam or AdamW.')
        parser.add_argument ('-lr', '--learning_rate', type=float, default=0.001,
                             help='Put learning rate value.')
        parser.add_argument ('-wd', '--weight_decay', type=float, default=0,
                            help='Put weight decay value.')

        # train
        parser.add_argument ('-n_epochs', '--num_epochs', type=int, default=1000,
                            help='Number of epochs in training process.')
        parser.add_argument ('-save_dir', '--save_dir', default="/workdir/results",
                            help='Path to the directory where output data will be saved.')
        parser.add_argument ('-exp_name', '--exp_name', default="exp00",
                            help='Name of the experiment.')
    else:
        parser = argparse.ArgumentParser(description='Input parameters to test the model.')

        parser.add_argument ('-path2weights', '--path2weights', 
                            default="/workdir/weights/YinNet_exp1_3_model_best_auc.pth",
                            help='Path to weights.')

        parser.add_argument ('-path2image', '--path2image', 
                            default="/workdir/landmarks_task/300W/test/2353849_1.jpg",
                            help='Path to image.')

        parser.add_argument ('-save_dir', '--save_dir', default="/workdir/results",
                            help='Path to the directory where output data will be saved.')

    return parser
