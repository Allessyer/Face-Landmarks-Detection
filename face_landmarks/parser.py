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
        parser.add_argument ('-dataset', '--dataset_name', choices=("300W","Menpo","joint"), 
                            default="300W", help='Choose, which dataset to use: 300W, Menpo or both.')

        # transforms
        parser.add_argument ('-resize', '--resize', type=int, default=48, 
                            help='Resize image to this size. For ONet, use 48, for ResNet use 128')
        parser.add_argument ('-augment', '--augment', type=int, choices=(1,0), default=0, 
                            help='Add augmentation.')

        # model
        parser.add_argument ('-model', '--model_name', choices=("ONet","ResNet18"), 
                            default="ResNet18", help='Choose, which model to use: ONet or ResNet18.')

        # train
        parser.add_argument ('-n_epochs', '--num_epochs', type=int, default=200,
                            help='Number of epochs in training process.')
        parser.add_argument ('-save_dir', '--save_dir', default="/workdir/results",
                            help='Path to the directory where output data will be saved.')
        parser.add_argument ('-exp_name', '--exp_name', default="exp00",
                            help='Name of the experiment.')
    else:
        parser = argparse.ArgumentParser(description='Input parameters to test the model.')

        parser.add_argument ('-model', '--model_name', choices=("ONet","ResNet18"), 
                            default="ONet", help='Choose, which model to use: ONet or ResNet18.')
        parser.add_argument ('-exp_name', '--exp_name', choices=("exp12","exp13","exp21","exp31","exp32"), 
                            default="exp12", help='Choose, which experiment weights to use.')
        parser.add_argument ('-ddir', '--dataset_dir', default="/workdir/landmarks_task",
                            help='Path to the dataset directory.')
        parser.add_argument ('-path2weights', '--path2weights', default="/workdir/weights",
                            help='Path to the dataset directory.')                   
        parser.add_argument ('-save_dir', '--save_dir', default="/workdir/results",
                            help='Path to the directory where output data will be saved.')
        parser.add_argument ('-n_gpu', '--n_gpu', type=int, default=0,
                            help='GPU core number.')
        parser.add_argument ('-dataset', '--dataset_name', choices=("300W","Menpo","joint"), 
                            default="300W", help='Choose, which dataset to use to test: 300W, Menpo or both.')
        

    return parser
