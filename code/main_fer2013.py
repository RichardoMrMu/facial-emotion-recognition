import os
import json
import random
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import argparse
import imgaug
import torch
import torch.multiprocessing as mp
import numpy as np


seed = 1234
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


import models
from models import segmentation,get_face_model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def main(config_path):
    """
    This is the main function to make the training up

    Parameters:
    -----------
    config_path : srt
        path to config file
    """
    # load configs and set random seed
    configs = json.load(open(config_path))
    configs["cwd"] = os.getcwd()

    # load model and data_loader
    model = get_model(configs)

    train_set, val_set, test_set = get_dataset(configs)

    # init trainer and make a training
    # from trainers.fer2013_trainer import FER2013Trainer
    from trainers.tta_trainer import FER2013Trainer

    # from trainers.centerloss_trainer import FER2013Trainer
    trainer = FER2013Trainer(model, train_set, val_set, test_set, configs)

    if configs["distributed"] == 1:
        ngpus = torch.cuda.device_count()
        mp.spawn(trainer.train, nprocs=ngpus, args=())
    else:
        trainer.train(1)


def get_model(configs):
    """
    This function get raw models from models package

    Parameters:
    ------------
    configs : dict
        configs dictionary
    """
    if configs['type'] == 0:
        try:
            return models.__dict__[configs["arch"]]
        except KeyError:
            return segmentation.__dict__[configs["arch"]]
    elif configs['type'] == 1:
        return get_face_model(name=configs['emo_name'],pretrained=True)
    else:
        return models.__dict__[configs["arch"]]

def get_dataset(configs):
    """
    This function get raw dataset
    """
    from utils.datasets.fer2013dataset import fer2013
    import torch
    # todo: add transform
    train_set = fer2013("train", configs)
    val_size = int(0.125*len(train_set))
    train_size = len(train_set) - val_size
    train_set,val_set = torch.utils.data.random_split(train_set,[train_size,val_size])
    # val_set = fer2013("val", configs)
    test_set = fer2013("test", configs, tta=True, tta_size=10)
    return train_set, val_set, test_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='emotion')
    parser.add_argument('--config', default="fer2013_config.json",type=str, help='config path')
    args = parser.parse_args()
    main(args.config)
