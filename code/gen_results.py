import os

import random
import json
import imgaug
import torch
import numpy as np
import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 1234
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from tqdm import tqdm
import models
import torch.nn.functional as F
from utils.datasets.fer2013dataset import fer2013
from utils.generals import make_batch

model_dict_new = [
    ("efficientnet_b2b", "../user_data/model_data/efficientnet_b2b_2021Jul25_17.08"),
    ("efficientnet_b3b", "../user_data/model_data/efficientnet_b3b_2021Jul25_20.08"),
    ("cbam_resnet50", "../user_data/model_data/cbam_resnet50_test_2021Jul24_19.18"),
    ("resmasking_dropout1", "../user_data/model_data/resmasking_dropout1_test_2021Jul25_10.03"),
    ("resmasking", "../user_data/model_data/resmasking_test_2021Jul26_14.33"),
    ("resnest269e","../user_data/model_data/tbw_resnest269e_test_2021Aug02_11.39"),
    ("hrnet","../user_data/model_data/tbw_hrnet_test_2021Aug01_17.13"),
    ("swin_large_patch4_window7_224","../user_data/model_data/tbw_swin_large_patch4_window7_224_test_2021Aug02_21.36")
]

def main():
    parser = argparse.ArgumentParser(description='emotion')
    parser.add_argument('--config', default="fer2013_config.json",type=str, help='config path')
    parser.add_argument('--model_name', default="resmasking_dropout1",type=str, help='config path')
    parser.add_argument('--checkpoint_path', default="resmasking_dropout1_test_2021Aug01_17.13",type=str, help='config path')
    args = parser.parse_args()
    with open(args.config) as f:
        configs = json.load(f)

    test_set = fer2013("test", configs, tta=True, tta_size=10)

    # for model_name, checkpoint_path in model_dict_new:
    prediction_list = []  # each item is 7-ele array

    print("Processing", args.checkpoint_path)
    if os.path.exists("../user_data/temp_data/{}.npy".format(args.checkpoint_path)):
        return
    if configs['type'] == 0:
        model = getattr(models, args.model_name)
        model = model(in_channels=3, num_classes=7)
    elif configs['type'] == 1:
        model = models.get_face_model(name=args.model_name)
    else:
        model = getattr(models, args.model_name)
        model = model()
    state = torch.load(os.path.join("../user_data/model_data/", args.checkpoint_path),map_location=torch.device('cpu'))
    ckpt = {k.replace("module.",''):v for k,v in state['net'].items()}
    model.load_state_dict(ckpt)
    # model = torch.nn.DataParallel(model)
    model.cuda()
    model.eval()

    with torch.no_grad():
        for idx in tqdm(range(len(test_set)), total=len(test_set), leave=False):
            images, targets = test_set[idx]
            images = make_batch(images)
            images = images.cuda(non_blocking=True)

            outputs = model(images).cpu()
            outputs = F.softmax(outputs, 1)
            outputs = torch.sum(outputs, 0)  # outputs.shape [tta_size, 7]

            outputs = [round(o, 4) for o in outputs.numpy()]
            prediction_list.append(outputs)
    prediction_list = np.asarray(prediction_list)
    if args.checkpoint_path.split('_')[0] != 'efficientnet':
        data_4 = prediction_list[:,4].copy()
        prediction_list[:,4] = prediction_list[:,6]
        prediction_list[:,6] = prediction_list[:,5]
        prediction_list[:,5] = data_4
    np.save("../temp_data/{}.npy".format(args.checkpoint_path), prediction_list)


if __name__ == "__main__":
    main()
