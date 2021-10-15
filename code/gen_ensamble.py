import random
import imgaug
import torch
import numpy as np
from utils.utils import model_dict,model_dict_proba_list
seed = 1024
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


    
def main2():
    global model_dict_proba_list
    test_results_list = []
    for model_name, checkpoint_path in model_dict:
        test_results = np.load(
            "../user_data/temp_data/{}.npy".format(checkpoint_path), allow_pickle=True
        )
        '''
        min = np.min(test_results, axis=1)
        min = np.expand_dims(min, axis=1)
        sigma = np.std(test_results, axis=1)
        sigma = np.expand_dims(sigma, axis=1)
        test_results = (test_results - min) / sigma
        '''
        test_results_list.append(test_results)
    test_results_list = np.array(test_results_list)
    print(test_results_list.shape)

    model_dict_proba_list = np.array(model_dict_proba_list)
    #print(model_dict_proba_list.shape)
    model_dict_proba_list = np.transpose(model_dict_proba_list, (1,0))
    print(model_dict_proba_list.shape)
    test_results_list = np.transpose(test_results_list,(1,0,2))
    tmp_test_result_list = np.multiply(test_results_list,model_dict_proba_list)
    print(tmp_test_result_list.shape)
    tmp_test_result_list = np.sum(tmp_test_result_list, axis=1)
    print(tmp_test_result_list.shape)
    tmp_test_result_list = np.argmax(tmp_test_result_list, axis=1)
    tmp_test_result_list = tmp_test_result_list.reshape(-1)
    return tmp_test_result_list

def gen_csv():
    import pandas as pd
    import sys
    sys.path.append("..")
    # from emotion2.preprocess import EMOTION_DICT
    EMOTION_DICT = {
    0: "angry",
    1: "disgusted",
    2: "fearful",
    3: "happy",
    4: "sad",
    5: "surprised",
    6: "neutral",
}
    res = main2()
    res_labels = []
    mapping = np.loadtxt("mapping.txt", dtype=np.int)
    for i in range(len(res)):
        res_labels.append(EMOTION_DICT[res[i]])
    res_labels = np.asarray(res_labels)
    res_labels = res_labels[mapping[:, 1]]
    csv_data = []
    for i in range(len(res_labels)):
        name = str(i + 1).zfill(5)
        csv_data.append([name + '.png', res_labels[i]])
    csv_data = pd.DataFrame(csv_data)
    csv_data.to_csv("../prediction_result/result.csv", header=['name', 'label'], index=None)

if __name__ == "__main__":
    gen_csv()