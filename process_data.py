from ptb import *
import numpy as np
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
import json


def get_tokens(paths, name):
    p_lst, m_lst = [], []

    for i, path in tqdm(enumerate(paths), total=len(paths)):
        performance_tokens, melody_tokens = encode_performance_and_melody(path)
        p_lst.append(performance_tokens)
        m_lst.append(melody_tokens)

    p_lst = np.array(p_lst)
    m_lst = np.array(m_lst)
    np.save("performance_tokens_{}.npy".format(name), p_lst)
    np.save("melody_tokens_{}.npy".format(name), m_lst)


def get_augmented_data():

    # pitch augmentation
    def pitch_augmentation(tokens):
        res = [tokens]
        for value in [-3, -2, -1, 1, 2, 3]:
            cur = np.copy(tokens)
            cur[cur < 178] += value
            cur[cur < 0] = 0        # precaution for negative values
            res.append(cur)
        return res

    performance_tokens_train = np.load("data/performance_tokens_train_v2.npy", allow_pickle=True)
    performance_tokens_val = np.load("data/performance_tokens_val_v2.npy", allow_pickle=True)
    performance_tokens_test = np.load("data/performance_tokens_test_v2.npy", allow_pickle=True)

    counter = 0
    for performance_tokens in [performance_tokens_train, performance_tokens_val, performance_tokens_test]:

        new_tokens = []
        for idx, token in enumerate(performance_tokens):
            print(idx, len(performance_tokens), end="\r")
            token = np.array(token)
            cur_token = None
            
            # split length
            if len(token) < 2048:
                new_token = np.pad(token, (0, 2048 - len(token)), 'constant', constant_values=0)
                augmented_tokens = pitch_augmentation(new_token)
                new_tokens += augmented_tokens
            else:
                i = 0
                while i < 30:
                    index = random.randint(0, len(token) - 2048 - 1)
                    new_token = token[index:index+2048]
                    augmented_tokens = pitch_augmentation(new_token)
                    new_tokens += augmented_tokens
                    i += 1
        
        if counter == 0:
            np.save("data/performance_tokens_augmented_train_v2.npy", np.array(new_tokens))
        elif counter == 1:
            np.save("data/performance_tokens_augmented_val_v2.npy", np.array(new_tokens))
        elif counter == 2:
            np.save("data/performance_tokens_augmented_test_v2.npy", np.array(new_tokens))
        
        counter += 1


def split_data():
    data = np.load("data/performance_tokens_augmented.npy")
    np.random.seed(777)
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    data = data[idx]
    tlen, vlen = int(0.8 * len(data)), int(0.9 * len(data))

    print("Finish loading data...")

    data_train = data[:tlen]
    data_val = data[tlen:vlen]
    data_test = data[vlen:]

    np.save("data/performance_train.npy", np.array(data_train))
    np.save("data/performance_val.npy", np.array(data_val))
    np.save("data/performance_test.npy", np.array(data_test))


def get_paths_v2():
    train_paths, val_paths, test_paths = [], [], []

    with open("/data/maestro-v2.0.0/maestro-v2.0.0.json", "r+") as f:
        json_dict = json.load(f)
    
    for _, elem in tqdm(enumerate(json_dict), total=len(json_dict)):
        if elem["split"] == "train":
            train_paths.append("/data/maestro-v2.0.0/" + elem["midi_filename"])
        elif elem["split"] == "test":
            test_paths.append("/data/maestro-v2.0.0/" + elem["midi_filename"])
        else:
            val_paths.append("/data/maestro-v2.0.0/" + elem["midi_filename"])
    
    print(len(train_paths), len(val_paths), len(test_paths))
    return train_paths, val_paths, test_paths

        

get_augmented_data()
# split_data()
# train_paths, val_paths, test_paths = get_paths_v2()
# print("Running val...")
# get_tokens(val_paths, "val_v2")
# print("Running test...")
# get_tokens(test_paths, "test_v2")
# print("Running train...")
# get_tokens(train_paths, "train_v2")



        
        
