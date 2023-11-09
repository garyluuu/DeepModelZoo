import torch
import numpy as np
import os
import deepMZ

from ReadInput import TrainParamReader

def model_count(model):
    return sum(p.numel() for p in model.parameters())

def find_file_in_savePath(file_name_to_find, savePath):
    for root, _, files in os.walk(savePath):
        for file_name in files:
            if file_name == file_name_to_find:
                return os.path.join(root, file_name)
    return None 

def load_model(model_path, config: TrainParamReader):
    net = getattr(deepMZ.temporal.nn, config.nn)(**config.nnParams)
    if config.wrapper:
        net = deepMZ.temporal.wrapper(net, **config.wrapperParams)
    if os.path.exists(model_path):
        # net.load_state_dict(torch.load(model_path))
        net.load_state_dict(torch.load(model_path))
        print("Successfully loaded model")
    else:
        print("Model not found")
        exit()
    return net

if __name__ == "__main__":
    import sys

    config = TrainParamReader(sys.argv[1])

    model_paths = config.savePath
    file_name_to_find = f"{config.nn}_model-501epoches@500epoch-{config.wrapperParams['evolveLen']}evolveLen_{config.wrapperParams['inputFnArgs']['inputLen']}inputLen)"
    # file_name_to_find = f"727_model-501epoches@500epoch-{config.wrapperParams['evolveLen']}evolveLen"    
    folder_name = f"{config.nn}_{config.wrapperParams['evolveLen']}evolveLen_{config.wrapperParams['inputFnArgs']['inputLen']}inputLen"
    savePath = os.path.join(config.savePath,folder_name)
    target_file_path = find_file_in_savePath(file_name_to_find, savePath)

    model = load_model(target_file_path, config)

    print(model_count(model))