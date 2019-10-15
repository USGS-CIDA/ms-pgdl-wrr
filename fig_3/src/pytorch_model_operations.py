# https://github.com/jdwillard19/lake_modeling/blob/e51ef62b26cd280217272b295a4c39a8a8bed551/src/models/pytorch_model_operations.py

import torch

def saveModel(model_state, optimizer_state, save_path):
    state = {'state_dict': model_state,
                    'optimizer': optimizer_state }
    torch.save(state, save_path)