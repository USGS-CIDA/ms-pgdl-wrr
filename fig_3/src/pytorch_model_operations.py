import torch

def saveModel(model_state, optimizer_state, save_path):
    state = {'state_dict': model_state,
                    'optimizer': optimizer_state }
    torch.save(state, save_path)