from .resi_score_model import ResiLevelTensorProductScoreModel
from .DDPM_resi_score_model import ResiLevelTensorProductScoreModel as DDPM_ResiLevelTensorProductScoreModel 
import torch

class ScoreModel(torch.nn.Module):
    def __init__(self, args):
        super(ScoreModel, self).__init__()
        self.enn = ResiLevelTensorProductScoreModel(args)
        
    def forward(self, data):
        return self.enn(data)

def get_model(args):
    return ScoreModel(args)

class DDPM_ScoreModel(torch.nn.Module):
    def __init__(self, args):
        super(DDPM_ScoreModel, self).__init__()
        self.enn = DDPM_ResiLevelTensorProductScoreModel(args)
        
    def forward(self, data):
        return self.enn(data)

def DDPM_get_model(args):
    return DDPM_ScoreModel(args)

