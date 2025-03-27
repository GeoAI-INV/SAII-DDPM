'by Hongling Chen, Xian Jiaotong university'

import torch
import torch.nn.functional as F


class ImpedanceOperator():
    def __init__(self, wav):
        self.wav = wav

    def DIFFZ(self, z):  # nonlinear operator
        nz = z.shape[2]
        tmp1 = z[..., 1:nz, :]
        tmp2 = z[..., nz - 1:nz, :]
        tmp3 = torch.cat((tmp1, tmp2), axis=2)
        DZ = (tmp3 - z) / (tmp3 + z)
        return DZ

    def DIFFZ_linear(self, z):  # linear operator
        nz = z.shape[2]
        S = torch.diag(0.5 * torch.ones(nz - 1), diagonal=1) - torch.diag(
            0.5 * torch.ones(nz - 1), diagonal=-1)
        S[0] = S[-1] = 0
        DZ = torch.matmul(S.to('cuda'), z)
        return DZ

    def forward(self, z):
        WEIGHT = torch.tensor(self.wav.reshape(1, 1, self.wav.shape[0], 1)).type(torch.cuda.FloatTensor)
        For_syn = F.conv2d(self.DIFFZ(torch.exp(z)), WEIGHT, stride=1, padding='same')
        return For_syn

    def forward_linear(self, z):
        WEIGHT = torch.tensor(self.wav.reshape(1, 1, self.wav.shape[0], 1)).type(torch.cuda.FloatTensor)
        For_syn = F.conv2d(self.DIFFZ_linear(z), WEIGHT, stride=1, padding='same')
        return For_syn











