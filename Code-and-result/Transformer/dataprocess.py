import scipy.io as scio
import os

lables=scio.loadmat("reT2回波/BZ1_1_3_ECHO.mat")['BZ1_1_3_ECHO']
features=scio.loadmat('T2岩心已归位/BZ1_1_3_Amp.mat')['BZ1_1_3_Amp']


print(lables)
print(features)