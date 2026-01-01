#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 14:28:30 2023

@author: venkatesh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 11:02:28 2022

@author: venkatesh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 10:24:27 2021

@author: venkatesh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 17:44:32 2021

@author: venkatesh
"""


import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch import nn



import numpy as np
import time
import scipy.io
import tqdm
import matplotlib.pyplot as plt
import scipy.io
import os
#%%
from qsm_modules.qsm_data_loader.QSM_Dataset_updated import mydataloader
from qsm_modules.qsm_dw_models.model_for_dw_deepqsm_lambda_p_trainable import DeepQSM
from qsm_modules.qsm_dw_models.model_for_dw_QSMnet_lambda_p_trainable import QSMnet
from qsm_modules.qsm_loss_modules.loss import *
from qsm_modules.qsm_dw_models.WideResnet import WideResNet
from qsm_modules.qsm_dw_models.model_for_dw_normal_cnn_lambda_p_trainable import Dw
#%%

matrix_size = [176,176, 160]
voxel_size = [1,  1,  1]

#%%
#loading the model\
K_unrolling=4
batch_size=1
device_id=0
is_data_normalized=True

model='WideResNet'

#%%

epoch=83

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="epoch number for testing ",type=int,default=epoch)
args = parser.parse_args()

epoch=args.epoch;

print('epoch:',epoch)
#%%
data_source='given_single_patient_data'
Training_patient_no=1
data_source_no=2

if(data_source=='generated_data'):

    raw_data_path='../QSM_data/data_for_experiments/generated_data/raw_data_noisy_sigma_0.05/'
    data_path='../QSM_data/data_for_experiments/generated_data/data_source_1/'
    patients_list =[7,32,9,10]

elif(data_source=='given_data'):

    raw_data_path='../QSM_data/data_for_experiments/given_data/raw_data_names_modified/'

    if(data_source_no==1):
        patients_list =[7,8,9,10,11,12]
        data_path='../QSM_data/data_for_experiments/given_data/data_source_1/'
        csv_path='../QSM_data/data_for_experiments/given_data/data_source_1//'
        data_path='../QSM_data/data_for_experiments/given_data/data_as_patches/'

    elif(data_source_no==2):
        patients_list =[10,11,12,1,2,3]
        data_path='../QSM_data/data_for_experiments/given_data/data_source_2/'
        csv_path='../QSM_data/data_for_experiments/given_data/data_source_2//'
        data_path='../QSM_data/data_for_experiments/given_data/data_as_patches/'

    elif(data_source_no==3):
        patients_list =[1,2,3,4,5,6]
        data_path='../QSM_data/data_for_experiments/given_data/data_source_3/'
        csv_path='../QSM_data/data_for_experiments/given_data/data_source_3//'
        data_path='../QSM_data/data_for_experiments/given_data/data_as_patches/'

    elif(data_source_no==4):
        patients_list =[4,5,6,7,8,9]
        data_path='../QSM_data/data_for_experiments/given_data/data_source_4/'
        csv_path='../QSM_data/data_for_experiments/given_data/data_source_4//'
        data_path='../QSM_data/data_for_experiments/given_data/data_as_patches/'

elif(data_source=='generated_noisy_data'):
    raw_data_path='../QSM_data/data_for_experiments/generated_data/raw_data/'
    data_path='../QSM_data/data_for_experiments/generated_data/data_source_1/'
    patients_list =[7,32,9,10]



elif(data_source=='generated_undersampled_data'):
    raw_data_path='../QSM_data/data_for_experiments/generated_data/sampling_data/sampled_0.05//'
    data_path='../QSM_data/data_for_experiments/generated_data/data_source_1/'
    patients_list =[7,32,9,10]

elif(data_source=='given_single_patient_data'):
    raw_data_path='../QSM_data/data_for_experiments/given_data/raw_data_names_modified/'
    if(Training_patient_no==1):
        csv_path='../QSM_data/data_for_experiments/given_data/single_patient/patient_1/'
        data_path='../QSM_data/data_for_experiments/given_data/data_as_patches/'
        patients_list =[2,3,4,5,7,8,9,10,11,12]
    elif(Training_patient_no==2):
        csv_path='../QSM_data/data_for_experiments/given_data/single_patient/patient_2/'
        data_path='../QSM_data/data_for_experiments/given_data/data_as_patches/'
        patients_list =[1,3,4,5,7,8,9,10,11,12]

    elif(Training_patient_no==3):
        csv_path='../QSM_data/data_for_experiments/given_data/single_patient/patient_3/'
        data_path='../QSM_data/data_for_experiments/given_data/data_as_patches/'
        patients_list =[1,2,4,5,7,8,9,10,11,12]
    if(Training_patient_no==4):
        csv_path='../QSM_data/data_for_experiments/given_data/single_patient/patient_4/'
        data_path='../QSM_data/data_for_experiments/given_data/data_as_patches/'
        patients_list =[1,2,3,5,7,8,9,10,11,12]
    patients_list =[7,8,9,10,11,12]

import os
print(os.listdir(data_path))
print(os.listdir(raw_data_path))
print('csv_path',csv_path)
print('data_path:',data_path)
print('raw_data_path',raw_data_path)
#%%

experiments_folder="savedModels/Neumann_QSM_MODELS_dw_QSMnet/experiments_on_given_data/dw_WideResNet/full_data_training_without_sampling/single_patient/"
experiment_name="Oct_30_10_53_pm_model_K_4_given_single_patient_data_dw_WideResNet_patient_1//"

model_name="Spinet_QSM_model_"+str(epoch)+"_.pth"
model_path=experiments_folder+"/"+experiment_name+"/"+model_name
print('model_path:',model_path)

try:
    os.makedirs(experiments_folder+"/"+experiment_name+"/output_csv")
except:
    print("Exception...")

#%%

if(model=='deepqsm'):
    dw = DeepQSM().cuda(device_id)
elif(model=='QSMnet'):
    dw=QSMnet().cuda(device_id)
elif(model=='WideResNet'):
    dw=WideResNet().cuda(device_id)
elif(model=='simple_cnn'):
    dw=Dw().cuda(device_id)
#%%
print("dw.Eta_val",dw.Eta_val)
print("dw.p",dw.p)
#%%
dw.load_state_dict(torch.load(model_path))
#dw.load_state_dict(torch.load('./savedModels/Spinet_QSM_MODELS_dw_QSMnet_loss_l1_lambda_p_trainging/experiments_on_given_data/dw_WideResNet/ablation_study/Dec_02_06_12_pm_model_K_1_B_2_N_2000_dw_WideResNet_data_source_1_p_2.0/Spinet_QSM_model_32_.pth'))


#%%
#dw = torch.nn.DataParallel(dw, device_ids=[device_id])  


dw.eval()
dw = dw.cuda(device_id)
#%%
print("Evaluation happening")
print('dw.Eta_val',dw.Eta_val)
print('dw.p',dw.p)

#%%
last_string=model_path.split("/")[-1]
directory=model_path.replace(last_string,"")

print('directory:',directory)
print(os.listdir(directory))

#%%
ss = sobel_kernel()
ss=ss.float()
ss = ss.cuda(device_id)

dk = dipole_kernel(matrix_size, voxel_size, B0_dir=[0, 0, 1])
dk = torch.unsqueeze(dk, dim=0)
print(dk.shape)

dk=dk.float().cuda(device_id)
Dk_square=torch.multiply(dk, dk)
Dk_square=Dk_square.cuda(device_id)
#%%
# define the train data stats

stats = scipy.io.loadmat(csv_path+'/csv_files/tr-stats.mat')


if(not is_data_normalized):
    sus_mean=0
    sus_std=1
    print('\n\n data is not normalized..................\n\n ')

else:
    stats = scipy.io.loadmat(csv_path+'/csv_files/tr-stats.mat')
    sus_mean= torch.tensor(stats['out_mean']).cuda(device_id)
    sus_std = torch.tensor(stats['out_std' ]).cuda(device_id)
    print(sus_mean,sus_std)


#%%

def tic():
    # Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        #print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
        print(str(time.time() - startTime_for_tictoc) )
    else:
        print("Toc: start time not set")

#%%

def z_real_to_z_complex(z_real):
  z_complex_recon=torch.complex(z_real[:,0,:,:,:].unsqueeze(1),z_real[:,1,:,:,:].unsqueeze(1))
  return z_complex_recon

def z_complex_to_z_real(z_complex):
  z_real=z_complex.real
  z_imag=z_complex.imag
  z_real_recon=torch.cat([z_real,z_imag],axis=1)
  return z_real_recon



#%%

outdir = directory+'predictions_'+str(epoch)+"/"
print(outdir)
import os
try:
    os.makedirs(outdir)
except:
    print("aalready tested..")



#%%
# temp={}
dk_repeat = dk.repeat(batch_size,1,1,1,1)

with torch.no_grad():


    for i in patients_list:
        print("Patinte:"+str(i)+"\n")
        for j in range(1,6):

            phs=scipy.io.loadmat(raw_data_path+'/patient_'+str(i)+'/phs'+str(j)+'.mat')['phs']
            sus=scipy.io.loadmat(raw_data_path+'/patient_'+str(i)+'/cos'+str(j)+'.mat')['cos']
            msk=scipy.io.loadmat(raw_data_path+'/patient_'+str(i)+'/msk'+str(j)+'.mat')['msk']
            
            # for saving spinet_qsm_output_with_all_stages
            # temp['msk']=msk
            # temp['cos']=msk

            phs=torch.unsqueeze(torch.unsqueeze(torch.tensor(phs),0),0)
            sus=torch.unsqueeze(torch.unsqueeze(torch.tensor(sus),0),0)
            msk=torch.unsqueeze(torch.unsqueeze(torch.tensor(msk),0),0)
    
            phs=phs.cuda(device_id)
            sus=sus.cuda(device_id)
            msk=msk.cuda(device_id)
    
            tic()

            B_0_complex = torch.fft.ifftn(dk_repeat*torch.fft.fftn(phs,dim=[2,3,4]),dim=[2,3,4])
            B_0_complex=dw.Eta_val*(B_0_complex)
            B_0_complex=B_0_complex.cuda(device_id)
            B_0_complex=B_0_complex*msk;

            # initialize with zeros....
            
            B_k_complex=B_0_complex.clone()*msk;
            # print('B_k_complex.shape',B_k_complex.shape)

            B_k_complex_sum=B_0_complex.clone()*msk;
            # print('B_k_complex_sum.shape',B_k_complex_sum.shape)

            B_k_set=[B_0_complex.clone()*msk]

            for k in range(K_unrolling):
                print('k:',k)
                # term-1 calculation
                X_T_X=torch.fft.ifftn(dk_repeat*dk_repeat*torch.fft.ifftn(B_k_complex,dim=[2,3,4]),dim=[2,3,4])
                
                # print(X_T_X.shape)
                
                
                term_1_complex=B_k_complex- dw.Eta_val*X_T_X
                # print('term_1_complex.shape',term_1_complex.shape)
                # term-2 calculation adding regularization
                B_k_real=z_complex_to_z_real(B_k_complex)

                # data normalization 
                B_k_real=(B_k_real-sus_mean)/sus_std;
                R_X_real_term=dw(B_k_real)
                # data de-normalization 
                R_X_real_term=R_X_real_term*sus_std+sus_mean
                R_X_real_term=R_X_real_term*msk

                R_X_complex_term=z_real_to_z_complex(R_X_real_term)
                R_X_complex_term=R_X_complex_term*msk
                # print(R_X_complex_term.shape)
                #updated B_k_complex
                B_k_complex=term_1_complex-dw.Eta_val*R_X_complex_term
                B_k_complex=B_k_complex*msk

                B_k_set.append(B_k_complex.clone()*msk)

                B_k_complex_sum=B_k_complex_sum+B_k_complex
                B_k_complex_sum=B_k_complex_sum*msk

            x_k_complex = B_k_complex_sum * msk

            loss=total_loss_l1(chi=x_k_complex.real, y=sus, b=phs, d=dk, m=msk, sobel=ss)




            toc()
            
            
            
            
            
            
            x_k_cpu=(x_k_complex.real.detach().cpu().numpy())*(msk.detach().cpu().numpy() )
            mdic  = {"modl" : x_k_cpu}
            filename  = outdir + 'modl-net-'+ str(i)+'-'+str(j)+'.mat'
            scipy.io.savemat(filename, mdic)

            # spinet_qsm_output_with_all_stages
            # scipy.io.savemat(outdir + 'spinet_qsm_output_with_all_stages'+ str(i)+'-'+str(j)+'.mat', temp)
            # print(temp.keys())



#%%




print(torch.cat(B_k_set, 0).shape)

x_k_temp_1=torch.sum(torch.cat(B_k_set, 0),axis=0)
print(x_k_temp_1.shape)

x_k_temp_1=torch.zeros_like(B_0_complex);
for i in range(len(B_k_set)):
    print(i)
    x_k_temp_1=x_k_temp_1+B_k_set[i]*msk

x_k_temp_1=x_k_temp_1*msk
print(x_k_temp_1.shape)


print(total_loss_l1(chi=x_k_complex.real, y=x_k_temp_1.real, b=phs, d=dk, m=msk, sobel=ss))