#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 23:35:41 2023

@author: venkatesh
"""

# argument parser for system call
import os

# Execute the "mkdir" command to create a new Directory
# exit_status = os.system("python testing_BASnet_and_u2_net.py --epoch 21")

#%%
for epoch_no in range(0,100):
    exit_status = os.system("python Neumann_QSM_test_for_k_fold_updated.py --epoch "+str(epoch_no))

#%%

import subprocess

command = 'matlab -nodisplay -nosplash -nodesktop -r "run(\'./measures/k_fold_get_metrics.m\');exit;"'
subprocess.run(command, shell=True)
