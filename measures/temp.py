#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 00:36:52 2023

@author: venkatesh
"""

import subprocess

# Define the MATLAB script and arguments
matlab_script = "my_script.m"
arg1 = "42"
arg2 = "3.14"

# Construct the system call
command = f"matlab -nodisplay -nosplash -nodesktop -r \"setenv('ARG1', '{arg1}'); setenv('ARG2', '{arg2}'); run('{matlab_script}'); exit;\""

# Execute the MATLAB script from Python
subprocess.call(command, shell=True)
