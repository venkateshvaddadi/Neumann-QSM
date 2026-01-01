% MATLAB script to process command line arguments
% my_matlab_script.m

% Read command line arguments
epoch = str2double(getenv('ARG1'));
disp(epoch)

for i=1:epoch
	disp(i)
end
