clc;
clear all;
close all;
%%

% old data
old_data_path='../../qsm_data_making/qsm_preprocessing/QSM_Preprocessing_updated/9VolumeData/';

ls(old_data_path);
%%
for i=1:9
    old_file_name=strcat(old_data_path,'IMG_Sub',num2str(i),'.mat');
    disp(old_file_name);
    patient_i_v1=load(old_file_name).IMG;
    disp(size(patient_i_v1))
end
%%
% new data
new_data_path='../../Data-SNU/'
ls(new_data_path)



%%

patient_name='train_subject1/'
file_name=strcat(new_data_path,patient_name,'IMG.mat');
patient_01_v2=load(file_name).IMG;
disp(size(patient_01_v2))
%%

patient_02_v1=load('../../qsm_data_making/qsm_preprocessing/QSM_Preprocessing_updated/9VolumeData/IMG_Sub2.mat').IMG;
% new data
patient_02_v2=load('../../Data-SNU/test_subject2/IMG.mat').IMG;


disp(size(patient_02_v1))
disp(size(patient_02_v2))

%%
%%
patient_name='validation/'
file_name=strcat(new_data_path,patient_name,'IMG.mat');
patient_01_v2=load(file_name).IMG;

image=real(IMG);





%%
clc;
clear all;
close all;

new_data_path='../../Data-SNU/'
ls(new_data_path)
patient_name='train_subject1/'
file_name=strcat(new_data_path,patient_name,'phscos.mat');


load(file_name)
image=multicos2;
image=image(:,:,:,1);

a=round(size(image,3)/2);
b=round(size(image,2)/2);
c=round(size(image,1)/2);  


% axial view of the cosmos

im1 = imrotate((squeeze(image(:,:,a))),-90);
im4 = imrotate((squeeze(image(:,:,a))),-90);

im2 = imrotate((squeeze(image(:,b,:))), 90);
im3 = imrotate((squeeze(image(c,:,:))), 90);        
im5 = imrotate((squeeze(image(:,b,:))), 90);
im6 = imrotate((squeeze(image(c,:,:))), 90);


%%
figure('Position', [1 1 1200 600],'Visible', 'on');        
subplot(2,3,1);
colormap('gray');
imagesc(im1,[-0.1, 0.1]);
xlabel('Axial view');
colorbar;
subplot(2,3,2);
colormap('gray');
imagesc(im2,[-0.1, 0.1]);
xlabel('coronal view');
colorbar;
subplot(2,3,3);
colormap('gray');
imagesc(im3,[-0.1, 0.1]);
xlabel('Sagittal');
colorbar;
subplot(2,3,4);
colormap('gray');
imagesc(im4,[-0.1, 0.1]);
xlabel('axial view');
colorbar;
subplot(2,3,5);
colormap('gray');
imagesc(im5,[-0.1, 0.1]);
xlabel('coronal view');
colorbar;
subplot(2,3,6);
colormap('gray');
imagesc(im6,[-0.1, 0.1]);
xlabel('Sagittal');
colorbar;   







