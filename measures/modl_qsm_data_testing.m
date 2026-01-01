clc;
close;
clear;

%%
path='../../QSM_data/data_for_experiments/modl_qsm_data/data/QSM_challenge_data/'
ls(path)


%%

load(strcat(path,'/test_data.mat'))


%%

temp=figure('Position', [1 1 1200 600],'Visible', 'on');

subplot(1,3,1)
colormap('gray')
imagesc(phi(:,:,80))
xlabel('phs')
colorbar;

subplot(1,3,2)
colormap('gray')
imagesc(mask(:,:,80))
xlabel('msk')
colorbar;

subplot(1,3,3)
colormap('gray')
imagesc(labels(:,:,80),[-0.1,0.1])
xlabel(' cos')
colorbar;

%%
clc;
close;
clear;

load('../../QSM_data/data_for_experiments/qsm_2016_recon_challenge/data_12orientations/phs_all.mat')
load('../../QSM_data/data_for_experiments/qsm_2016_recon_challenge/data_12orientations/R_tot.mat')

