clear all;
close all;

%%


file_name='noised_phs_5';
disp(file_name)
path='../../../QSM_venkatesh/MoDL_QSM_PyTorch/data/Data/Testing_Data/';
addpath(strcat(path,file_name))

%%

snr_list=zeros(6,5);
for i = 1:6
    for j=1:5
        file_path=strcat('phs-',num2str(i),num2str(j),'.mat');
        load(file_path);
        snr=getsnr(phs);
        snr_list(i,j)=snr;
        
    end
end
disp(snr_list)
disp(mean(snr_list(:)))
%%

%%

% snr calculation

function snr=getsnr(img)
    img=double(img(:));
    ima=max(img(:));
    imi=min(img(:));
    ims=std(img(:));
    snr=10*log((ima-imi)./ims);

end
