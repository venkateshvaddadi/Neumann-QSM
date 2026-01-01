clc; clear all; close all; addpath('./'); format short;

%%


test_path='../../MoDL_QSM_PyTorch/data/Data/Testing_Data/'
ls(test_path)

%%
for i=1:6
    for j=1:5
        idx = strcat('-',num2str(i),num2str(j),'.mat');
        file_name=strcat(test_path,'phs/phs' ,idx);
        target_file_name=strcat(test_path,'noised_phs_5_updated/phs' ,idx);
        
        disp(file_name)
        disp(target_file_name)
        load(file_name);
        load(strcat(test_path,'msk/msk' ,idx));
        disp(strcat(test_path,'msk/msk' ,idx))
        add_noise_dec=1;
        if add_noise_dec

                disp('hello')
                TE = 0.025;  % ms
                B0 = 3;   % T
                gyro = 2 * pi * 42.58;        
                scale_factor = TE * B0 * gyro;        
                phs = phs * scale_factor;



                SNRdB = 25;
                sgnl_power = std(phs(msk>0))^2;
                sigma_noise = sqrt(sgnl_power/(10^(SNRdB/10)))
                rng(45);
                N = sigma_noise*randn(size(phs));
                phs_noise = (phs + N).*single(msk);
                phs = phs_noise/scale_factor;
                save(target_file_name,'phs')
                disp(size(msk))
                clear phs;
                clear msk;


        end
    end 
end

%%
