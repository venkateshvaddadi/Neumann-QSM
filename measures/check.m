function check()

    num_cases = 6;  
    orients=5;

    ssim_scores = zeros(6,5);
    rmse_scores = zeros(6,5);
    psnr_scores = zeros(6,5);
    hfen_scores = zeros(6,5);

    savefigure = 1;


    experiements_folder="../savedModels/MODL_QSM_MODELS/";
    experiment="27_Sep_11_15_am_model_K2/";
    full_path_for_results=strcat(experiements_folder,experiment,'predictions/modl-net');
    disp(full_path_for_results)
    
    load(strcat(full_path_for_results,'-11.mat'))
    disp(size(modl))
    
    for i=1:6
    disp(i)
        for j=1:5
        idx = strcat('-',num2str(i),num2str(j),'.mat');
        cos=load(strcat('../data/Data/Testing_Data/cos/cos',idx));
        modl=load(strcat(full_path_for_results,idx));
        phs=load(strcat('../data/Data/Testing_Data/phs/phs' ,idx));
        msk=load(strcat('../data/Data/Testing_Data/msk/msk' ,idx));
        modl = squeeze(modl);
        
        ssim_scores(i,j) = round(compute_ssim(modl,cos), 4);      

        
        end
    end
        
end

