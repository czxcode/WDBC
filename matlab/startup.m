%% 项目路径初始化
addpath(genpath(fullfile(pwd, 'matlab'))); 
if ~exist(fullfile(pwd, 'outputs', 'matlab_results'), 'dir')
    mkdir(fullfile(pwd, 'outputs', 'matlab_results'));
end