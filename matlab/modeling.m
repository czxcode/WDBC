% MATLAB建模验证模块 - WDBC乳腺癌分类
% 功能：逻辑回归建模、性能评估、结果可视化
% 在脚本开头强制清理内存
clear all
close all
clc

% 预分配大型数组（示例：处理10万样本）
n_samples = 100000;
X = zeros(n_samples, 30);  % 预分配代替动态扩展

% 分块处理大数据
chunk_size = 5000;
for i = 1:chunk_size:n_samples
    end_idx = min(i+chunk_size-1, n_samples);
    process_chunk(X(i:end_idx,:)); 
end
%% 环境初始化
clear; close all; clc;
warning('off', 'stats:glmfit:BadScaling'); % 关闭尺度警告

% 配置参数结构体
config = struct();
config.DataPath = fullfile('..', 'data', 'processed', 'processed_data.csv'); % 数据路径
config.OutputDir = fullfile('..', 'outputs', 'matlab_results'); % 输出目录
config.RandomSeed = 42;       % 随机种子
config.TrainRatio = 0.7;      % 训练集比例
config.Verbose = true;        % 显示过程信息

% 创建输出目录
if ~exist(config.OutputDir, 'dir')
    mkdir(config.OutputDir);
end

%% 数据加载与预处理
try
    % 读取预处理后的CSV数据
    % 注意：MATLAB可能自动转换列名中的下划线，此处保持原始列名
    opts = detectImportOptions(config.DataPath);
    opts.PreserveVariableNames = true; % 保持列名不变
    data = readtable(config.DataPath, opts);
    
    % 数据验证
    assert(ismember('Diagnosis', data.Properties.VariableNames), ...
        '诊断结果列 Diagnosis 缺失');
    required_features = {'radius_mean', 'texture_mean', 'perimeter_mean', ...
        'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', ...
        'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean'};
    assert(all(ismember(required_features, data.Properties.VariableNames)), ...
        '必需特征列缺失');
    
    % 准备特征矩阵和响应变量
    X = data{:, required_features};  % 选择核心特征
    y = categorical(data.Diagnosis, {'B', 'M'}, {'Benign', 'Malignant'}); % 转换为分类变量
    
    if config.Verbose
        fprintf('数据加载成功，维度: %d样本 x %d特征\n', size(X));
        tabulate(y) % 显示类别分布
    end
    
catch ME
    error('数据加载失败: %s', ME.message);
end

%% 数据拆分（分层抽样）
rng(config.RandomSeed); % 固定随机种子
cv = cvpartition(y, 'HoldOut', 1 - config.TrainRatio);

% 训练集
X_train = X(training(cv), :);
y_train = y(training(cv), :);

% 测试集
X_test = X(test(cv), :);
y_test = y(test(cv), :);

if config.Verbose
    fprintf('数据集拆分:\n  训练集: %d样本\n  测试集: %d样本\n', ...
        sum(training(cv)), sum(test(cv)));
end

%% 逻辑回归建模
try
    % 构建逻辑回归模型
    % 使用 'Binomial' 表示二分类问题
    model = fitglm(X_train, y_train, ...
        'Distribution', 'binomial', ...
        'CategoricalVars', [], ... % 无分类特征（所有特征已标准化）
        'VarNames', [required_features, {'Diagnosis'}], ...
        'Intercept', true);
    
    % 显示模型摘要
    if config.Verbose
        disp('模型系数估计:');
        disp(model.Coefficients);
    end
    
    % 保存模型
    modelPath = fullfile(config.OutputDir, 'wdbc_logreg_model.mat');
    save(modelPath, 'model');
    fprintf('模型已保存至: %s\n', modelPath);
    
catch ME
    error('模型训练失败: %s', ME.message);
end

%% 模型评估
% 预测测试集概率
pred_probs = predict(model, X_test); % 预测概率值（0-1之间）

% 转换预测结果为分类
pred_labels = categorical(pred_probs > 0.5, [true, false], {'Malignant', 'Benign'});

% 计算性能指标
accuracy = sum(pred_labels == y_test) / numel(y_test);
confMat = confusionmat(y_test, pred_labels, 'Order', {'Malignant', 'Benign'});
precision = confMat(1,1) / sum(confMat(:,1));
recall = confMat(1,1) / sum(confMat(1,:));
f1Score = 2 * (precision * recall) / (precision + recall);

% 显示评估结果
fprintf('\n=== 模型性能 ===\n');
fprintf('准确率: %.2f%%\n', accuracy*100);
fprintf('精确率: %.2f%%\n', precision*100);
fprintf('召回率: %.2f%%\n', recall*100);
fprintf('F1分数: %.2f\n', f1Score);

%% ROC曲线分析
try
    % 计算ROC曲线参数
    [rocX, rocY, ~, auc] = perfcurve(y_test, pred_probs, 'Malignant');
    
    % 绘制ROC曲线
    figure('Position', [100 100 800 600]);
    plot(rocX, rocY, 'b-', 'LineWidth', 2);
    hold on;
    plot([0 1], [0 1], 'k--'); % 随机猜测基线
    xlabel('假阳性率 (FPR)');
    ylabel('真阳性率 (TPR)');
    title(sprintf('ROC曲线 (AUC = %.3f)', auc));
    legend(sprintf('逻辑回归 (AUC=%.3f)', auc), '随机猜测', ...
        'Location', 'southeast');
    grid on;
    
    % 保存图像
    rocFigPath = fullfile(config.OutputDir, 'roc_curve.png');
    exportgraphics(gcf, rocFigPath, 'Resolution', 300);
    fprintf('ROC曲线已保存至: %s\n', rocFigPath);
    
catch ME
    warning('ROC分析失败: %s', ME.message);
end

%% 混淆矩阵可视化
try
    figure('Position', [100 100 600 500]);
    confusionchart(y_test, pred_labels, ...
        'Title', '混淆矩阵', ...
        'RowSummary', 'row-normalized', ...
        'ColumnSummary', 'column-normalized');
    
    % 保存图像
    confMatPath = fullfile(config.OutputDir, 'confusion_matrix.png');
    exportgraphics(gcf, confMatPath, 'Resolution', 300);
    fprintf('混淆矩阵已保存至: %s\n', confMatPath);
    
catch ME
    warning('混淆矩阵生成失败: %s', ME.message);
end

%% 特征重要性分析
try
    % 提取标准化系数
    coefficients = model.Coefficients.Estimate(2:end); % 排除截距项
    
    % 生成特征重要性图
    figure('Position', [100 100 900 600]);
    barh(coefficients);
    yticks(1:numel(required_features));
    yticklabels(required_features);
    xlabel('回归系数值');
    title('特征重要性（标准化系数）');
    grid on;
    
    % 保存图像
    featImpPath = fullfile(config.OutputDir, 'feature_importance.png');
    exportgraphics(gcf, featImpPath, 'Resolution', 300);
    fprintf('特征重要性图已保存至: %s\n', featImpPath);
    
catch ME
    warning('特征分析失败: %s', ME.message);
end

%% 保存评估结果
results = struct();
results.Accuracy = accuracy;
results.Precision = precision;
results.Recall = recall;
results.F1Score = f1Score;
results.AUC = auc;
results.ConfusionMatrix = confMat;

save(fullfile(config.OutputDir, 'evaluation_results.mat'), 'results');
fprintf('\n评估结果已保存至: %s\n', fullfile(config.OutputDir, 'evaluation_results.mat'));

% 清理工作区
clearvars -except config model results;
% 在MATLAB命令窗口执行（若能短暂运行）
diag_tool = matlab.diag.SystemDiagnostic;
diag_tool.run;
diag_tool.save('crash_report.zip');