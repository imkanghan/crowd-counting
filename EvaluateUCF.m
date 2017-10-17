close all
clear all
clc

addpath('MRF');
MRFParams = single([3500 1000 0.85]);

load('data/features_UCF.mat');
load('data/partition_UCF.mat');
load('data/predictions_UCF.mat');
load('data/ground_truth_UCF.mat');
n = numel(counts);
finalcount = zeros(n, 1);
partition = partition + 1;

for i = 1 : 5
    index = partition(i, :);
    patchPredictions = predictions{i};
    
    k = 1;
    for j = 1 : numel(index)
        patchCount = counts{index(j)};

        [height, width] = size(patchCount);
        p = reshape(patchPredictions(k: k + height * width - 1), width, height);
        k = k + height * width;

        % The marginal data of the predicted count matrix is 0 after apply MRF, 
        % so first extending the predicted count matrix by copy marginal data.
        p = uint8(p)';
        p = [p(1,:); p];
        p = [p ;p(end,:)];
        p = [p(:, 1) p];
        p = [p p(:, end)];
        % apply MRF
        p = MRF(p, MRFParams);
        p = p(2:end-1, 2: end-1);

        finalcount(index(j)) = FinalCount(p);
    end
end

MAE = mean(abs(finalcount - gt));
MSE = mean((finalcount - gt).^2)^0.5;
fprintf('MAE: %f\n', MAE);
fprintf('MSE: %f\n', MSE);
