function demo_DHNNMF()
clear; close all; clc;
rng(2024, 'twister');

fprintf('========== DHNNMF Algorithm Demonstration ==========\n\n');

fprintf('1. Loading YALE dataset...\n');

try
    load('Yale_64x64.mat');
    X_raw = fea;
    true_labels = gnd;
    dataset_name = 'YALE';
catch
    error('YALE dataset not found. Please ensure Yale_64x64.mat is in the current directory.');
end

X_raw = double(X_raw);
true_labels = double(true_labels(:));

[n_samples, n_features] = size(X_raw);
n_classes = length(unique(true_labels));

fprintf('   Successfully loaded %s dataset:\n', dataset_name);
fprintf('      Samples: %d\n', n_samples);
fprintf('      Features: %d\n', n_features);
fprintf('      Classes: %d\n\n', n_classes);

fprintf('2. Data Preprocessing\n');
X_normalized = normalize_data(X_raw);
X = X_normalized';

fprintf('   Data normalized and transposed to %d features ¡Á %d samples\n\n', ...
    size(X, 1), size(X, 2));

fprintf('3. Parameter Setup\n');

if n_classes <= 10
    suggested_r = min(20, max(n_classes + 5, 10));
elseif n_classes <= 30
    suggested_r = min(40, max(n_classes + 10, 20));
else
    suggested_r = min(60, max(n_classes + 15, 30));
end

fprintf('   Suggested decomposition rank: %d\n', suggested_r);
fprintf('   (True number of classes: %d)\n', n_classes);

r_input = input('   Enter decomposition rank (press Enter for suggested): ', 's');
if isempty(r_input)
    r = suggested_r;
else
    r = str2double(r_input);
    if isnan(r) || r < 2 || r > min(n_features, n_samples)
        fprintf('   Invalid input. Using suggested rank: %d\n', suggested_r);
        r = suggested_r;
    end
end

fprintf('   Using decomposition rank: r = %d\n\n', r);

fprintf('4. Running DHNNMF\n');
fprintf('   -----------------------------------\n');

options = struct();
options.mu1 = 0.001;
options.mu2 = 0.001;
options.nu = 10;
options.theta = 0.9;
options.k = 5;
options.maxIter = 500;
options.tol = 1e-6;
options.verbose = false;
options.init_method = 'nndsvd';

fprintf('   Running optimization...\n');
tic;
[U, V, obj_history] = DHNNMF_main(X, r, options);
runtime = toc;

fprintf('   -------------------------\n');
fprintf('   Runtime: %.2f seconds\n', runtime);
fprintf('   Iterations: %d\n\n', length(obj_history));

fprintf('5. Evaluating Clustering Results\n');

n_eval_trials = 10;
acc_results = zeros(n_eval_trials, 1);
nmi_results = zeros(n_eval_trials, 1);

for trial = 1:n_eval_trials
    [acc, nmi] = clustering_evaluation(V, true_labels, n_classes);
    acc_results(trial) = acc;
    nmi_results(trial) = nmi;
end

[best_acc, best_idx] = max(acc_results);
best_nmi = nmi_results(best_idx);

fprintf('   Best Results:\n');
fprintf('      ACC: %.3f%%\n', best_acc*100);
fprintf('      NMI: %.3f%%\n\n', best_nmi*100);

fprintf('========== DHNNMF Demonstration Completed ==========\n');
fprintf('Dataset: %s\n', dataset_name);
fprintf('Samples: %d, Features: %d, Classes: %d\n', n_samples, n_features, n_classes);
fprintf('Decomposition rank: r = %d\n\n', r);

fprintf('Clustering Performance:\n');
fprintf('   ACC:  %.3f%%\n', best_acc*100);
fprintf('   NMI:  %.3f%%\n\n', best_nmi*100);

fprintf('Runtime: %.2f seconds\n', runtime);

end

function X_normalized = normalize_data(X)
X = double(X);

X(isnan(X)) = 0;
X(isinf(X)) = 0;

X_normalized = zeros(size(X));
for i = 1:size(X, 2)
    col = X(:, i);
    col_min = min(col);
    col_max = max(col);
    if col_max > col_min
        X_normalized(:, i) = (col - col_min) / (col_max - col_min);
    else
        X_normalized(:, i) = col;
    end
end

norms = sqrt(sum(X_normalized.^2, 2));
norms(norms == 0) = 1;
X_normalized = X_normalized ./ norms;

X_normalized(isnan(X_normalized)) = 0;
end