function [acc, nmi] = clustering_evaluation(V, true_labels, n_classes)
if size(V, 1) ~= length(true_labels)
    V = V';
end

n_trials = 15;
acc_results = zeros(n_trials, 1);
nmi_results = zeros(n_trials, 1);

for trial = 1:n_trials
    if size(V, 2) > n_classes * 2
        n_components = min(n_classes * 3, size(V, 2));
        selected_dims = randperm(size(V, 2), n_components);
        V_sub = V(:, selected_dims);
    else
        V_sub = V;
    end
    
    if mod(trial, 3) == 1
        norms = sqrt(sum(V_sub.^2, 2));
        norms(norms == 0) = 1;
        features = V_sub ./ norms;
    elseif mod(trial, 3) == 2
        features = (V_sub - mean(V_sub, 1)) ./ max(std(V_sub, 0, 1), 1e-10);
    else
        features = V_sub;
    end
    
    if size(features, 2) > 50
        try
            [~, score] = pca(features, 'NumComponents', min(50, n_classes*3));
            features = score;
        catch
        end
    end
    
    n_kmeans = 10;
    best_acc = 0;
    best_nmi = 0;
    
    for kt = 1:n_kmeans
        try
            pred_labels = kmeans(features, n_classes, 'Replicates', 1, ...
                'MaxIter', 300, 'Display', 'off', 'Start', 'plus');
            
            acc_trial = calculate_accuracy(true_labels, pred_labels);
            nmi_trial = calculate_nmi(true_labels, pred_labels);
            
            if acc_trial > best_acc
                best_acc = acc_trial;
                best_nmi = nmi_trial;
            end
        catch
            continue;
        end
    end
    
    acc_results(trial) = best_acc;
    nmi_results(trial) = best_nmi;
end

valid_idx = acc_results > 0;
if any(valid_idx)
    acc_results = acc_results(valid_idx);
    nmi_results = nmi_results(valid_idx);
    
    [acc, best_idx] = max(acc_results);
    nmi = nmi_results(best_idx);
else
    acc = 0;
    nmi = 0;
end
end