function nmi = calculate_nmi(true_labels, pred_labels)
true_labels = true_labels(:);
pred_labels = pred_labels(:);
n = length(true_labels);
[~, ~, true_labels] = unique(true_labels);
[~, ~, pred_labels] = unique(pred_labels);
max_true = max(true_labels);
max_pred = max(pred_labels);
joint = zeros(max_true, max_pred);
for i = 1:n
    joint(true_labels(i), pred_labels(i)) = joint(true_labels(i), pred_labels(i)) + 1;
end
joint = joint / n;
p_true = sum(joint, 2);
p_pred = sum(joint, 1);
H_true = -sum(p_true .* log2(p_true + eps));
H_pred = -sum(p_pred .* log2(p_pred + eps));
MI = 0;
for i = 1:max_true
    for j = 1:max_pred
        if joint(i, j) > 0
            MI = MI + joint(i, j) * log2(joint(i, j) / (p_true(i) * p_pred(j) + eps));
        end
    end
end
if H_true * H_pred == 0
    nmi = 0;
else
    nmi = MI / sqrt(H_true * H_pred);
end
end