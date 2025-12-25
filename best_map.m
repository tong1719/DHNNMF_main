function mapped_labels = best_map(true_labels, pred_labels)
true_labels = true_labels(:);
pred_labels = pred_labels(:);
true_unique = unique(true_labels);
pred_unique = unique(pred_labels);
n_true = length(true_unique);
n_pred = length(pred_unique);
C = zeros(n_true, n_pred);
for i = 1:n_true
    for j = 1:n_pred
        C(i, j) = sum((true_labels == true_unique(i)) & (pred_labels == pred_unique(j)));
    end
end
assignment = zeros(1, n_pred);
used = false(1, n_true);
[C_sorted, idx] = sort(C(:), 'descend');
[rows, cols] = ind2sub(size(C), idx);
for k = 1:length(C_sorted)
    i = rows(k);
    j = cols(k);
    if ~used(i) && assignment(j) == 0
        assignment(j) = i;
        used(i) = true;
    end
end
unassigned_pred = find(assignment == 0);
unassigned_true = find(~used);
if ~isempty(unassigned_pred) && ~isempty(unassigned_true)
    for j = 1:min(length(unassigned_pred), length(unassigned_true))
        assignment(unassigned_pred(j)) = unassigned_true(j);
    end
end
mapped_labels = zeros(size(pred_labels));
for j = 1:n_pred
    if assignment(j) > 0 && assignment(j) <= length(true_unique)
        mapped_labels(pred_labels == pred_unique(j)) = true_unique(assignment(j));
    else
        mapped_labels(pred_labels == pred_unique(j)) = pred_unique(j);
    end
end
end