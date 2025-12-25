function accuracy = calculate_accuracy(true_labels, pred_labels)
true_labels = true_labels(:);
pred_labels = pred_labels(:);
mapped_labels = best_map(true_labels, pred_labels);
accuracy = sum(true_labels == mapped_labels) / length(true_labels);
end