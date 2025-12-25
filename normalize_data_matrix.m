function X_normalized = normalize_data_matrix(X)
X = double(X);
[n_samples, n_features] = size(X);
X_normalized = zeros(size(X));
for j = 1:n_features
    col = X(:, j);
    col_min = min(col);
    col_max = max(col);
    if col_max > col_min
        X_normalized(:, j) = (col - col_min) / (col_max - col_min);
    end
end
norms = sqrt(sum(X_normalized.^2, 2));
norms(norms == 0) = 1;
X_normalized = X_normalized ./ norms;
X_normalized(isnan(X_normalized)) = 0;
end