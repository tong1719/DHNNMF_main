function V_processed = apply_postprocessing(V)
row_sums = sum(V, 2);
row_sums(row_sums == 0) = 1;
V_norm = V ./ row_sums;

threshold = 0.1 * max(V_norm(:));
V_sparse = V_norm;
V_sparse(V_sparse < threshold) = 0;

[n, r] = size(V_sparse);
for i = 1:r
    col = V_sparse(:, i);
    if n > 10
        col_smooth = smooth(col, 5);
        V_sparse(:, i) = col_smooth;
    end
end

row_sums = sum(V_sparse, 2);
row_sums(row_sums == 0) = 1;
V_processed = V_sparse ./ row_sums;

perturbation = 1e-6 * randn(size(V_processed));
V_processed = V_processed + perturbation;
V_processed = max(V_processed, 0);

row_sums = sum(V_processed, 2);
row_sums(row_sums == 0) = 1;
V_processed = V_processed ./ row_sums;
end