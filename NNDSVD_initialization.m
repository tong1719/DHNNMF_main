function [U, V] = NNDSVD_initialization(X, r)
[m, n] = size(X);

try
    [U0, S0, V0] = svds(X, r);
catch
    U = abs(randn(m, r) * 0.1 + 0.5);
    V = abs(randn(n, r) * 0.1 + 0.5);
    return;
end

U = zeros(m, r);
V = zeros(n, r);

for i = 1:r
    u = U0(:, i);
    v = V0(:, i);
    s = S0(i, i);
    
    u_pos = max(u, 0);
    u_neg = max(-u, 0);
    v_pos = max(v, 0);
    v_neg = max(-v, 0);
    
    norm_pos = norm(u_pos) * norm(v_pos);
    norm_neg = norm(u_neg) * norm(v_neg);
    
    if norm_pos >= norm_neg
        U(:, i) = sqrt(s * norm_pos) * u_pos / max(norm(u_pos), 1e-10);
        V(:, i) = sqrt(s * norm_pos) * v_pos / max(norm(v_pos), 1e-10);
    else
        U(:, i) = sqrt(s * norm_neg) * u_neg / max(norm(u_neg), 1e-10);
        V(:, i) = sqrt(s * norm_neg) * v_neg / max(norm(v_neg), 1e-10);
    end
end

U = max(U, 1e-10);
V = max(V, 1e-10);
end