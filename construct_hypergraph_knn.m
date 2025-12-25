function [L_hyper, D_hyper, S_hyper] = construct_hypergraph_knn(Y, k, mu_weight)
n = size(Y, 1);

k = min(k, n-1);
k = max(k, 1);

try
    D = pdist2(Y, Y);
    
    H = zeros(n, n);
    for i = 1:n
        [~, idx] = sort(D(i, :));
        neighbors = idx(1:min(k+1, n));
        H(i, neighbors) = 1;
    end
    
    sigma = median(D(D > 0)) / 2;
    sigma = max(sigma, 1e-10);
    
    W = zeros(n, n);
    for i = 1:n
        for j = 1:n
            if H(i, j) > 0 && i ~= j
                W(i, j) = exp(-D(i, j)^2 / (2 * sigma^2));
            end
        end
    end
    
    W = (W + W') / 2;
    
    D_v = diag(sum(W, 2));
    D_e = diag(sum(H, 1)');
    
    D_e_inv = diag(1 ./ max(diag(D_e), 1e-10));
    A = H * (W * D_e_inv * H');
    A = (A + A') / 2;
    
    if mu_weight > 0
        A = A * mu_weight;
        D_v = D_v * mu_weight;
    end
    
    L_hyper = D_v - A;
    
    D_hyper = D_v;
    S_hyper = A;
    
catch
    L_hyper = eye(n);
    D_hyper = eye(n);
    S_hyper = zeros(n);
end
end