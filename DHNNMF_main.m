function [U, V, obj_history] = DHNNMF_main(X, r, options)

if ~isfield(options, 'mu1'), options.mu1 = 1.0; end
if ~isfield(options, 'mu2'), options.mu2 = 1.0; end
if ~isfield(options, 'nu'), options.nu = 0.01; end
if ~isfield(options, 'theta'), options.theta = 0.3; end
if ~isfield(options, 'k'), options.k = 5; end
if ~isfield(options, 'maxIter'), options.maxIter = 500; end
if ~isfield(options, 'tol'), options.tol = 1e-6; end
if ~isfield(options, 'verbose'), options.verbose = true; end
if ~isfield(options, 'init_method'), options.init_method = 'nndsvd'; end
if ~isfield(options, 'normalize_data'), options.normalize_data = true; end

mu1 = options.mu1;
mu2 = options.mu2;
nu = options.nu;
theta = options.theta;
k = options.k;
maxIter = options.maxIter;
tol = options.tol;
verbose = options.verbose;
init_method = options.init_method;
normalize_data = options.normalize_data;

[m, n] = size(X);

if normalize_data
    if verbose, fprintf('Normalizing input data...\n'); end
    
    X_min = min(X(:));
    if X_min < 0
        X = X - X_min;
        if verbose
            fprintf('  Added %.4f to make data nonnegative\n', -X_min);
        end
    end
    
    X_mean = mean(X, 2);
    X_std = std(X, 0, 2);
    X_std(X_std == 0) = 1;
    X = (X - X_mean) ./ X_std;
    
    X_min_after = min(X(:));
    if X_min_after < 0
        X = X - X_min_after;
    end
    
    X_max = max(X(:));
    if X_max > 0
        X = X / X_max;
    end
end

if verbose, fprintf('Initializing U and V...\n'); end

switch lower(init_method)
    case 'svd'
        try
            [U_svd, S_svd, V_svd] = svds(X, r);
            U = abs(U_svd);
            V = abs(V_svd');
            if size(V, 1) ~= n
                V = V(1:n, :);
            end
            V = V * diag(diag(S_svd));
        catch
            if verbose
                fprintf('SVD initialization failed, using random initialization\n');
            end
            U = abs(randn(m, r) * 0.1 + 0.5);
            V = abs(randn(n, r) * 0.1 + 0.5);
        end
    case 'nndsvd'
        [U, V] = NNDSVD_initialization(X, r);
    otherwise
        U = abs(randn(m, r) * 0.1 + 0.5);
        V = abs(randn(n, r) * 0.1 + 0.5);
end

U = max(U, 1e-10);
V = max(V, 1e-10);

S = (1 - theta) * eye(r) + (theta / r) * ones(r, r);

obj_history = zeros(maxIter, 1);

if verbose, fprintf('Building dual hypergraphs...\n'); end

k = min(k, n-1);
k = max(k, 1);

try
    [L_hyper_V, D_hyper_V, S_hyper_V] = construct_hypergraph_knn(X', k, mu1);
    [L_hyper_U, D_hyper_U, S_hyper_U] = construct_hypergraph_knn(X, k, mu2);
catch
    if verbose
        fprintf('Hypergraph construction failed, using identity matrices\n');
    end
    L_hyper_V = eye(n);
    D_hyper_V = eye(n);
    S_hyper_V = zeros(n);
    
    L_hyper_U = eye(m);
    D_hyper_U = eye(m);
    S_hyper_U = zeros(m);
end

if verbose, fprintf('Starting optimization...\n'); end

prev_obj = compute_objective(X, U, S, V, L_hyper_U, L_hyper_V, mu1, mu2, nu);
if prev_obj < 0
    prev_obj = abs(prev_obj);
end

converged = false;

for iter = 1:maxIter
    try
        X_V = X * V;
        X_V_S = X_V * S';
        
        numerator_U = X_V_S + mu2 * S_hyper_U * U + 2 * nu * U;
        
        VtV = V' * V;
        S_VtV_S = S * VtV * S';
        U_S_VtV_S = U * S_VtV_S;
        
        denominator_U = U_S_VtV_S + mu2 * D_hyper_U * U + 2 * nu * (U * U') * U;
        
        denominator_U = max(denominator_U, 1e-10);
        
        U = U .* (numerator_U ./ denominator_U);
        U = max(U, 1e-10);
    catch ME
        if verbose && iter == 1
            fprintf('Error updating U: %s\n', ME.message);
        end
        break;
    end
    
    try
        Xt_U = X' * U;
        Xt_U_S = Xt_U * S;
        
        numerator_V = Xt_U_S + mu1 * S_hyper_V * V + 2 * nu * V;
        
        UtU = U' * U;
        St_UtU_S = S' * UtU * S;
        V_St_UtU_S = V * St_UtU_S;
        
        denominator_V = V_St_UtU_S + mu1 * D_hyper_V * V + 2 * nu * (V * V') * V;
        
        denominator_V = max(denominator_V, 1e-10);
        
        V = V .* (numerator_V ./ denominator_V);
        V = max(V, 1e-10);
    catch ME
        if verbose && iter == 1
            fprintf('Error updating V: %s\n', ME.message);
        end
        break;
    end
    
    if mod(iter, 20) == 0
        col_norms_U = sqrt(sum(U.^2, 1));
        col_norms_U(col_norms_U == 0) = 1;
        U = U ./ col_norms_U;
        V = V .* col_norms_U;
    end
    
    obj = compute_objective(X, U, S, V, L_hyper_U, L_hyper_V, mu1, mu2, nu);
    obj = max(obj, 0);
    obj_history(iter) = obj;
    
    if iter > 1
        rel_change = abs(obj - prev_obj) / max(abs(prev_obj), 1);
        
        if rel_change < tol
            if verbose
                fprintf('Iter %d: Obj=%.6f, RelChange=%.2e (converged)\n', ...
                    iter, obj, rel_change);
            end
            converged = true;
            break;
        end
    end
    
    prev_obj = obj;
    
    if verbose && (mod(iter, 50) == 0 || iter == 1 || iter == maxIter)
        if iter > 1
            rel_change_display = abs(obj - obj_history(iter-1)) / max(abs(obj_history(iter-1)), 1);
        else
            rel_change_display = 0;
        end
        fprintf('Iter %4d: Obj=%.6f, RelChange=%.2e\n', ...
            iter, obj, rel_change_display);
    end
end

if verbose && ~converged
    fprintf('Reached max iterations: %d\n', maxIter);
end

if iter < maxIter
    obj_history = obj_history(1:iter);
end

col_norms = sqrt(sum(U.^2, 1));
col_norms(col_norms == 0) = 1;
U = U ./ col_norms;
V = V .* col_norms;

row_norms = sqrt(sum(V.^2, 2));
row_norms(row_norms == 0) = 1;
V = V ./ row_norms;

V = apply_postprocessing(V);

if verbose
    fprintf('Optimization completed.\n');
    fprintf('  Final objective: %.6f\n', obj);
    fprintf('  U range: [%.2e, %.2e]\n', min(U(:)), max(U(:)));
    fprintf('  V range: [%.2e, %.2e]\n', min(V(:)), max(V(:)));
end

end