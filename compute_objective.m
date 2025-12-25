function obj = compute_objective(X, U, S, V, L_hyper_U, L_hyper_V, mu1, mu2, nu)
recon_error = norm(X - U * S * V', 'fro')^2;

graph_term_U = 0;
graph_term_V = 0;

if mu1 > 0
    graph_term_V = mu1 * trace(V' * L_hyper_V * V);
    graph_term_V = max(graph_term_V, 0);
end

if mu2 > 0
    graph_term_U = mu2 * trace(U' * L_hyper_U * U);
    graph_term_U = max(graph_term_U, 0);
end

orth_term_U = 0;
orth_term_V = 0;

if nu > 0
    r = size(U, 2);
    I = eye(r);
    orth_term_U = nu * norm(I - U' * U, 'fro')^2;
    orth_term_V = nu * norm(I - V' * V, 'fro')^2;
    
    orth_term_U = max(orth_term_U, 0);
    orth_term_V = max(orth_term_V, 0);
end

obj = recon_error + graph_term_U + graph_term_V + orth_term_U + orth_term_V;
obj = max(obj, 0);
end