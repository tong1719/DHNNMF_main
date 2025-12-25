function D = EuDist2(fea_a, fea_b, bSqrt)
if nargin < 2
    fea_b = fea_a;
end
if nargin < 3
    bSqrt = 1;
end
D = pdist2(fea_a, fea_b);
if ~bSqrt
    D = D.^2;
end
end