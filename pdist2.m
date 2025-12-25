function D = pdist2(X, Y)
if nargin < 2
    Y = X;
end

m = size(X, 1);
n = size(Y, 1);
D = zeros(m, n);

for i = 1:m
    for j = 1:n
        D(i, j) = norm(X(i, :) - Y(j, :));
    end
end
end