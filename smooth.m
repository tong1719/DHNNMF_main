function y = smooth(x, window_size)
n = length(x);
y = zeros(size(x));
half_window = floor(window_size / 2);

for i = 1:n
    start_idx = max(1, i - half_window);
    end_idx = min(n, i + half_window);
    y(i) = mean(x(start_idx:end_idx));
end
end