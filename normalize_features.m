function fea = normalize_features(fea)
norms = sqrt(sum(fea.^2, 2));
norms(norms == 0) = 1;
fea = fea ./ norms;
fea(isnan(fea)) = 0;
end