function knn = knn(X_norm, k_nn)

% X1 = X_norm(:,1:lvl_loc-1);
% X2 = X_norm(:,lvl_loc+lvl_n:end);
% X_norm = [X1 X2];

KNN = zeros(size(X_norm,1), k_nn);
for i=1:size(X_norm,1)
    knn_distsance = sqrt(sum((repmat(X_norm(i,:),size(X_norm,1),1)-X_norm).^2,2));
    [val, sort_index] = sort(knn_distsance,'ascend');
    KNN(i,:) = sort_index(1:k_nn); 
end
knn = KNN;
end