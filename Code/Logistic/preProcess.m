function prep = preProcess(data_encode, data, p_name, lvl_loc, lvl_n)

M = table2array(data_encode);
% We want to predict the last column...
y = M(:,end);
% ...based on the others
X = M(:,1:end-1);


p = grp2idx(data.(p_name));
%remove the protected feature from learning
X1 = X(:,1:lvl_loc-1);
X2 = X(:,lvl_loc+lvl_n:end);
X = [X1 X2];



[n, m] = size(X);
% Add intercept term to X
X = [ones(n, 1) X];


prep.X = X;
prep.y =y;
prep.p = p;

