function prep = preProcess(data_encode, data, B_name, lvl_loc, lvl_n)
%data_encode, data, numner of data points, starting row of sample

M = table2array(data_encode);
% We want to predict the first column...
Y = M(:,end);
% ...based on the others
X = M(:,1:end-1);

X = [X, grp2idx(data.(B_name))]; %add race to the encoded data
%remove B from learning
B_x = X(:,lvl_loc:lvl_loc+lvl_n-1);
X1 = X(:,1:lvl_loc-1);
X2 = X(:,lvl_loc+lvl_n:end);
X = [X1 X2];
B_loc = size(X,2); %protected feature (not encoded) location in X

%
%remove B from column names
cols = data_encode.Properties.VariableNames;
cols = cols(:,1:end-1);
cols1 = cols(:,1:lvl_loc-1);
cols2 = cols(:,lvl_loc+lvl_n:end);
cols = [cols1 cols2];
cols = [cols, B_name];

prep.X = X;
prep.Y = Y-1;%We assume that the target column is 1/2 and not 0/1
prep.B = B_loc;
prep.cols = cols;


