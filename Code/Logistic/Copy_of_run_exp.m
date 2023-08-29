%%Assumptions
% 1- We assume that we have a single protected attribute with binary levels
% 2- We don't use the protected feature as one of the features in the
% training process


function res = Copy_of_run_exp(data_train, data_train_enc, data_test, data_test_enc, ...
    data_val, data_val_enc, B_name, positive_class, deprived_group, lvl_loc, lvl_n)

% Preprocess the data
preprocess = preProcess(data_train_enc, data_train, B_name, lvl_loc, lvl_n);
X = preprocess.X; % The one-hot encoded features with intercept (protected feature excluded)
y = preprocess.y; % class column
p = preprocess.p; % The column of protected feature

preprocess_test = preProcess(data_test_enc, data_test, B_name, lvl_loc, lvl_n);
X_test = preprocess_test.X;
y_test = preprocess_test.y;
p_test = preprocess_test.p;

preprocess_val = preProcess(data_val_enc, data_val, B_name, lvl_loc, lvl_n);
X_val = preprocess_val.X;
y_val = preprocess_val.y;
p_val = preprocess_val.p;


global ind_fair; global group_fair; global lambda; global p_lvl; global M;
p_lvl = zeros(2,1); % size of each lvl of the protected feature
M = 1; %n1*n2
for i = 1:length(unique(p))
    protected_levels = unique(p);
    p_lvl(i,1) = sum(p==protected_levels(i));
    M = M * p_lvl(i,1);% n1*n2 
end

% ============ Compute Cost and Gradient ============
%  In this part, we will implement the cost and gradient
%  for logistic regression

[n,m] = size(X);
% Initialize fitting parameters
initial_theta = zeros(m, 1);

% Compute and display initial cost and gradient
[cost] = costFunction(initial_theta, X, y, p);


% ============= Optimizing using fminunc  =============
%  In this part, we will use a built-in function (fminunc) to find the
%  optimal parameters theta.

%  Set options for fminunc
tic
% options = optimset('GradObj', 'on', 'Algorithm','trust-region', 'MaxIter', 400);
options = optimset('MaxIter', 800);


%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = fminunc(@(t)(costFunction(t, X, y, p)), initial_theta, options);

fprintf('lambda = %d ,indi = %d ,group = %d\n',lambda , ind_fair, group_fair );
toc


%============== Predict and Accuracies ==============

% Compute accuracy on our training set
tr_pred = predict(theta, X);
res.tr_acc = mean(double(tr_pred == y)) * 100;
fprintf('Train Accuracy: %f\n', res.tr_acc);


% get_sp(X_tes(:,B_loc),pred_labels_tes,deprived_group,positive_class);
res.tr_sp = get_sp(p,tr_pred,deprived_group,positive_class);
fprintf('training statistical parity %f\n', res.tr_sp);

% Compute accuracy on our test set
te_pred = predict(theta, X_test);
res.te_acc = mean(double(te_pred == y_test)) * 100;
fprintf('Test Accuracy: %f\n', res.te_acc);

res.te_sp = get_sp(p_test,te_pred,deprived_group,positive_class);
fprintf('test statistical parity %f\n', res.te_sp);

% Compute accuracy on our val set
tr_pred = predict(theta, X_val);
res.tr_acc = mean(double(tr_pred == y_val)) * 100;
fprintf('Val Accuracy: %f\n', res.tr_acc);

res.tr_sp = get_sp(p_val,tr_pred,deprived_group,positive_class);
fprintf('val statistical parity %f\n', res.tr_sp);