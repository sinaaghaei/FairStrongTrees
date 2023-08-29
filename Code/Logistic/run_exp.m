%%Assumptions
% 1- We assume that we have a single protected attribute with binary levels
% 2- We don't use the protected feature as one of the features in the
% training process

%%
%Load data
clearvars
% data_group = 'compas';
data_path = "../../DataSets/KamiranVersion/";
splits = ["1", "2", "3", "4", "5"];
data_group = "compas";

train_set = append(data_path, data_group, "_train_calibration_", splits, ".csv");
train_set_enc = append(data_path, data_group, "_train_calibration_enc_", splits, ".csv");
test_set = append(data_path, data_group, "_test_", splits, ".csv");
test_set_enc = append(data_path, data_group, "_test_enc_", splits, ".csv");
val_set = append(data_path, data_group, "_calibration_", splits, ".csv");
val_set_enc = append(data_path, data_group, "_calibration_enc_", splits, ".csv");

data_train = readtable('../../DataSets/KamiranVersion/compas_train_2.csv');
data_train_enc = readtable('../../DataSets/KamiranVersion/compas_train_enc_2.csv');
% data_train = readtable('../../DataSets/KamiranVersion/compas_train_2.csv');
% data_train_enc = readtable('../../DataSets/KamiranVersion/compas_train_enc_2.csv');
% data_train = readtable('../../DataSets/KamiranVersion/compas_train_2.csv');
% data_train_enc = readtable('../../DataSets/KamiranVersion/compas_train_enc_2.csv');
% fprintf('%d', height(data_train_enc));
% hi = str2double(data_train_enc.target);
%data specific parameters
positive_class = 1;
deprived_group = 1; 
lvl_loc = 1; %location of encoded protected feature
lvl_n =1; %Number of encoded column for the protected feature. For two levels we only need one binary column

%%
% Preprocess the data
preprocess = preProcess(data_train_enc, data_train, 'race', lvl_loc, lvl_n);
X = preprocess.X; % The one-hot encoded features with intercept (protected feature excluded)
y = preprocess.y; % class column
p = preprocess.p; % The column of protected feature

%%
%General Parameters
global ind_fair; global group_fair;  global lambda; global p_lvl; global M;

group_fair = 1;
ind_fair =  0;
lambda = .1;

%% 
p_lvl = zeros(2,1); % size of each lvl of the protected feature
M = 1; %n1*n2
for i = 1:length(unique(p))
    protected_levels = unique(p);
    p_lvl(i,1) = sum(p==protected_levels(i));
    M = M * p_lvl(i,1);% n1*n2 
end


%% ============ Compute Cost and Gradient ============
%  In this part, we will implement the cost and gradient
%  for logistic regression

[n,m] = size(X);
% Initialize fitting parameters
initial_theta = zeros(m, 1);

% Compute and display initial cost and gradient
[cost] = costFunction(initial_theta, X, y, p);

% fprintf('Cost at initial theta (zeros): %f\n', cost);
% fprintf('Gradient at initial theta (zeros): \n');
% fprintf(' %f \n', grad);


%% ============= Optimizing using fminunc  =============
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

% Print theta to screen
%fprintf('Cost at theta found by fminunc: %f\n', cost);
%fprintf('theta: \n');
%fprintf(' %f \n', theta);


%% ============== Predict and Accuracies ==============

% Compute accuracy on our training set
tr_pred = predict(theta, X);
tr_acc = mean(double(tr_pred == y)) * 100;
fprintf('Train Accuracy: %f\n', tr_acc);

tr_sp = get_sp(p,tr_pred,deprived_group,positive_class);
fprintf('training statistical parity %f\n',tr_sp);

% Compute accuracy on our test set
te_pred = predict(theta, X);
te_acc = mean(double(te_pred == y)) * 100;
fprintf('Train Accuracy: %f\n', te_acc);

te_sp = get_sp(p,te_pred,deprived_group,positive_class);
fprintf('training statistical parity %f\n',te_sp);

% Compute accuracy on our val set
tr_pred = predict(theta, X);
tr_acc = mean(double(tr_pred == y)) * 100;
fprintf('Train Accuracy: %f\n', tr_acc);

tr_sp = get_sp(p,tr_pred,deprived_group,positive_class);
fprintf('training statistical parity %f\n',tr_sp);