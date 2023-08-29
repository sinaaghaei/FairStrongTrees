%%Assumptions
% 1- We assume that we have a single protected attribute with binary levels
% 2- We don't use the protected feature as one of the features in the
% training process

%%
%Load data
clearvars
data_path = "../../DataSets/KamiranVersion/";
% splits = ["1", "2", "3", "4", "5"];
splits = ["1"];
data_group = "compas";

train_set = append(data_path, data_group, "_train_calibration_", splits, ".csv");
train_set_enc = append(data_path, data_group, "_train_calibration_enc_", splits, ".csv");
test_set = append(data_path, data_group, "_test_", splits, ".csv");
test_set_enc = append(data_path, data_group, "_test_enc_", splits, ".csv");
val_set = append(data_path, data_group, "_calibration_", splits, ".csv");
val_set_enc = append(data_path, data_group, "_calibration_enc_", splits, ".csv");


%data specific parameters
positive_class = 1;
deprived_group = 1; 
lvl_loc = 1; %location of encoded protected feature
lvl_n = 1; %Number of encoded column for the protected feature. For two levels we only need one binary column
B_name = 'race';

%General Parameters
global ind_fair; global group_fair; global lambda; global p_lvl; global M;

group_fair = 1;
ind_fair =  0;
% lambdas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
lambdas = [0.1];
for s = 1:length(splits)
    data_train = readtable(train_set(s));
    data_train_enc = readtable(train_set_enc(s));
    data_test = readtable(test_set(s));
    data_test_enc = readtable(test_set_enc(s));
    data_val = readtable(val_set(s));
    data_val_enc = readtable(val_set_enc(s));
    for i = 1:length(lambdas)
        lambda = lambdas(i);
        res = Copy_of_run_exp(data_train, data_train_enc, data_test, data_test_enc, ...
        data_val, data_val_enc, B_name, positive_class, deprived_group, lvl_loc, lvl_n);
%         outf = append(out_dir, data_group, "-", num2str(splits(i)), "-", num2str(epsilons(eps)), "-", num2str(depths(d)), ".csv");
%         csvwrite(outf, [i, epsilons(eps), depths(d), res.acc_tr_pre, res.disc_tr_pre, res.acc_te_pre, res.disc_te_pre, ...
%            res.acc_val_pre, res.disc_val_pre, res.liacc_tr_post, res.disc_tr_post, res.acc_te_post, res.disc_te_post, ...
%           res.acc_val_post, res.disc_val_post])
    end
end


%% ============ Compute Cost and Gradient ============
%  In this part, we will implement the cost and gradient
%  for logistic regression

[n,m] = size(X);
% Initialize fitting parameters
initial_theta = zeros(m, 1);

% Compute and display initial cost and gradient
[cost] = costFunction(initial_theta, X, y, p);


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