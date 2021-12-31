% things to vary include: dataset_group, training/test split (1-5
clearvars
data_path = "../../DataSets/KamiranVersion/";
splits = ["1", "2", "3", "4", "5"];
data_group = "compas";

%data specific parameters
B_name = "race";
positive_class = 2;
deprived_group = 1; % non-white is the deprived group
lvl_loc = 1; lvl_n = 1; %location of encoded protected feature

train_set = append(data_path, data_group, "_train_calibration_", splits, ".csv");
train_set_enc = append(data_path, data_group, "_train_calibration_enc_", splits, ".csv");
test_set = append(data_path, data_group, "_test_", splits, ".csv");
test_set_enc = append(data_path, data_group, "_test_enc_", splits, ".csv");

epsilons = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0,75, 0.8, 0.85, 0.9];
depths = [1, 2, 3]; % max depth is depth + 1
fair = 1;

data_train = readtable(train_set(1));
data_train_enc = readtable(train_set_enc(1));
data_test = readtable(test_set(1));
data_test_enc = readtable(test_set_enc(1));
eps = 0.05;
Dep_lim = 2;

res = Copy_of_run_exp(data_train, data_train_enc, data_test, data_test_enc, ...
     B_name, positive_class, deprived_group, lvl_loc, lvl_n, eps, fair, Dep_lim);
% out_dir = "../../Results/Kamiran/";
% outf = append(out_dir, data_group, "-", "1", "-", num2str(eps), "-", num2str(Dep_lim), "-", num2str(fair), ".csv")
% csvwrite(outf, [res.acc_tr_pre, res.disc_tr_pre, res.acc_te_pre, res.disc_te_pre, ...
%     res.acc_tr_post, res.disc_tr_post, res.acc_te_post, res.disc_te_post]);
% fprintf(res.acc_tr_pre)
% fprintf(res.disc_tr_pre)
% fprintf(res.acc_te_pre)
% fprintf(res.disc_te_pre)
% fprintf(res.acc_tr_post)
% fprintf(res.disc_tr_post)
% fprintf(res.acc_te_post)
% fprintf(res.disc_te_post)
%%
% things to vary include: dataset_group, training/test split (1-5
clearvars
data_path = "../../DataSets/Kamiran Version/";
splits = ["1", "2", "3", "4", "5"];
data_group = "german";

%data specific parameters
B_name = "age";
positive_class = 1;
deprived_group = 1; % non-white is the deprived group
lvl_loc = 55; lvl_n = 1; %location of encoded protected feature

train_set = append(data_path, data_group, "_train_calibration_", splits, ".csv");
train_set_enc = append(data_path, data_group, "_train_calibration_enc_", splits, ".csv");
test_set = append(data_path, data_group, "_test_", splits, ".csv");
test_set_enc = append(data_path, data_group, "_test_enc_", splits, ".csv");

epsilons = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0,75, 0.8, 0.85, 0.9];
depths = [1, 2, 3]; % max depth is depth + 1
fair = 1;

data_train = readtable(train_set(1));
data_train_enc = readtable(train_set_enc(1));
data_test = readtable(test_set(1));
data_test_enc = readtable(test_set_enc(1));
eps = 0.05;
Dep_lim = 2;

res = Copy_of_run_exp(data_train, data_train_enc, data_test, data_test_enc, ...
     B_name, positive_class, deprived_group, lvl_loc, lvl_n, eps, fair, Dep_lim);

%%
data_path = "../../DataSets/KamiranVersion/";
splits = ["1", "2", "3", "4", "5"];
data_group = "compas";

%data specific parameters
B_name = 'race';
positive_class = 0;
deprived_group = 1; % non-white is the deprived group
lvl_loc = 1; lvl_n = 1; %location of encoded protected feature

train_set = append(data_path, data_group, "_train_", splits, ".csv");
train_set_enc = append(data_path, data_group, "_train_enc_", splits, ".csv");
test_set = append(data_path, data_group, "_test_", splits, ".csv");
test_set_enc = append(data_path, data_group, "_test_enc_", splits, ".csv");

epsilons = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, ...
    0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, ...
    0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, ...
    0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, ...
    0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, ...
    0.5, 0.51, 0.52, 0.53, 0.54, 0.55];
depths = [1, 2, 3]; % max depth is depth + 1
fair = 1;

out_dir = "../../Results/Kamiran/raw/";

for i = 1:length(splits)
    for eps = 1:length(epsilons)
        for d = 1:length(depths)
            data_train = readtable(train_set(i));
            data_train_enc = readtable(train_set_enc(i));
            data_test = readtable(test_set(i));
            data_test_enc = readtable(test_set_enc(i));
            res = Copy_of_run_exp(data_train, data_train_enc, data_test, data_test_enc, ...
         B_name, positive_class, deprived_group, lvl_loc, lvl_n, epsilons(eps), fair, depths(d));
            outf = append(out_dir, data_group, "-", num2str(i), "-", num2str(epsilons(eps)), "-", num2str(depths(d)), ".csv")
            csvwrite(outf, [i, epsilons(eps), depths(d), res.acc_tr_pre, res.disc_tr_pre, res.acc_te_pre, res.disc_te_pre, ...
                res.acc_tr_post, res.disc_tr_post, res.acc_te_post, res.disc_te_post])
        end
    end
end

%%
data_path = "../../DataSets/KamiranVersion/";
splits = ["1", "2", "3", "4", "5"];
data_group = "german";

%data specific parameters
B_name = 'age';
positive_class = 1;
deprived_group = 1; % non-white is the deprived group
lvl_loc = 55; lvl_n = 1; %location of encoded protected feature

train_set = append(data_path, data_group, "_train_", splits, ".csv");
train_set_enc = append(data_path, data_group, "_train_enc_", splits, ".csv");
test_set = append(data_path, data_group, "_test_", splits, ".csv");
test_set_enc = append(data_path, data_group, "_test_enc_", splits, ".csv");

epsilons = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, ...
    0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, ...
    0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, ...
    0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, ...
    0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, ...
    0.5, 0.51, 0.52, 0.53, 0.54, 0.55];
depths = [1, 2, 3]; % max depth is depth + 1
fair = 1;

out_dir = "../../Results/Kamiran/raw/";

for i = 1:length(splits)
    for eps = 1:length(epsilons)
        for d = 1:length(depths)
            data_train = readtable(train_set(i));
            data_train_enc = readtable(train_set_enc(i));
            data_test = readtable(test_set(i));
            data_test_enc = readtable(test_set_enc(i));
            res = Copy_of_run_exp(data_train, data_train_enc, data_test, data_test_enc, ...
         B_name, positive_class, deprived_group, lvl_loc, lvl_n, epsilons(eps), fair, depths(d));
            outf = append(out_dir, data_group, "-", num2str(i), "-", num2str(epsilons(eps)), "-", num2str(depths(d)), ".csv")
            csvwrite(outf, [i, epsilons(eps), depths(d), res.acc_tr_pre, res.disc_tr_pre, res.acc_te_pre, res.disc_te_pre, ...
                res.acc_tr_post, res.disc_tr_post, res.acc_te_post, res.disc_te_post])
        end
    end
end


%%
data_path = "../../DataSets/KamiranVersion/";
splits = ["1", "2", "3", "4", "5"];
% splits = ["5"]
data_group = "default";

%data specific parameters
B_name = 'SEX';
positive_class = 0; % i.e. >50, which is target=2 in raw file
deprived_group = 2; % female is the deprived group
lvl_loc = 5; lvl_n = 1; %location of encoded protected feature

train_set = append(data_path, data_group, "_train_", splits, ".csv");
train_set_enc = append(data_path, data_group, "_train_enc_", splits, ".csv");
test_set = append(data_path, data_group, "_test_", splits, ".csv");
test_set_enc = append(data_path, data_group, "_test_enc_", splits, ".csv");

epsilons = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, ...
    0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, ...
    0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, ...
    0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, ...
    0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, ...
    0.5, 0.51, 0.52, 0.53, 0.54, 0.55];
depths = [1, 2, 3]; % max depth is depth + 1
fair = 1;

out_dir = "../../Results/Kamiran/raw/";

for i = 1:length(splits)
    for eps = 1:length(epsilons)
        for d = 1:length(depths)
            data_train = readtable(train_set(i));
            data_train_enc = readtable(train_set_enc(i));
            data_test = readtable(test_set(i));
            data_test_enc = readtable(test_set_enc(i));
            res = Copy_of_run_exp(data_train, data_train_enc, data_test, data_test_enc, ...
         B_name, positive_class, deprived_group, lvl_loc, lvl_n, epsilons(eps), fair, depths(d));
            outf = append(out_dir, data_group, "-", num2str(i), "-", num2str(epsilons(eps)), "-", num2str(depths(d)), ".csv")
            csvwrite(outf, [i, epsilons(eps), depths(d), res.acc_tr_pre, res.disc_tr_pre, res.acc_te_pre, res.disc_te_pre, ...
                res.acc_tr_post, res.disc_tr_post, res.acc_te_post, res.disc_te_post])
        end
    end
end


%%
data_path = "../../DataSets/KamiranVersion/";
splits = ["1", "2", "3", "4", "5"];
% splits = ["5"]
data_group = "limited-adult";

%data specific parameters
B_name = 'sex';
positive_class = 1; % i.e. >50, which is target=2 in raw file
deprived_group = 1; % female is the deprived group
lvl_loc = 41; lvl_n = 1; %location of encoded protected feature

train_set = append(data_path, data_group, "_train_", splits, ".csv");
train_set_enc = append(data_path, data_group, "_train_enc_", splits, ".csv");
test_set = append(data_path, data_group, "_test_", splits, ".csv");
test_set_enc = append(data_path, data_group, "_test_enc_", splits, ".csv");

epsilons = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, ...
    0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, ...
    0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, ...
    0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, ...
    0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, ...
    0.5, 0.51, 0.52, 0.53, 0.54, 0.55];
depths = [1, 2, 3]; % max depth is depth + 1
fair = 1;

out_dir = "../../Results/Kamiran/raw/";

for i = 1:length(splits)
    for eps = 1:length(epsilons)
        for d = 1:length(depths)
            data_train = readtable(train_set(i));
            data_train_enc = readtable(train_set_enc(i));
            data_test = readtable(test_set(i));
            data_test_enc = readtable(test_set_enc(i));
            res = Copy_of_run_exp(data_train, data_train_enc, data_test, data_test_enc, ...
         B_name, positive_class, deprived_group, lvl_loc, lvl_n, epsilons(eps), fair, depths(d));
            outf = append(out_dir, data_group, "-", num2str(i), "-", num2str(epsilons(eps)), "-", num2str(depths(d)), ".csv")
            csvwrite(outf, [i, epsilons(eps), depths(d), res.acc_tr_pre, res.disc_tr_pre, res.acc_te_pre, res.disc_te_pre, ...
                res.acc_tr_post, res.disc_tr_post, res.acc_te_post, res.disc_te_post])
        end
    end
end



