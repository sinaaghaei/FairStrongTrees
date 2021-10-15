%% Load the auto data
% clearvars -except defaencod defa
% clearvars -except censusincometes censusincometesencod censusincometr censusincometrencod
clearvars -except com comencod
%%
%fair coef (on/off)
fair =.1; %fair coef
Dep_lim = 2; % max depth is Dep_lim + 1


%% Compas tr
B_name = 'race';
lvl_loc = 4; lvl_n =6; %location of encoded protected feature
% preprocess = preprocc(comencod, com, 100, 1, B_name, lvl_loc, lvl_n); %try
% preprocess = preprocc(comencod, com, 1000, 1, B_name, lvl_loc, lvl_n); %s1 (data_encode, data, nn, start, B_name, lvl_loc, lvl_n)
% preprocess = preprocc(comencod, com, 2000, 4103, B_name, lvl_loc, lvl_n); %s2
% preprocess = preprocc(comencod, com, 1000, 6104, B_name, lvl_loc, lvl_n); %s3
% preprocess = preprocc(comencod, com, 3000, 7001, B_name, lvl_loc, lvl_n); %s4
preprocess = preprocc(comencod, com, 5000, 1, B_name, lvl_loc, lvl_n); %s5

%% Census tr
% B_name = 'race';
% lvl_loc = 54; lvl_n =5; %location of encoded protected feature
% preprocess = preprocc(censusincometrencod, censusincometr, 100, 1, B_name, lvl_loc, lvl_n); %try
% preprocess = preprocc(censusincometrencod, censusincometr, 1000, 2901, B_name, lvl_loc, lvl_n); %s1 (data_encode, data, nn, start, B_name, lvl_loc, lvl_n)
% preprocess = preprocc(censusincometrencod, censusincometr, 2000, 16001, B_name, lvl_loc, lvl_n); %s2
% preprocess = preprocc(censusincometrencod, censusincometr, 2000, 3001, B_name, lvl_loc, lvl_n); %s3
% preprocess = preprocc(censusincometrencod, censusincometr, 4000, 8001, B_name, lvl_loc, lvl_n); %s4
% preprocess = preprocc(censusincometrencod, censusincometr, 5000, 12001, B_name, lvl_loc, lvl_n); %s5

%% Default tr
% protected feature is B
% B_name = 'SEX';
% lvl_loc = 2; lvl_n =2; %location of encoded protected feature
% preprocess = preprocc(defaencod, defa, 100, 1, B_name, lvl_loc, lvl_n); %try
% preprocess = preprocc(defaencod, defa, 1000, 1, B_name, lvl_loc, lvl_n); %s1 (data_encode, data, nn, start, B_name, lvl_loc, lvl_n)
% preprocess = preprocc(defaencod, defa, 1000, 2001, B_name, lvl_loc, lvl_n); %s2
% preprocess = preprocc(defaencod, defa, 2000, 4001, B_name, lvl_loc, lvl_n); %s3
% preprocess = preprocc(defaencod, defa, 4000, 8001, B_name, lvl_loc, lvl_n); %s4
% preprocess = preprocc(defaencod, defa, 6000, 17001, B_name, lvl_loc, lvl_n); %s5
% preprocess = preprocc(defaencod, defa, 20000, 1, B_name, lvl_loc, lvl_n); %all

%% Compas tes
% prediction on test set before Relab
% protected feature
B_name_tes = 'race';
lvl_loc = 4; lvl_n =6; %location of encoded protected feature
% preprocess_tes = preprocc(comencod, com, 100, 2001, B_name_tes, lvl_loc, lvl_n); %try
% preprocess_tes = preprocc(comencod, com, 1000, 1101, B_name_tes, lvl_loc, lvl_n); %s1
% preprocess_tes = preprocc(comencod, com, 2000, 2102, B_name_tes, lvl_loc, lvl_n); %s2
% preprocess_tes = preprocc(comencod, com, 1000, 8105, B_name_tes, lvl_loc, lvl_n); %s3
% preprocess_tes = preprocc(comencod, com, 3000, 1000, B_name_tes, lvl_loc, lvl_n); %s4
preprocess_tes = preprocc(comencod, com, 4000, 5001, B_name_tes, lvl_loc, lvl_n); %s5

%% Census tes
% prediction on test set before Relab
% protected feature
% B_name_tes = 'race';
% lvl_loc = 54; lvl_n =5; %location of encoded protected feature
% preprocess_tes = preprocc(censusincometesencod, censusincometes, 100, 2001, B_name_tes, lvl_loc, lvl_n); %try
% preprocess_tes = preprocc(censusincometesencod, censusincometes, 1000, 4001, B_name_tes, lvl_loc, lvl_n); %s1
% preprocess_tes = preprocc(censusincometesencod, censusincometes, 1000, 10001, B_name_tes, lvl_loc, lvl_n); %s2
% preprocess_tes = preprocc(censusincometesencod, censusincometes, 2000, 11001, B_name_tes, lvl_loc, lvl_n); %s3
% preprocess_tes = preprocc(censusincometesencod, censusincometes, 4000, 1, B_name_tes, lvl_loc, lvl_n); %s4
% preprocess_tes = preprocc(censusincometesencod, censusincometes, 4000, 3001, B_name_tes, lvl_loc, lvl_n); %s5

%% Default tes
% prediction on test set before Relab
% protected feature
% B_name_tes = 'SEX';
% lvl_loc = 2; lvl_n =2; %location of encoded protected feature
% preprocess_tes = preprocc(defaencod, defa, 100, 2001, B_name_tes, lvl_loc, lvl_n); %try
% preprocess_tes = preprocc(defaencod, defa, 1000, 2001, B_name_tes, lvl_loc, lvl_n); %s1
% preprocess_tes = preprocc(defaencod, defa, 500, 3501, B_name_tes, lvl_loc, lvl_n); %s2
% preprocess_tes = preprocc(defaencod, defa, 2000, 6001, B_name_tes, lvl_loc, lvl_n); %s3
% preprocess_tes = preprocc(defaencod, defa, 5000, 12001, B_name_tes, lvl_loc, lvl_n); %s4
% preprocess_tes = preprocc(defaencod, defa, 7000, 23001, B_name_tes, lvl_loc, lvl_n); %s5
% preprocess_tes = preprocc(defaencod, defa, 10000, 10001, B_name_tes, lvl_loc, lvl_n); %all

%%
X = preprocess.X;
Y = preprocess.Y;
B_loc = preprocess.B; %protected feature (not encoded) location in X
cols = preprocess.cols;
B_x = preprocess.B_x; % encoded B of the X (removed)

%% Build the decision tree
tic
tree = build_tree(X,Y,cols,fair,B_loc,Dep_lim);

% Display the tree
treeplot(tree.p');
title('Decision tree ("**" is an inconsistent node)');
[xs,ys,h,s] = treelayout(tree.p');
leaf_label = {};

for i = 2:numel(tree.p)
	% Get my coordinate
	my_x = xs(i);
	my_y = ys(i);

	% Get parent coordinate
	parent_x = xs(tree.p(i));
	parent_y = ys(tree.p(i));

	% Calculate weight coordinate (midpoint)
	mid_x = (my_x + parent_x)/2;
	mid_y = (my_y + parent_y)/2;

    % Edge label
	text(mid_x,mid_y,tree.labels{i-1});
    
    % Leaf label
    if ~isempty(tree.inds{i})
        val = Y(tree.inds{i});
        if numel(unique(val))==1
            text(my_x, my_y, sprintf('y=%2.2f\nn=%d', val(1), numel(val)));
            leaf_label = [leaf_label; [i , val(1)]]; 
        else
            %inconsistent data
            text(my_x, my_y, sprintf('**y=%2.2f\nn=%d', mode(val), numel(val)));
            leaf_label = [leaf_label; [i , mode(val)]]; 
        end
    end
end
%predictin indices
decsn = cell2mat(tree.decsn); % A decision rule for each node
p = tree.p ;
leaf_label = cell2mat(leaf_label);

%%
%acc and disc before Relab

% evaluation
prediction = pred(X, decsn, p, Dep_lim, leaf_label);
pred_inds_tr = prediction.inds;
pred_labels_tr = prediction.pred_labels;
acc_tr = accuracy(X, Y, leaf_label, decsn, p, Dep_lim, pred_inds_tr);
fprintf('acc tr before Relab %f\n',acc_tr*100);
disc_tr = discrep(X, Y, B_x, leaf_label, lvl_n, decsn, p, Dep_lim, pred_labels_tr);
fprintf('disc tr before Relab %f\n',disc_tr);

%%
X_tes = preprocess_tes.X;
Y_tes = preprocess_tes.Y;
B_x_tes = preprocess_tes.B_x; 

%%
% evaluation

prediction = pred(X_tes, decsn, p, Dep_lim, leaf_label);
pred_inds_tes = prediction.inds;
pred_labels_tes = prediction.pred_labels;
acc_tes = accuracy(X_tes, Y_tes, leaf_label, decsn, p, Dep_lim, pred_inds_tes);
fprintf('acc tes before Relab %f\n',acc_tes*100);
disc_tes = discrep(X_tes, Y_tes, B_x_tes, leaf_label, lvl_n, decsn, p, Dep_lim, pred_labels_tes);
fprintf('disc tes before Relab %f\n',disc_tes);
fprintf('fair = %d\n', fair); 

%%
% use training to calculate the relabaling costs and gains
deltas = cal_delta(X, Y, B_x, leaf_label, lvl_n, decsn, p, Dep_lim);

% Relab
eps = 0.05;
Relab = relab(deltas, leaf_label, eps,...
    discrep(X, Y, B_x, leaf_label, lvl_n, decsn, p, Dep_lim, pred_labels_tr ),...
    accuracy(X, Y, leaf_label, decsn, p, Dep_lim, pred_inds_tr));
toc
fprintf("acc train after relab = %f\n" , Relab.acc * 100);
fprintf("disc train after relab = %f\n", Relab.disc);

%%
% prediction = pred(X, decsn, p, Dep_lim, Relab.labls);
% pred_inds_tr = prediction.inds;
% pred_labels_tr = prediction.pred_labels;

prediction = pred(X_tes, decsn, p, Dep_lim, Relab.labls);
pred_inds_tes = prediction.inds;
pred_labels_tes = prediction.pred_labels;

% acc_tr = accuracy(X, Y, Relab.labls, decsn, p, Dep_lim, pred_inds_tr);
% fprintf('acc tr after Relab %f\n',acc_tr*100);
% disc_tr = discrep(X, Y, B_x, Relab.labls, lvl_n, decsn, p, Dep_lim, pred_labels_tr);
% fprintf('disc tr after Relab %f\n',disc_tr);

acc_tes = accuracy(X_tes, Y_tes, Relab.labls, decsn, p, Dep_lim, pred_inds_tes);
fprintf('acc tes after Relab %f\n',acc_tes*100);
disc_tes = discrep(X_tes, Y_tes, B_x_tes, Relab.labls, lvl_n, decsn, p, Dep_lim, pred_labels_tes);
fprintf('disc tes after Relab %f\n',disc_tes);

