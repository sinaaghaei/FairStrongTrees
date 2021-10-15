%%
%Load data
clearvars
% data_group = 'compas';
data_train = readtable('../../DataSets/compas_train_calibration_1.csv');
data_train_enc = readtable('../../DataSets/compas_train_calibration_enc_1.csv');
data_train_enc.('target') = str2double(data_train_enc.('target'));

%%
%fair coef (on/off)
fair =1; %fair coef
Dep_lim = 2; % max depth is Dep_lim + 1

%%
B_name = 'race';
lvl_loc = 1; lvl_n =4; %location of encoded protected feature
preprocess = preProcess(data_train_enc, data_train, B_name, lvl_loc, lvl_n)

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

