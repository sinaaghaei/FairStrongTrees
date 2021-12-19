%%Assumptions
% 1- We assume that we have a single protected attribute with binary levels
% 2- We don't use the protected feature as a splitting feature while
% growing the tree

function res = Copy_of_run_exp(data_train, data_train_enc, data_test, data_test_enc, ...
    B_name, positive_class, deprived_group, lvl_loc, lvl_n, eps, fair, Dep_lim)
% Preprocess the data
preprocess = preProcess(data_train_enc, data_train, B_name, lvl_loc, lvl_n)
X = preprocess.X;
Y = preprocess.Y;
B_loc = preprocess.B; %protected feature (not encoded) location in X
cols = preprocess.cols;

% Preprocess the test data
preprocess_tes = preProcess(data_test_enc, data_test, B_name, lvl_loc, lvl_n)
X_tes = preprocess_tes.X;
Y_tes = preprocess_tes.Y;

% Build decision tree
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

%acc and disc before Relab

% evaluation
prediction = pred(X, decsn, p, Dep_lim, leaf_label);
pred_inds_tr = prediction.inds;
pred_labels_tr = prediction.pred_labels;
% acc_tr = accuracy(X, Y, leaf_label, decsn, p, Dep_lim, pred_inds_tr);
acc_tr = sum(Y==pred_labels_tr)/size(Y,1);
res.acc_tr_pre = acc_tr*100;
fprintf('acc tr before Relab %f\n',acc_tr*100);
disc_tr = get_sp(X(:,B_loc),pred_labels_tr,deprived_group,positive_class);
res.disc_tr_pre = disc_tr;
fprintf('disc tr before Relab %f\n',disc_tr);

% evaluation

prediction = pred(X_tes, decsn, p, Dep_lim, leaf_label);
pred_inds_tes = prediction.inds;
pred_labels_tes = prediction.pred_labels;
% acc_tes = accuracy(X_tes, Y_tes, leaf_label, decsn, p, Dep_lim, pred_inds_tes);
acc_tes = sum(Y_tes==pred_labels_tes)/size(Y_tes,1);
fprintf('acc tes before Relab %f\n',acc_tes*100);
res.acc_te_pre = acc_tes*100;
disc_tes = get_sp(X_tes(:,B_loc),pred_labels_tes,deprived_group,positive_class);
fprintf('disc tes before Relab %f\n',disc_tes);
res.disc_te_pre = disc_tes;
fprintf('fair = %d\n', fair); 

% use training to calculate the relabaling costs and gains
deltas = cal_delta(X, Y, leaf_label, decsn, p, Dep_lim, B_loc, deprived_group, positive_class);

% Relab
Relab = relab(deltas, leaf_label, eps,...
    get_sp(X(:,B_loc),pred_labels_tr,deprived_group,positive_class),...
    accuracy(X, Y, leaf_label, decsn, p, Dep_lim, pred_inds_tr));
toc
fprintf("acc train after relab = %f\n" , Relab.acc * 100);
fprintf("disc train after relab = %f\n", Relab.disc);

res.acc_tr_post = Relab.acc*100;
res.disc_tr_post = Relab.disc;

%Evaluation on test after relab
prediction = pred(X_tes, decsn, p, Dep_lim, Relab.labls);
pred_inds_tes = prediction.inds;
pred_labels_tes = prediction.pred_labels;

acc_tes = sum(Y_tes==pred_labels_tes)/size(Y_tes,1);
fprintf('acc tes after Relab %f\n',acc_tes*100);
disc_tes = get_sp(X_tes(:,B_loc),pred_labels_tes,deprived_group,positive_class);
fprintf('disc tes after Relab %f\n',disc_tes);

res.acc_te_post = acc_tes*100;
res.disc_te_post = disc_tes;

