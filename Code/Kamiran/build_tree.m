function t = build_tree(X,Y,cols,fair,B_loc,Dep_lim)
% Builds a decision tree to predict Y from X.  The tree is grown by
% recursively splitting each node using the feature which gives the best
% information gain until the leaf is consistent or all inputs have the same
% feature values.
%
% X is an nxm matrix, where n is the number of points and m is the
% number of features.
% Y is an nx1 vector of classes
% cols is a cell-vector of labels for each feature
%
% RETURNS t, a structure with three entries:
% t.p is a vector with the index of each node's parent node
% t.inds is the rows of X in each node (non-empty only for leaves)
% t.labels is a vector of labels showing the decision that was made to get
%     to that node

% Create an empty decision tree, which has one node and everything in it
inds = {1:size(X,1)}; % A cell per node containing indices of all data in that node
p = 0; % Vector contiaining the index of the parent node for each node
labels = {}; % A label for each node
decsn = {}; % A decision rule for each node

% Create tree by splitting on the root
Dep = 0; % control the depth (number of nodes)
[inds, p ,labels, decsn] = split_node(X, Y, inds, p, labels, cols, decsn, 1, fair, B_loc, Dep, Dep_lim);


t.inds = inds;
t.p = p;
t.labels = labels;
t.decsn = decsn;



function [inds ,p ,labels, decsn] = split_node(X, Y, inds, p, labels, cols, decsn, node, fair, B_loc, Dep, Dep_lim)
% Recursively splits nodes based on information gain

% Check if the current leaf is consistent

if numel(unique(Y(inds{node}))) == 1 || numel(unique(X(inds{node},B_loc))) == 1
% if numel(unique(Y(inds{node}))) == 1 
    return;
end

% Check if all inputs have the same features
% We do this by seeing if there are multiple unique rows of X
if size(unique(X(inds{node},:),'rows'),1) == 1
    return;
end

% Otherwise, we need to split the current node on some feature

best_ig = -inf; %best information gain
best_feature = 0; %best feature to split on
best_val = 0; % best value to split the best feature on

curr_X = X(inds{node},:);
curr_Y = Y(inds{node});
X_curr_B = X(inds{node},B_loc);
% Loop over each feature
for i = 1:size(X,2)
    feat = curr_X(:,i);
    
    % Deterimine the values to split on
    vals = unique(feat);
    splits = 0.5*(vals(1:end-1) + vals(2:end));
    if numel(vals) < 2
        continue
    end
    
    % Get binary values for each split value
    bin_mat = double(repmat(feat, [1 numel(splits)]) < repmat(splits', [numel(feat) 1]));
    
    % Compute the information gains
    H = ent(curr_Y,X_curr_B, fair);
    H_cond = zeros(1, size(bin_mat,2));
    for j = 1:size(bin_mat,2)
        H_cond(j) = cond_ent(curr_Y,X_curr_B, bin_mat(:,j), fair);
    end
    IG = H - H_cond; % IG = IGC+IGS
    
    % Find the best split
    [val, ind] = max(IG);
    if val > best_ig
        best_ig = val;
        best_feature = i;
        best_val = splits(ind);
    end
end

% Split the current node into two nodes
feat = curr_X(:,best_feature);
feat = feat < best_val;
inds = [inds; inds{node}(feat); inds{node}(~feat)];
inds{node} = [];
p = [p; node; node];
labels = [labels; sprintf('%s < %2.2f', cols{best_feature}, best_val); ...
    sprintf('%s >= %2.2f', cols{best_feature}, best_val)];
decsn = [decsn; [node, best_feature, best_val]]; %node, feature, value to splite on

% Recurse on newly-create nodes
n = numel(p)-2;
Dep = Dep +1;
if Dep <= Dep_lim
    [inds, p, labels, decsn] = split_node(X, Y, inds, p, labels, cols, decsn, n+1, fair, B_loc, Dep, Dep_lim);
    [inds, p, labels, decsn] = split_node(X, Y, inds, p, labels, cols, decsn, n+2, fair, B_loc, Dep, Dep_lim);
end

