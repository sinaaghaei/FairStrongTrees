function pred_i = pred(X_pr, decsn, p, Dep_lim, leaf_label)

inds = {1:size(X_pr,1)}; % A cell per node containing indices of all data in that node

% Create tree by splitting on the root
Dep = 0; % control the depth that go in when predicting (number of nodes)
node = 1;
dum_p = 0;
[inds, dum_p] = pred_split(X_pr, inds, p, dum_p, decsn, node, Dep, Dep_lim); %we use dum_p as a dummy variable to control nodes

pred_i.inds = inds;

pred_labels = zeros(size(X_pr,1),1); % prediction of the records
for i=1:size(inds,1)
    if ~isempty(inds{i})
         if ~sum(ismember(inds{i},0))
             pred_labels(inds{i}) = leaf_label(leaf_label(:,1) == i,2); 
         end
    end
end
pred_i.pred_labels = pred_labels;
end

function [inds, dum_p] = pred_split(X_pr, inds, p, dum_p, decsn, node, Dep, Dep_lim)
% Recursively splits according to the nodes and features

% feature = cell2mat(decsn(cellfun(@(x) x(1) == node, decsn))); %name of the node is in the first 
% feature = feature(4);
feature = decsn(decsn(:,1) == node,2); %feature is in 2
val = decsn(decsn(:,1) == node,3); %splitting value is in 3

% if isempty(feature) || sum(inds{node} == 0) ~= 0
if isempty(feature) || sum(inds{node} == 0) ~= 0
%     Dep = Dep +1;
%     if Dep <= Dep_lim
%         [inds] = pred_split(X_pr, inds, p, decsn, node+1, Dep, Dep_lim);
%         [inds] = pred_split(X_pr, inds, p, decsn, node+2, Dep, Dep_lim);
%     end
    return
end

curr_X = X_pr(inds{node},:);
feat = curr_X(:,feature);
feat = feat < val;
if isempty(curr_X(feat,feature)) || isempty(curr_X(~feat,feature))
    if isempty(curr_X(feat,feature))
        inds = [inds; 0; inds{node}(~feat)];
    else
        inds = [inds; inds{node}(feat); 0];
    end
else
    inds = [inds; inds{node}(feat); inds{node}(~feat)];
end
inds{node} = [];
dum_p = [dum_p; node; node];

% Recurse on newly-create nodes
n = numel(dum_p)-2;
Dep = Dep +1;
if Dep <= Dep_lim && sum(ismember(p,node))   %if the node is a parent (is in p) the go to next childs
    [inds, dum_p] = pred_split(X_pr, inds, p, dum_p, decsn, n+1, Dep, Dep_lim);
    [inds, dum_p] = pred_split(X_pr, inds, p, dum_p, decsn, n+2, Dep, Dep_lim);
end
end