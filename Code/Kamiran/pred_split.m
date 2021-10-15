% function [inds] = pred_split(node, X, Y, inds, decsn, Dep, Dep_lim)
function [inds] = pred_split(node, X, Y, inds, decsn)
% Recursively splits according to the nodes and features

% feature = cell2mat(decsn(cellfun(@(x) x(1) == node, decsn))); %name of the node is in the first 
% feature = feature(4);
feature = decsn(decsn(:,1) == node,4); %feature is in 4 
val = decsn(decsn(:,1) == node,5); %splitting value is in 4
child1 = decsn(decsn(:,1) == node,2);
child2 = decsn(decsn(:,1) == node,3);

if  isempty(val)
    return
end

curr_X = X(inds{node},:);
feat = curr_X(:,feature);
feat = feat < val;
inds = [inds; inds{node}(feat); inds{node}(~feat)];
inds{node} = [];

% Dep = Dep +1;
% if Dep <= Dep_lim
% [inds] = pred_split(child1, X, Y, inds, decsn, Dep, Dep_lim);
% [inds] = pred_split(child2, X, Y, inds, decsn, Dep, Dep_lim);
% end
[inds] = pred_split(child1, X, Y, inds, decsn);
[inds] = pred_split(child2, X, Y, inds, decsn);
end