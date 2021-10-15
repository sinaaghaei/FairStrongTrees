function acc = accuracy(X_pr, Y_pr, leaf_label, decsn, p, Dep_lim, pred_inds)

% prediction = pred(X_pr, decsn, p, Dep_lim, leaf_label);
% pred_inds = prediction.inds;

acc = 0; %number of accurate predictions
for i=1:size(pred_inds,1)
    if ~isempty(pred_inds{i})
         if ~sum(ismember(pred_inds{i},0)) 
             if ~isempty( Y_pr(pred_inds{i}) == repmat(leaf_label(leaf_label(:,1) == i,2),size(pred_inds{i}))')
                 acc = acc + sum( double( Y_pr(pred_inds{i}) == repmat(leaf_label(leaf_label(:,1) == i,2),size(pred_inds{i}))') ); 
             end
         end
    end
end
acc = acc/size(Y_pr,1);

end