function delta = cal_delta(X, Y, leaf_label, decsn, p, Dep_lim, B_loc, deprived_group, positive_class)

acc = leaf_label;
acc(:,2) = 0;
disc = leaf_label;
disc(:,2) = 0;

for i=1:size(leaf_label,1)
    tem_leaf_label = leaf_label;
    tem_leaf_label(i,2) = ~tem_leaf_label(i,2);
    
    prediction_new = pred(X, decsn, p, Dep_lim, tem_leaf_label);
    pred_inds_new = prediction_new.inds;
    pred_labels_new = prediction_new.pred_labels;
    
    prediction = pred(X, decsn, p, Dep_lim, leaf_label);
    pred_inds = prediction.inds;
    pred_labels = prediction.pred_labels;
    
    acc(i,2) = accuracy(X, Y, tem_leaf_label, decsn, p, Dep_lim, pred_inds_new)- ...
        accuracy(X, Y, leaf_label, decsn, p, Dep_lim, pred_inds);
    
    disc(i,2) = get_sp(X(:,B_loc),pred_labels_new,deprived_group,positive_class)- ...
                get_sp(X(:,B_loc),pred_labels,deprived_group,positive_class);
end

delta.acc = acc;
delta.disc = disc;

end