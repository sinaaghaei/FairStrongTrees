function rel = relab(deltas, leaf_label, eps, discrep, accuracy)

labls = leaf_label;
deltas.disc(deltas.disc(:,2)>0,2) = 0;
leaf_ratio = [leaf_label(:,1) , abs(deltas.disc(:,2)) ./ abs(deltas.acc(:,2))];
[a, index] = sort(leaf_ratio,'descend');
disc = discrep;
acc = accuracy;
i = 1;
while  disc > eps && i <= size(leaf_label,1)
    if deltas.disc(index(i,2),2) <0
        disc = disc + deltas.disc(index(i,2),2);
        acc = acc + deltas.acc(index(i,2),2);
        
        labls(index(i,2),2) = ~labls(index(i,2),2);
    end
    i = i+1;
    if disc < 0 
        disc = 0;
    end
end

rel.labls = labls;
rel.disc = disc;
rel.acc = acc;

end