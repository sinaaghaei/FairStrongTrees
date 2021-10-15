function dis = discrep(X_pr, Y_pr, B_x_pr, leaf_label, lvl_n, decsn, p, Dep_lim, pred_labels)

classes = 0:1;
m = size(X_pr,1);

disc = 0;
% for i = 1:lvl_n
%     for j = i+1:lvl_n
%         if (size(X_pr((B_x_pr(:,i)==1) & (Y_pr == 1) ,:),1) ~= 0) && (size(X_pr( (B_x_pr(:,j)==1) & (Y_pr == 1),:),1) ~=0)
%             dic_tem_1 = accuracy(X_pr((B_x_pr(:,i)==1) & (Y_pr == 1) ,:)...
%                 , Y_pr((B_x_pr(:,i)==1) & (Y_pr == 1),:), leaf_label, decsn, p, Dep_lim);
%             dic_tem_2 = accuracy(X_pr((B_x_pr(:,j)==1) & (Y_pr == 1) ,:)...
%                 , Y_pr((B_x_pr(:,j)==1) & (Y_pr == 1),:), leaf_label, decsn, p, Dep_lim);
%             disc = disc + abs(dic_tem_1 - dic_tem_2);
%         end
%     end
% end

% prediction = pred(X_pr, decsn, p, Dep_lim, leaf_label);
% pred_labels = prediction.pred_labels;

half_range = (max(X_pr)- min(X_pr))/2;
half_range(half_range==0)=1;
X_pr_norm = (X_pr-repmat(mean(X_pr),m,1))./repmat(half_range,m,1);

k_nn = 20;
knn_mat = knn(X_pr_norm(:,1:end-1), k_nn);

for up=classes
    for Bj=1:lvl_n
        for Bk=Bj+1:lvl_n
            if (size(X_pr((B_x_pr(:,Bj)==1) & (Y_pr == up) ,:),1) ~= 0) && (size(X_pr( (B_x_pr(:,Bk)==1) & (Y_pr == up),:),1) ~=0)
                num = 0;
                denum_sum_d = 0;
                for j=1:m
                    for jp=1:k_nn
                        if (B_x_pr(j,Bj)==1) && (B_x_pr(knn_mat(j,jp),Bk)==1) && (Y_pr(j)==up) && (Y_pr(knn_mat(j,jp))==up)
                            num = num + abs(pred_labels(j,1)-pred_labels(knn_mat(j,jp),1)); % protected is in the end and I remove it
%                             denum_sum_d = denum_sum_d + dis_inverse(X_pr_norm(j,1:end-1),X_pr_norm(knn_mat(j,jp),1:end-1));
                            denum_sum_d = denum_sum_d + 1;
                        end
                    end
                end
                if denum_sum_d==0
                    denum_sum_d=1;
                end
                disc = disc+ num/denum_sum_d;
            end
        end
    end
end
dis = disc;
end