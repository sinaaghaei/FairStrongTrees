function disc_sp = get_sp(B, Y_pr, deprived_group, positive_class)

prob_pos_outcome_deprived = size(B(B==deprived_group & Y_pr==positive_class,:))/size(B(B==deprived_group,:));
fprintf('size deprived %f\n',size(B(B==deprived_group & Y_pr==positive_class,:)));
fprintf('size deprived total %f\n',size(B(B==deprived_group,:)));

prob_pos_outcome_non_deprived = size(B(B~=deprived_group & Y_pr==positive_class,:))/size(B(B~=deprived_group,:));
fprintf('size notdeprived %f\n',size(B(B~=deprived_group & Y_pr==positive_class,:)));
fprintf('size notdeprived total %f\n',size(B(B~=deprived_group,:)));
disc_sp = prob_pos_outcome_non_deprived - prob_pos_outcome_deprived;