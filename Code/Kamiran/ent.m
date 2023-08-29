function result = ent(Y,X_B,fair)
% Calculates the entropy of a vector of values 

if fair == 0
    % Get frequency table
    tab = tabulate(Y);
    prob = tab(:,3) / 100;
    % Filter out zero-entries
    prob = prob(prob~=0);
    % Get entropy
    result = -sum(prob .* log2(prob));
else
    % Get frequency table
    tab_class = tabulate(Y);
    prob_class = tab_class(:,3) / 100;
    % Filter out zero-entries
    prob_class = prob_class(prob_class~=0);
    % Get entropy
    ent_class = -sum(prob_class .* log2(prob_class));
    
    % Get frequency table w.r.t B
    tab_B = tabulate(X_B);
    prob_B = tab_B(:,3) / 100;
    % Filter out zero-entries
    prob_B = prob_B(prob_B~=0);
    % Get entropy
    ent_B = -sum(prob_B .* log2(prob_B));
    
    % WE CHANGE THIS TO BE - if we want to do IGC-IGS
    result = ent_class - fair * ent_B; %H_class + H_B
end
end


