function result = cond_ent(Y,X_B, X,fair)
% Calculates the conditional entropy of y given x
result = 0;

if fair == 0
    tab = tabulate(X);
    % Remove zero-entries
    tab = tab(tab(:,3)~=0,:);
    for i = 1:size(tab,1)
        % Get entropy for y values where x is the current value
        H = ent(Y(X == tab(i,1)),0,fair);
        % Get probability
        prob = tab(i, 3) / 100;
        % Combine
        result = result + prob * H;
    end
else
    tab = tabulate(X);
    % Remove zero-entries
    tab = tab(tab(:,3)~=0,:);
    for i = 1:size(tab,1)
        % Get entropy for y values where x is the current value w.r.t class
        % and B at same time in ent
        H_class_B = ent(Y(X == tab(i,1)),X_B(X == tab(i,1)),fair);
        % Get probability
        prob = tab(i, 3) / 100;
        % Combine
        result = result + prob * H_class_B;
    end
end

    
