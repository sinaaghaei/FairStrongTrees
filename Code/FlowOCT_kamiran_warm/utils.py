import numpy as np


def get_node_status(grb_model, b, beta, p, n):
    '''
    This function give the status of a given node in a tree. By status we mean whether the node
        1- is pruned? i.e., we have made a prediction at one of its ancestors
        2- is a branching node? If yes, what feature do we branch on
        3- is a leaf? If yes, what is the prediction at this node?

    :param grb_model: the gurobi model solved to optimality (or reached to the time limit)
    :param b: The values of branching decision variable b
    :param beta: The values of prediction decision variable beta
    :param p: The values of decision variable p
    :param n: A valid node index in the tree
    :return: pruned, branching, selected_feature, leaf, value

    pruned=1 iff the node is pruned
    branching = 1 iff the node branches at some feature f
    selected_feature: The feature that the node branch on
    leaf = 1 iff node n is a leaf in the tree
    value: if node n is a leaf, value represent the prediction at this node
    '''
    tree = grb_model.tree
    pruned = False
    branching = False
    leaf = False
    value = None
    selected_feature = None

    p_sum = 0
    for m in tree.get_ancestors(n):
        p_sum = p_sum + p[m]
    if p[n] > 0.5:  # leaf
        leaf = True
        for k in grb_model.labels:
            if beta[n, k] > 0.5:
                value = k
    elif p_sum == 1:  # Pruned
        pruned = True

    if n in tree.Nodes:
        if (pruned == False) and (leaf == False):  # branching
            for f in grb_model.cat_features:
                if b[n, f] > 0.5:
                    selected_feature = f
                    branching = True

    return pruned, branching, selected_feature, leaf, value


def print_tree(grb_model, b, beta, p):
    '''
    This function print the derived tree with the branching features and the predictions asserted for each node
    :param grb_model: the gurobi model solved to optimality (or reached to the time limit)
    :param b: The values of branching decision variable b
    :param beta: The values of prediction decision variable beta
    :param p: The values of decision variable p
    :return: print out the tree in the console
    '''
    tree = grb_model.tree
    for n in tree.Nodes + tree.Leaves:
        pruned, branching, selected_feature, leaf, value = get_node_status(grb_model, b, beta, p, n)
        print('#########node ', n)
        if pruned:
            print("pruned")
        elif branching:
            print(selected_feature)
        elif leaf:
            print('leaf {}'.format(value))




def get_leaf_info(grb_model,train_enc,label, b, beta, p,zeta):
    leaf_dict = {}
    for current in grb_model.tree.Nodes + grb_model.tree.Leaves:
        pruned, branching, selected_feature, leaf, value = get_node_status(grb_model, b, beta, p, current)
        if leaf:
            tmp_train = train_enc[train_enc[label] == value]
            # denum = sum_{i} zeta[i,current,value]
            denum = dict(filter(lambda elem: elem[0][1] == current and elem[0][2] == value, zeta.items()))
            denum = sum(denum.values())

            # num = sum_{i: y_i = value} zeta[i,current,value]
            num = dict(filter(lambda elem: elem[0][0] in tmp_train.index and elem[0][1] == current and elem[0][2] == value, zeta.items()))
            num = sum(num.values())


            prob =  num/denum
            leaf_dict[current] = (value, prob)
    return leaf_dict

def get_predicted_probability(grb_model, local_data,leaf_dict, b, beta, p, i):
    '''
    This function returns the predicted probability for a given datapoint
    :param grb_model: The gurobi model we solved
    :param local_data: The dataset we want to compute accuracy for
    :param b: The value of decision variable b
    :param beta: The value of decision variable beta
    :param p: The value of decision variable p
    :param i: Index of the datapoint we are interested in
    :return: The predicted value for datapoint i in dataset local_data
    '''
    tree = grb_model.tree
    current = 1

    while True:
        pruned, branching, selected_feature, leaf, value = get_node_status(grb_model, b, beta, p, current)
        if leaf:
            return leaf_dict[current][1]
        elif branching:
            if local_data.at[i, selected_feature] == 1:  # going right on the branch
                current = tree.get_right_children(current)
            else:  # going left on the branch
                current = tree.get_left_children(current)


def get_predicted_value(grb_model, local_data, b, beta, p, i):
    '''
    This function returns the predicted value for a given datapoint
    :param grb_model: The gurobi model we solved
    :param local_data: The dataset we want to compute accuracy for
    :param b: The value of decision variable b
    :param beta: The value of decision variable beta
    :param p: The value of decision variable p
    :param i: Index of the datapoint we are interested in
    :return: The predicted value for datapoint i in dataset local_data
    '''
    tree = grb_model.tree
    current = 1

    while True:
        pruned, branching, selected_feature, leaf, value = get_node_status(grb_model, b, beta, p, current)
        if leaf:
            return value
        elif branching:
            if local_data.at[i, selected_feature] == 1:  # going right on the branch
                current = tree.get_right_children(current)
            else:  # going left on the branch
                current = tree.get_left_children(current)

def get_sp(grb_model, local_data_enc, local_data_reg, b, beta, p, deprived_group, positive_class, source):
    '''
        This function returns the statistical parity for a given combination of the protected feature
        :param grb_model: The gurobi model we solved
        :param local_data: The dataset we want to compute accuracy for
        :param b: The value of decision variable b
        :param beta: The value of decision variable beta
        :param p: The value of decision variable p
        :param deprived_group: deprived Group
        :param source: Statistical parity for given data label or predicted label
        :return: The statistical parity for a given combination of the protected feature
        '''
    # If source == "Predictions", then we will use prediction values
    # If source == "Data", then we will use data values
    protected_feature = grb_model.protected_feature

    # The label here will then be the true label of the data
    label = grb_model.label

    # For our purposes, we only want to look at values of protected group and non-protected group hence
    # we are creating 2 new df's with only the groups of interest
    df_non_deprived= local_data_reg[local_data_reg[protected_feature] != deprived_group]
    df_deprived = local_data_reg[local_data_reg[protected_feature] == deprived_group]

    # Create dataframe for the protected group only, then count how many rows exist
    count_non_deprived= df_non_deprived.count()[label]
    # Create dataframe for the protected group prime only, then count how many rows exist
    count_deprived = df_deprived.count()[label]

    # Looking at the statistical parity for the true label between groups
    # Akin to looking at data bias
    if source == "Data":

        # Let's count number of positive values from protected group, then divide by the total to get the SP for
        # both groups
        sp_non_deprived = 0
        sp_deprived = 0
        if count_non_deprived != 0 and count_deprived != 0:
            sp_non_deprived = (1/count_non_deprived) * df_non_deprived[df_non_deprived[label] == positive_class].count()[label]
            sp_deprived = (1/count_deprived) * df_deprived[df_deprived[label] == positive_class].count()[label]
            # Return SP between two groups
            return (sp_non_deprived - sp_deprived)
        else:
            return None


    # Looking at the statistcal parity between the two groups relative to predicted values
    # Akin to measuring our model's bias
    elif source == "Predictions":

        # Define statistical parity for both groups
        sp_non_deprived_predictions = 0
        sp_deprived_predictions = 0
        if count_non_deprived != 0 and count_deprived != 0:
            sp_non_deprived_predictions = (1 / count_non_deprived) * df_non_deprived[df_non_deprived['Predictions'] == positive_class].count()[label]
            sp_deprived_predictions = (1 / count_deprived) * df_deprived[df_deprived['Predictions'] == positive_class].count()[label]

            # Return sp between both groups
            return (sp_non_deprived_predictions - sp_deprived_predictions)

        else:
            return None

    # If no source is given, then we will return an error
    else:
        print('No valid source passed')


def get_acc(grb_model, local_data, b, beta, p):
    '''
    This function returns the accuracy of the prediction for a given dataset
    :param grb_model: The gurobi model we solved
    :param local_data: The dataset we want to compute accuracy for
    :param b: The value of decision variable b
    :param beta: The value of decision variable beta
    :param p: The value of decision variable p
    :return: The accuracy (fraction of datapoints which are correctly classified)
    '''
    label = grb_model.label
    acc = 0
    for i in local_data.index:
        yhat_i = get_predicted_value(grb_model, local_data, b, beta, p, i)
        y_i = local_data.at[i, label]
        if yhat_i == y_i:
            acc += 1

    acc = acc / len(local_data.index)
    return acc
