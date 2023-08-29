#!/usr/bin/python
from gurobipy import *
import pandas as pd
import sys
import time
from Tree import Tree
from FlowOCT import FlowOCT
import logger
import getopt
import csv
from sklearn.model_selection import train_test_split
from utils import *
from logger import logger
from itertools import combinations
import operator
import os


def main(argv):
    print(argv)
    train_file_reg = None
    train_file_enc = None
    test_file_reg = None
    test_file_enc = None
    calibration_file_reg = None
    calibration_file_enc = None
    depth = None
    time_limit = None
    _lambda = None
    fairness_type = None
    fairness_bound = None
    protected_feature = None
    positive_class = None
    conditional_feature = None
    calibration_mode = None
    input_sample = None
    '''
    Depending on the value of input_sample we choose one of the following random seeds and then split the whole data
    into train, test and calibration
    '''
    random_states_list = [41, 23, 45, 36, 19, 123]

    try:
        opts, args = getopt.getopt(argv,'a:b:c:d:e:f:g:h:i:j:k:l:m:n:o:p:',
                                   ["train_file_reg=", "train_file_enc=",
                                   "test_file_reg=", "test_file_enc=",
                                    "calibration_file_reg=", "calibration_file_enc=",
                                    "depth=", "timelimit=", "_lambda=",
                                    "fairness_type=", "fairness_bound=","protected_feature=",
                                    'positive_class=', "conditional_feature=","calibration_mode=","sample="])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-a',"--train_file_reg"):
            train_file_reg = arg
        elif opt in ('-b',"--train_file_enc"):
            train_file_enc = arg
        elif opt in ('-c',"--test_file_reg"):
            test_file_reg = arg
        elif opt in ('-d',"--test_file_enc"):
            test_file_enc = arg
        elif opt in ('-e',"--calibration_file_reg"):
            calibration_file_reg = arg
        elif opt in ('-f',"--calibration_file_enc"):
            calibration_file_enc = arg
        elif opt in ('-g',"--depth"):
            depth = int(arg)
        elif opt in ('-h',"--timelimit"):
            time_limit = int(arg)
        elif opt in ('-i',"--_lambda"):
            _lambda = float(arg)
        elif opt in ('-j',"--fairness_type"):
            fairness_type = arg
        elif opt in ('-k',"--fairness_bound"):
            fairness_bound = float(arg)
        elif opt in ('-l',"--protected_feature"):
            protected_feature = arg
        elif opt in ('-m',"--positive_class"):
            positive_class = int(arg)
        elif opt in ('-n',"--conditional_feature"):
            conditional_feature = arg
        elif opt in ('-o',"--calibration_mode"):
            calibration_mode = arg
        elif opt in ('-p',"--sample"):
            input_sample = int(arg)

    ##########################################################
    # output setup
    ##########################################################
    approach_name = 'FlowOCT'
    out_put_name = f'{train_file_reg}_{approach_name}_d_{depth}_t_{time_limit}_lambda_{_lambda}_ft_{fairness_type}_fb_{fairness_bound}'
    out_put_path = os.getcwd() + '/../../Results/'

    # check if CSV already exists -- if so, don't run the code
    if os.path.exists(os.path.join(out_put_path, f'{out_put_name}.csv')):
        return

    # Using logger we log the output of the console in a text file
    sys.stdout = logger(out_put_path + out_put_name + '.txt')


    data_path = os.getcwd() + '/../../DataSets/'

    data_train_reg = pd.read_csv(data_path + train_file_reg)
    data_train_enc = pd.read_csv(data_path + train_file_enc)
    data_test_reg = pd.read_csv(data_path + test_file_reg)
    data_test_enc = pd.read_csv(data_path + test_file_enc)
    data_calibration_reg = pd.read_csv(data_path + calibration_file_reg)
    data_calibration_enc = pd.read_csv(data_path + calibration_file_enc)



    train_len = len(data_train_enc.index)


    '''Name of the column in the dataset representing the class label.
    In the datasets we have, we assume the label is target. Please change this value at your need'''
    label = 'target'

    # Tree structure: We create a tree object of depth d
    tree = Tree(depth)

    ##########################################################
    # Creating and Solving the problem
    ##########################################################
    # We create the MIP problem by passing the required arguments
    primal = FlowOCT(data_train_enc, data_train_reg, label, tree, _lambda, time_limit, fairness_type, fairness_bound, protected_feature, positive_class, conditional_feature)

    primal.create_primal_problem()
    start_time = time.time()
    primal.model.update()
    primal.model.optimize()
    end_time = time.time()
    solving_time = end_time - start_time

    ##########################################################
    # Preparing the output
    ##########################################################
    b_value = primal.model.getAttr("X", primal.b)
    beta_value = primal.model.getAttr("X", primal.beta)
    p_value = primal.model.getAttr("X", primal.p)
    zeta_value = primal.model.getAttr("X", primal.zeta)


    print("\n\n")
    print_tree(primal,b_value, beta_value, p_value)

    print('\n\nTotal Solving Time', solving_time)

    # print(b_value)
    # print(p_value)
    # print(beta_value)
    ##########################################################
    # Evaluation
    ##########################################################
    '''
    For classification performance we report accuracy

    over training, test and the calibration set
    '''
    train_acc = test_acc = calibration_acc = 0

    train_acc = get_acc(primal, data_train_enc, b_value, beta_value, p_value)
    test_acc = get_acc(primal, data_test_enc, b_value, beta_value, p_value)
    calibration_acc = get_acc(primal, data_calibration_enc, b_value, beta_value, p_value)

    print("obj value", primal.model.getAttr("ObjVal"))
    print("train acc", train_acc)
    print("test acc", test_acc)
    print("calibration acc", calibration_acc)


    # Let's pass in the predicted values for data's predicted values
    yhat_train = []
    yhat_test = []
    yhat_calib = []

    # yhat_prob_train = []
    # yhat_prob_test = []
    # yhat_prob_calib = []
    # leaf_dict = get_leaf_info(primal,data_train_enc,label, b_value, beta_value, p_value,zeta_value)
    for i in data_train_enc.index:
        yhat_train.append(get_predicted_value(primal, data_train_enc, b_value, beta_value, p_value, i))
        # yhat_prob_train.append(get_predicted_probability(primal, data_train_enc, leaf_dict, b_value, beta_value, p_value, i))
    for i in data_test_enc.index:
        yhat_test.append(get_predicted_value(primal, data_test_enc, b_value, beta_value, p_value, i))
        # yhat_prob_test.append(get_predicted_probability(primal, data_test_enc,leaf_dict, b_value, beta_value, p_value, i))
    for i in data_calibration_enc.index:
        yhat_calib.append(get_predicted_value(primal, data_calibration_enc, b_value, beta_value, p_value, i))
        # yhat_prob_calib.append(get_predicted_probability(primal, data_calibration_enc, leaf_dict, b_value, beta_value, p_value, i))


    data_train_reg['Predictions'] = yhat_train
    data_test_reg['Predictions'] = yhat_test
    data_calibration_reg['Predictions'] = yhat_calib

    # data_train_reg['Predictions_prob'] = yhat_prob_train
    # data_test_reg['Predictions_prob'] = yhat_prob_test
    # data_calibration_reg['Predictions_prob'] = yhat_prob_calib



    data_dict = {'train':(data_train_enc,data_train_reg),
                 'test':(data_test_enc,data_test_reg),
                 'calib':(data_calibration_enc,data_calibration_reg)}
    protected_levels = data_train_reg[protected_feature].unique()
    conditional_feature_levels = data_train_reg[conditional_feature].unique()
    fairness_metrics_dict = {}
    # fairness_metrics_dict[('SP','train','pred','max_diff')] = (max_val, p, p_prime)
    #fairness_metrics_dict[('CSP','train','pred','max_diff')] = (max_val, p, p_prime, L_name, L_level)
    def getFairnessResults(fairness_const_type):
        if fairness_const_type == 'SP':
            var_func = get_sp
        elif fairness_const_type == 'CSP':
            var_func = get_csp
        elif fairness_const_type == 'PE':
            var_func = get_pe
        elif fairness_const_type == 'EOpp':
            var_func = get_eopp
        elif fairness_const_type == 'EOdds':
            var_func = get_eodds


        for data_set in ['train','test','calib']:
            data_set_enc, data_set_reg = data_dict[data_set]
            for source in ['Data','Predictions']:
                max_value = 0
                if source == 'Data' and fairness_const_type in ['PE','EOpp','EOdds']:
                    continue
                for combos in combinations(protected_levels, 2):
                    p = combos[0]
                    p_prime = combos[1]
                    if fairness_const_type != 'CSP':
                        if  not would_be_added(fairness_const_type, p, p_prime,protected_feature,None,None, data_train_reg, label, positive_class):
                            continue
                        tmp_value = var_func(primal, data_set_enc, data_set_reg, b_value, beta_value, p_value, p, p_prime, positive_class, source, None, None)
                        if tmp_value >= max_value:
                            max_value = tmp_value
                            fairness_metrics_dict[(fairness_const_type,data_set,source,'max_diff')] = (max_value, p, p_prime)
                    else:
                        for feature_value in conditional_feature_levels:
                            if  not would_be_added(fairness_const_type, p, p_prime,protected_feature,conditional_feature,feature_value, data_train_reg, label, positive_class):
                                continue
                            tmp_value = var_func(primal, data_set_enc, data_set_reg, b_value, beta_value, p_value, p, p_prime, positive_class, source, conditional_feature, feature_value)
                            if tmp_value >= max_value:
                                max_value = tmp_value
                                fairness_metrics_dict[(fairness_const_type,data_set,source,'max_diff')] = (max_value, p, p_prime, conditional_feature, feature_value)


    # Print all maximum values with the corresponding protected varibles
    for fairness_const_type in ['SP','CSP','PE','EOpp','EOdds']:
        getFairnessResults(fairness_const_type)
        print('###################{} Results'.format(fairness_const_type))
        for data_set in ['train','test','calib']:
            for source in ['Data','Predictions']:
                if source == 'Data' and fairness_const_type in ['PE','EOpp','EOdds']:
                    continue
                if fairness_const_type != 'CSP':
                    max_value, p, p_prime = fairness_metrics_dict[(fairness_const_type,data_set,source,'max_diff')]
                    print('{} & {} has {} {} {}: {} '.format(p,p_prime, data_set,source,fairness_const_type, max_value))
                else:
                    max_value, p, p_prime, conditional_feature, feature_value = fairness_metrics_dict[(fairness_const_type,data_set,source,'max_diff')]
                    print('{} & {} with feature and feature value: {} = {} has {} {} {}: {} '.format(p,p_prime,conditional_feature, feature_value, data_set,source,fairness_const_type, max_value))

    ##########################################################
    # writing info to the file
    ##########################################################
    primal.model.write(out_put_path + out_put_name + '.lp')
    # writing info to the file
    result_file = out_put_name + '.csv'
    with open(out_put_path + result_file, mode='a') as results:
        results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        results_writer.writerow(
            [approach_name, train_file_reg,input_sample, fairness_type, fairness_bound, train_len,calibration_mode, depth, _lambda, time_limit,
             primal.model.getAttr("Status"), primal.model.getAttr("ObjVal"), train_acc,
             primal.model.getAttr("MIPGap") * 100, primal.model.getAttr("NodeCount"), solving_time,
             test_acc, calibration_acc,
             fairness_metrics_dict[('SP','train','Data','max_diff')][0], fairness_metrics_dict[('SP','train','Predictions','max_diff')][0],
             fairness_metrics_dict[('SP','test','Data','max_diff')][0], fairness_metrics_dict[('SP','test','Predictions','max_diff')][0],
             fairness_metrics_dict[('SP','calib','Data','max_diff')][0], fairness_metrics_dict[('SP','calib','Predictions','max_diff')][0],
             fairness_metrics_dict[('CSP','train','Data','max_diff')][0], fairness_metrics_dict[('CSP','train','Predictions','max_diff')][0],
             fairness_metrics_dict[('CSP','test','Data','max_diff')][0], fairness_metrics_dict[('CSP','test','Predictions','max_diff')][0],
             fairness_metrics_dict[('CSP','calib','Data','max_diff')][0], fairness_metrics_dict[('CSP','calib','Predictions','max_diff')][0],
             fairness_metrics_dict[('PE','train','Predictions','max_diff')][0],
             fairness_metrics_dict[('PE','test','Predictions','max_diff')][0],
             fairness_metrics_dict[('PE','calib','Predictions','max_diff')][0],
             fairness_metrics_dict[('EOpp','train','Predictions','max_diff')][0],
             fairness_metrics_dict[('EOpp','test','Predictions','max_diff')][0],
             fairness_metrics_dict[('EOpp','calib','Predictions','max_diff')][0],
             fairness_metrics_dict[('EOdds','train','Predictions','max_diff')][0],
             fairness_metrics_dict[('EOdds','test','Predictions','max_diff')][0],
             fairness_metrics_dict[('EOdds','calib','Predictions','max_diff')][0]])


if __name__ == "__main__":
    main(sys.argv[1:])
