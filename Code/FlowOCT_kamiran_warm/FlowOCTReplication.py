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



def warm_train(data_train_enc, data_train_reg, label, depth, _lambda, time_limit, fairness_type, fairness_bound, protected_feature, positive_class, deprived_group, b_warm):
    tree = Tree(depth)
    primal = FlowOCT(data_train_enc, data_train_reg, label, tree, _lambda, time_limit, fairness_type, fairness_bound, protected_feature, positive_class, deprived_group, b_warm)
    primal.create_primal_problem()
    start_time = time.time()
    primal.model.update()
    primal.model.optimize()
    end_time = time.time()
    solving_time = end_time - start_time

    return solving_time, primal


def warm_evaluation(primal, solving_time, data_train_enc, data_test_enc, data_calibration_enc,data_train_reg, data_test_reg, data_calibration_reg, deprived_group, positive_class, protected_feature):
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
        fairness_metrics_dict = {}
        # fairness_metrics_dict[('SP','train','pred')] = sp_value
        def getFairnessResults(fairness_const_type):
            var_func = get_sp

            for data_set in ['train','test','calib']:
                data_set_enc, data_set_reg = data_dict[data_set]
                for source in ['Data','Predictions']:
                    sp_value = var_func(primal, data_set_enc, data_set_reg, b_value, beta_value, p_value, deprived_group, positive_class, source)
                    fairness_metrics_dict[(fairness_const_type,data_set,source)] = sp_value


        # Print all maximum values with the corresponding protected varibles
        for fairness_const_type in ['SP']:
            getFairnessResults(fairness_const_type)
            print('###################{} Results'.format(fairness_const_type))
            for data_set in ['train','test','calib']:
                for source in ['Data','Predictions']:
                    sp_value = fairness_metrics_dict[(fairness_const_type,data_set,source)]
                    print('{} {} {}: {} '.format(data_set,source,fairness_const_type, sp_value))



        warm_results = [primal.model.getAttr("Status"), primal.model.getAttr("ObjVal"), train_acc,
         primal.model.getAttr("MIPGap") * 100, primal.model.getAttr("NodeCount"), solving_time,
         test_acc, calibration_acc,
         fairness_metrics_dict[('SP','train','Data')], fairness_metrics_dict[('SP','train','Predictions')],
         fairness_metrics_dict[('SP','test','Data')], fairness_metrics_dict[('SP','test','Predictions')],
         fairness_metrics_dict[('SP','calib','Data')], fairness_metrics_dict[('SP','calib','Predictions')]]

        return warm_results



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
    deprived_group = None
    warm_start_depth = 1
    '''
    Depending on the value of input_sample we choose one of the following random seeds and then split the whole data
    into train, test and calibration
    '''
    random_states_list = [41, 23, 45, 36, 19, 123]

    try:
        opts, args = getopt.getopt(argv,'a:b:c:d:e:f:g:h:i:j:k:l:m:n:o:p:q:r:',
                                   ["train_file_reg=", "train_file_enc=",
                                   "test_file_reg=", "test_file_enc=",
                                    "calibration_file_reg=", "calibration_file_enc=",
                                    "depth=", "timelimit=", "_lambda=",
                                    "fairness_type=", "fairness_bound=","protected_feature=",
                                    'positive_class=', "conditional_feature=","calibration_mode=","sample=","deprived=","warm_start_depth="])
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
            calibration_mode = int(arg)
        elif opt in ('-p',"--sample"):
            input_sample = int(arg)
        elif opt in ('-q',"--deprived"):
            deprived_group = int(arg)
        elif opt in ('-r',"--warm_start_depth"):
            warm_start_depth = int(arg)

    data_path = os.getcwd() + '/../../DataSets/KamiranVersion/'

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


    ##########################################################
    # output setup
    ##########################################################
    approach_name = 'FlowOCT_kamiran_warm'
    out_put_name = f'{train_file_reg}_{approach_name}_d_{depth}_warmStartDepth_{warm_start_depth}_t_{time_limit}_lambda_{_lambda}_ft_{fairness_type}_fb_{fairness_bound}_calibration_{calibration_mode}'
    out_put_path = os.getcwd() + '/../../Results/'
    # Using logger we log the output of the console in a text file
    sys.stdout = logger(out_put_path + out_put_name + '.txt')

    ##########################################################
    # Creating and Solving the problem
    ##########################################################
    # We create the MIP problem by passing the required arguments
    # if depth <=2:
    #     solving_time, primal = my_train(data_train_enc, data_train_reg, label, depth, _lambda, time_limit, fairness_type, fairness_bound, protected_feature, positive_class, deprived_group, None)
    # else:
    #     b_warm = None
    #     solving_time = 0
    #     for warm_depth in range(2,depth+1):
    #         solving_time_warm, primal = my_train(data_train_enc, data_train_reg, label, warm_depth, _lambda, time_limit, fairness_type, fairness_bound, protected_feature, positive_class, deprived_group, b_warm)
    #         solving_time += solving_time_warm
    #         b_warm = primal.model.getAttr("X", primal.b)

    # No warm start
    # solving_time, primal = my_train(data_train_enc, data_train_reg, label, depth, _lambda, time_limit, fairness_type, fairness_bound, protected_feature, positive_class, deprived_group, None)

    #with warm start
    Results_dic = {}
    b_warm = None
    time_limit_warm = time_limit
    solving_time = 0
    for warm_depth in range(warm_start_depth,depth+1):
        print(f'######################################################## Intermediate depth = {warm_depth}')
        solving_time_warm, primal = warm_train(data_train_enc, data_train_reg, label, warm_depth, _lambda, time_limit_warm, fairness_type, fairness_bound, protected_feature, positive_class, deprived_group, b_warm)
        solving_time += solving_time_warm
        b_warm = primal.model.getAttr("X", primal.b)
        warm_results = warm_evaluation(primal, solving_time, data_train_enc, data_test_enc, data_calibration_enc,data_train_reg, data_test_reg, data_calibration_reg, deprived_group, positive_class, protected_feature)
        Results_dic[warm_depth] = [approach_name, train_file_reg, input_sample, fairness_type, fairness_bound, train_len, calibration_mode, warm_depth, _lambda, time_limit_warm] + warm_results
        time_limit_warm = max(time_limit - solving_time,100)
    ##########################################################
    # writing info to the file
    ##########################################################
    primal.model.write(out_put_path + out_put_name + '.lp')
    # writing info to the file
    result_file = out_put_name + '.csv'
    with open(out_put_path + result_file, mode='a') as results:
        results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        for r in Results_dic.values():
                results_writer.writerow(r)


if __name__ == "__main__":
    main(sys.argv[1:])
