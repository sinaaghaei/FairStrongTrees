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


def main(argv):
    print(argv)
    input_file = None
    depth = None
    time_limit = None
    _lambda = None
    input_sample = None
    calibration = None
    fairness_type = None
    fairness_bound = None
    protected_feature = None
    '''
    Depending on the value of input_sample we choose one of the following random seeds and then split the whole data
    into train, test and calibration
    '''
    random_states_list = [41, 23, 45, 36, 19, 123]

    try:
        opts, args = getopt.getopt(argv, "f:d:t:l:i:c:a:b:e:",
                                   ["input_file=", "depth=", "timelimit=", "lambda=",
                                    "input_sample=",
                                    "calibration=", "fairness_type=", "fairness_bound=","protected_feature="])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-f", "--input_file"):
            input_file = arg
        elif opt in ("-d", "--depth"):
            depth = int(arg)
        elif opt in ("-t", "--timelimit"):
            time_limit = int(arg)
        elif opt in ("-l", "--lambda"):
            _lambda = float(arg)
        elif opt in ("-i", "--input_sample"):
            input_sample = int(arg)
        elif opt in ("-c", "--calibration"):
            calibration = int(arg)
        elif opt in ("-a", "--fairness_type"):
            fairness_type = arg
        elif opt in ("-b", "--fairness_bound"):
            fairness_bound = float(arg)
        elif opt in ("-e", "--protected_feature"):
            protected_feature = arg

    start_time = time.time()
    data_path = os.getcwd() + '/../DataSets/'
    data = pd.read_csv(data_path + input_file)
    '''Name of the column in the dataset representing the class label.
    In the datasets we have, we assume the label is target. Please change this value at your need'''
    label = 'target'

    # Tree structure: We create a tree object of depth d
    tree = Tree(depth)

    ##########################################################
    # output setup
    ##########################################################
    approach_name = 'FlowOCT'
    out_put_name = input_file + '_' + str(input_sample) + '_' + approach_name + '_d_' + str(depth) + '_t_' + str(
        time_limit) + '_lambda_' + str(
        _lambda) + '_c_' + str(calibration)
    out_put_path = os.getcwd() + '/../Results/'
    # Using logger we log the output of the console in a text file
    sys.stdout = logger(out_put_path + out_put_name + '.txt')

    ##########################################################
    # data splitting
    ##########################################################
    '''
    Creating  train, test and calibration datasets
    We take 50% of the whole data as training, 25% as test and 25% as calibration

    When we want to calibrate _lambda, for a given value of _lambda we train the model on train and evaluate
    the accuracy on calibration set and at the end we pick the _lambda with the highest accuracy.

    When we got the calibrated _lambda, we train the mode on (train+calibration) which we refer to it as
    data_train_calibration and evaluate the accuracy on (test)

    '''
    data_train, data_test = train_test_split(data, test_size=0.25, random_state=random_states_list[input_sample - 1])
    data_train_calibration, data_calibration = train_test_split(data_train, test_size=0.33,
                                                                random_state=random_states_list[input_sample - 1])

    if calibration == 1:  # in this mode, we train on 50% of the data; otherwise we train on 75% of the data
        data_train = data_train_calibration

    train_len = len(data_train.index)
    ##########################################################
    # Creating and Solving the problem
    ##########################################################
    # We create the MIP problem by passing the required arguments
    primal = FlowOCT(data_train, label, tree, _lambda, time_limit, fairness_type, fairness_bound, protected_feature)

    primal.create_primal_problem()
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

    print("\n\n")
    print_tree(primal,b_value, beta_value, p_value)

    print('\n\nTotal Solving Time', solving_time)
    print("obj value", primal.model.getAttr("ObjVal"))

    print('Total Callback counter (Integer)', primal.model._callback_counter_integer)
    print('Total Successful Callback counter (Integer)', primal.model._callback_counter_integer_success)

    print('Total Callback Time (Integer)', primal.model._total_callback_time_integer)
    print('Total Successful Callback Time (Integer)', primal.model._total_callback_time_integer_success)


    # print(b_value)
    # print(p_value)
    # print(beta_value)
    ##########################################################
    # Evaluation
    ##########################################################
    '''
    For classification we report accuracy
    For regression we report MAE (Mean Absolute Error) , MSE (Mean Squared Error) and  R-squared

    over training, test and the calibration set
    '''
    train_acc = test_acc = calibration_acc = 0
    train_mae = test_mae = calibration_mae = 0
    train_r_squared = test_r_squared = calibration_r_squared = 0

    train_acc = get_acc(primal, data_train, b_value, beta_value, p_value)
    test_acc = get_acc(primal, data_test, b_value, beta_value, p_value)
    calibration_acc = get_acc(primal, data_calibration, b_value, beta_value, p_value)

    print("obj value", primal.model.getAttr("ObjVal"))
    print("train acc", train_acc)
    print("test acc", test_acc)
    print("calibration acc", calibration_acc)

    # For loop p pprime
    # Print the string of p pprime
    # Call get_SP

    # Get statistical parity for all different combinations of two groups from protected feature
    check = {}
    for protection in data[protected_feature].unique():
        for protection_prime in data[protected_feature].unique():

            # We do not care about the statistical parity between the same group since this = 0
            if protection == protection_prime or (protection, protection_prime) in check:
                continue
            check[(protection, protection_prime)] = 1
            check[(protection_prime, protection)] = 1

            # Print results
            print(str(protection) + " and " + str(protection_prime) + " Statistical Parity:")
            print(get_sp(primal, data, b_value, beta_value, p_value, protection, protection_prime, 'Data'))

    ##########################################################
    # writing info to the file
    ##########################################################
    primal.model.write(out_put_path + out_put_name + '.lp')
    # writing info to the file
    result_file = out_put_name + '.csv'
    with open(out_put_path + result_file, mode='a') as results:
        results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        results_writer.writerow(
            [approach_name, input_file, train_len, depth, _lambda, time_limit,
             primal.model.getAttr("Status"), primal.model.getAttr("ObjVal"), train_acc,
             primal.model.getAttr("MIPGap") * 100, primal.model.getAttr("NodeCount"), solving_time,
             primal.model._total_callback_time_integer, primal.model._total_callback_time_integer_success,
             primal.model._callback_counter_integer, primal.model._callback_counter_integer_success,
             test_acc, calibration_acc, input_sample])

if __name__ == "__main__":
    main(sys.argv[1:])
