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


def main(argv):
    print(argv)
    input_file_reg = None
    input_file_enc = None
    depth = None
    time_limit = None
    _lambda = None
    input_sample = None
    calibration = None
    fairness_type = None
    fairness_bound = None
    protected_feature = None
    positive_class = None
    '''
    Depending on the value of input_sample we choose one of the following random seeds and then split the whole data
    into train, test and calibration
    '''
    random_states_list = [41, 23, 45, 36, 19, 123]

    try:
        opts, args = getopt.getopt(argv, "r:f:d:t:l:i:c:a:b:e:g:",
                                   ["input_file_reg=", "input_file_enc=", "depth=", "timelimit=", "lambda=",
                                    "input_sample=",
                                    "calibration=", "fairness_type=", "fairness_bound=","protected_feature=",
                                    'positive_class='])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-r", "--input_file_reg"):
            input_file_reg = arg
        elif opt in ("-f", "--input_file_enc"):
            input_file_enc = arg
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
        elif opt in ("-g", "--positive_class"):
            positive_class = int(arg)

    start_time = time.time()
    data_path = os.getcwd() + '/../DataSets/'
    data_reg = pd.read_csv(data_path + input_file_reg)
    data_enc = pd.read_csv(data_path + input_file_enc)
    '''Name of the column in the dataset representing the class label.
    In the datasets we have, we assume the label is target. Please change this value at your need'''
    label = 'target'

    # Tree structure: We create a tree object of depth d
    tree = Tree(depth)

    ##########################################################
    # output setup
    ##########################################################
    approach_name = 'FlowOCT'
    out_put_name = input_file_enc + '_' + str(input_sample) + '_' + approach_name + '_d_' + str(depth) + '_t_' + str(
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
    data_train_enc, data_test_enc = train_test_split(data_enc, test_size=0.25, random_state=random_states_list[input_sample - 1])
    data_train_calibration_enc, data_calibration_enc = train_test_split(data_train_enc, test_size=0.33,
                                                                random_state=random_states_list[input_sample - 1])

    data_train_reg, data_test_reg = train_test_split(data_reg, test_size=0.25, random_state=random_states_list[input_sample - 1])
    data_train_calibration_reg, data_calibration_reg = train_test_split(data_train_reg, test_size=0.33,
                                                                random_state=random_states_list[input_sample - 1])

    if calibration == 1:  # in this mode, we train on 50% of the data; otherwise we train on 75% of the data
        data_train_enc = data_train_calibration_enc
        data_train_reg = data_train_calibration_reg

    train_len = len(data_train_enc.index)
    ##########################################################
    # Creating and Solving the problem
    ##########################################################
    # We create the MIP problem by passing the required arguments
    primal = FlowOCT(data_train_enc, data_train_reg, label, tree, _lambda, time_limit, fairness_type, fairness_bound, protected_feature, positive_class)

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

    train_acc = get_acc(primal, data_train_enc, b_value, beta_value, p_value)
    test_acc = get_acc(primal, data_test_enc, b_value, beta_value, p_value)
    calibration_acc = get_acc(primal, data_calibration_enc, b_value, beta_value, p_value)

    print("obj value", primal.model.getAttr("ObjVal"))
    print("train acc", train_acc)
    print("test acc", test_acc)
    print("calibration acc", calibration_acc)

    # For loop p pprime
    # Print the string of p pprime
    # Call get_SP

    # Get statistical parity for all different combinations of two groups from protected feature
    if fairness_type == "SP" or fairness_type == "None":

        # Define variables for maximum statistical parity for each dataset
        max_sp_train_data = 0
        max_sp_train_pred = 0
        max_sp_test_data = 0
        max_sp_test_pred = 0
        max_sp_calib_data = 0
        max_sp_calib_pred = 0

        # Loop through all possible combinations of the protected feature
        for combos in combinations(data_reg[protected_feature].unique(), 2):
            protection = combos[0]
            protection_prime = combos[1]

            # Print results
            # print(str(protection) + " and " + str(protection_prime) + " Statistical Parity:")

            # Let's construct the max SP for train, test, and calibration
            # We use an if statement to determine if there is a higher maximum
            sp_train_data = get_sp(primal, data_train_enc, data_train_reg, b_value, beta_value, p_value, protection, protection_prime, positive_class, 'Data')
            if sp_train_data >= max_sp_train_data:
                max_sp_train_data = sp_train_data
                max_sp_train_data_protection = protection
                max_sp_train_data_protection_prime = protection_prime

            sp_train_pred = get_sp(primal, data_train_enc, data_train_reg, b_value, beta_value, p_value, protection, protection_prime, positive_class, 'Predictions')
            if sp_train_pred >= max_sp_train_pred:
                max_sp_train_pred = sp_train_pred
                max_sp_train_pred_protection = protection
                max_sp_train_pred_protection_prime = protection_prime

            sp_test_data = get_sp(primal, data_test_enc, data_test_reg, b_value, beta_value, p_value, protection, protection_prime, positive_class, 'Data')
            if sp_test_data >= max_sp_test_data:
                max_sp_test_data = sp_test_data
                max_sp_test_data_protection = protection
                max_sp_test_data_protection_prime = protection_prime

            sp_test_pred = get_sp(primal, data_test_enc, data_test_reg, b_value, beta_value, p_value, protection, protection_prime, positive_class, 'Predictions')
            if sp_test_pred >= max_sp_test_pred:
                max_sp_test_pred = sp_test_pred
                max_sp_test_pred_protection = protection
                max_sp_test_pred_protection_prime = protection_prime

            sp_calib_data = get_sp(primal, data_calibration_enc, data_calibration_reg, b_value, beta_value, p_value, protection, protection_prime, positive_class, 'Data')
            if sp_calib_data >= max_sp_calib_data:
               max_sp_calib_data = sp_calib_data
               max_sp_calib_data_protection = protection
               max_sp_calib_data_protection_prime = protection_prime

            sp_calib_pred = get_sp(primal, data_calibration_enc, data_calibration_reg, b_value, beta_value, p_value, protection, protection_prime, positive_class, 'Predictions')
            if sp_calib_pred >= max_sp_calib_pred:
               max_sp_calib_pred = sp_calib_pred
               max_sp_calib_pred_protection = protection
               max_sp_calib_pred_protection_prime = protection_prime

        # Print all maximum values with the corresponding protected varibles 
        print(str(max_sp_train_data_protection) + " & " + str(max_sp_train_data_protection_prime) + " has train data SP: " + str(max_sp_train_data))
        print(str(max_sp_train_pred_protection) + " & " + str(max_sp_train_pred_protection_prime) + " has train pred SP: " + str(max_sp_train_pred))
        print(str(max_sp_test_data_protection) + " & " + str(max_sp_test_data_protection_prime) + " has test data SP: " + str(max_sp_test_data))
        print(str(max_sp_test_pred_protection) + " & " + str(max_sp_test_pred_protection_prime) + " has test pred SP: " + str(max_sp_test_pred))
        print(str(max_sp_calib_data_protection) + " & " + str(max_sp_calib_data_protection_prime) + " has calibration data SP: " + str(max_sp_calib_data))
        print(str(max_sp_calib_pred_protection) + " & " + str(max_sp_calib_pred_protection_prime) + " has calibration pred SP: " + str(max_sp_calib_pred))

    ##########################################################
    # writing info to the file
    ##########################################################
    primal.model.write(out_put_path + out_put_name + '.lp')
    # writing info to the file
    result_file = out_put_name + '.csv'
    with open(out_put_path + result_file, mode='a') as results:
        results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        results_writer.writerow(
            [approach_name, input_file_enc, fairness_type, fairness_bound, train_len, depth, _lambda, time_limit,
             primal.model.getAttr("Status"), primal.model.getAttr("ObjVal"), train_acc,
             primal.model.getAttr("MIPGap") * 100, primal.model.getAttr("NodeCount"), solving_time,
             primal.model._total_callback_time_integer, primal.model._total_callback_time_integer_success,
             primal.model._callback_counter_integer, primal.model._callback_counter_integer_success,
             test_acc, calibration_acc, input_sample, max_sp_train_data, max_sp_train_pred,max_sp_test_data,max_sp_test_pred,max_sp_calib_data,max_sp_calib_pred])

if __name__ == "__main__":
    main(sys.argv[1:])
