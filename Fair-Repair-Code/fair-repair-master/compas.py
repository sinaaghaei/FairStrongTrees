import pandas as pd
import numpy as np
import itertools
import argparse
import os.path
import random
import time

import eutil
import patch

########################################################
# Constrants for Compas dataset
########################################################

# Select categorical attributes
cols = [
    'race',
    'age_cat',
    'sex',
    # 'priors_count',
    'c_charge_degree',
    'length_of_stay',
    'score_text',
    # 'Class'
] 

# Get all attributes, continuous and discrete.
race = ['race_1','race_2','race_3','race_4']
age_cat = ['age_cat_1','age_cat_2','age_cat_3']
sex = ['sex_1','sex_2']
priors_count = 'priors_count'
length_of_stay = ['length_of_stay_1','length_of_stay_2','length_of_stay_3','length_of_stay_4','length_of_stay_5']
c_charge_degree = ['c_charge_degree_1','c_charge_degree_2']
score_text = ['score_text_1','score_text_2','score_text_3']

classes = (1,2)

# Gather all attributes into a map
attr_map = {
    'race' : race ,
    'age_cat': age_cat ,
    'sex': sex ,
    'priors_count': priors_count ,
    'length_of_stay': length_of_stay ,
    'c_charge_degree': c_charge_degree ,
    'score_text': score_text
}



# Some pre-defined refinement heuristics
refineHeuristics = [(race, True), (age_cat, True), (sex, True), 
                    (priors_count, False), (length_of_stay, True),
                    (c_charge_degree, True), (score_text, True),
                    ]
########################################################

def parse_args():
    parser = eutil.create_base_parser(
        description='Patch Compas dataset.',
        sensitive_attrs_default = "['race']",

        #####
        dataset_default = 'compas_train_1.csv',
        dataset_test_default = 'compas_test_1.csv',
        fairness_thresh_default = 0.8)
        ####
    args = parser.parse_args()
    evalu = eutil.EvalUtil(args)
    random.seed(args.random_seed)
    return evalu


if __name__ == '__main__':
    evalu = parse_args()
    patch.patch(evalu, cols, refineHeuristics, attr_map,classes)
    evalu.save_vals()
