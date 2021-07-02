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
# Constrants for Adult dataset
########################################################

# Select categorical attributes
cols = [
    'workclass',
    'education',
    'marital_status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native_country',
    'fnlwgt',
    'hours_per_week',
    'capital',
    'age_group'
] 

# Define the potential sensitive attributes and their vategorical values.
workclass = ['workclass_1','workclass_2','workclass_3','workclass_4']
education = ['education_1','education_2','education_3','education_4','education_5','education_6']
marital_status = ['marital_status_1','marital_status_2','marital_status_3','marital_status_4','marital_status_5']
occupation = ['occupation_1','occupation_2','occupation_3','occupation_4','occupation_5','occupation_6','occupation_7','occupation_8','occupation_9','occupation_10','occupation_11','occupation_12','occupation_13','occupation_14']
relationship = ['relationship_1','relationship_2','relationship_3','relationship_4','relationship_5','relationship_6']
race = ['race_1', 'race_2', 'race_3', 'race_4', 'race_5']
sex = ['sex_1','sex_2']          
native_country = ['native_country_1','native_country_2']
fnlwgt = ['fnlwgt_1','fnlwgt_2','fnlwgt_3','fnlwgt_4']
hours_per_week = ['hours_per_week_1','hours_per_week_2','hours_per_week_3','hours_per_week_4']
capital = ['capital_1','capital_2','capital_3']
age_group = ['age_group_1','age_group_2','age_group_3','age_group_4']

classes = (1,2)

attr_map = {
    'workclass' : workclass,
    'education' : education,
    'marital_status' : marital_status,
    'occupation' : occupation,
    'relationship' : relationship,
    'race' : race,
    'sex' : sex,
    'native_country' : native_country,
    'fnlwgt' : fnlwgt,
    'hours_per_week' : hours_per_week,
    'capital' : capital,
    'age_group' : age_group
}

# Some pre-defined refinement heuristics
refineHeuristics = [(workclass, True), (education, True),
                    (marital_status, True), (occupation, True),
                    (relationship, True), (race, True),
                    (sex, True), (native_country, True),
                    (fnlwgt, True), (hours_per_week, True),
                    (capital, True), (age_group, True)          
                    ]
########################################################

def parse_args():
    parser = eutil.create_base_parser(
        description='Patch adult dataset.',
        sensitive_attrs_default = "['sex']",
        dataset_default = 'adult_SMT.data',
        fairness_thresh_default = 0.8)
    args = parser.parse_args()
    evalu = eutil.EvalUtil(args)
    random.seed(args.random_seed)
    return evalu


if __name__ == '__main__':
    evalu = parse_args()
    patch.patch(evalu, cols, refineHeuristics, attr_map,classes)
    evalu.save_vals()
