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
# Constrants for German dataset
########################################################

# Select categorical attributes
cols = [
	'chek_acc',
    'month_duration',
    'credit_history',
    'purpose',
    'Credit_amo',
    'saving_amo',
    'present_employmment',
    'instalrate',
    'p_status',
    'guatan',
    'present_resident',
    'property',
    'age',
    'installment',
    'Housing',
    'existing_cards',
    'job',
    'no_people',
    'telephn',
    'foreign_worker',
] 

# Get all attributes, continuous and discrete.
chek_acc = ["chek_acc_1","chek_acc_2","chek_acc_3","chek_acc_4"]   
month_duration =  ["month_duration_1","month_duration_2","month_duration_3","month_duration_4"]  
credit_history =  ["credit_history_1","credit_history_2","credit_history_3","credit_history_4","credit_history_5"]  
purpose = ["purpose_1","purpose_2","purpose_3","purpose_4","purpose_5","purpose_6","purpose_7","purpose_8","purpose_9","purpose_10"] 
Credit_amo = ["Credit_amo_1","Credit_amo_2","Credit_amo_3","Credit_amo_4"]  
saving_amo = ["saving_amo_1","saving_amo_2","saving_amo_3","saving_amo_4","saving_amo_5"]
present_employmment = ["present_employmment_1","present_employmment_2","present_employmment_3","present_employmment_4","present_employmment_5"]
instalrate = ["instalrate_1","instalrate_2","instalrate_3"]
p_status = ["p_status_1","p_status_2","p_status_3","p_status_4"]
guatan = ["guatan_1","guatan_2","guatan_3"]
present_resident = ["present_resident_1","present_resident_2","present_resident_3"]
property = ["property_1","property_2","property_3","property_4"]
age = ["age_1","age_2","age_3","age_4"]
installment = ["installment_1","installment_2","installment_3"]
Housing = ["Housing_1","Housing_2","Housing_3"]
existing_cards = ["existing_cards_1","existing_cards_2","existing_cards_3","existing_cards_4"]
job = ["job_1","job_2","job_3","job_4"]
no_people = ["no_people_1","no_people_2"]
telephn = ["telephn_1","telephn_2"]
foreign_worker = ["foreign_worker_1","foreign_worker_2"]

classes = (1,2)

# Gather all attributes into a map
attr_map = {
    'chek_acc': chek_acc ,
    'month_duration': month_duration ,
    'credit_history': credit_history ,
    'purpose': purpose ,
    'Credit_amo': Credit_amo ,
    'saving_amo': saving_amo ,
    'present_employmment': present_employmment ,
    'instalrate' : instalrate,
    'p_status' : p_status,
    'guatan' : guatan,
    'present_resident' : present_resident,
    'property' : property,
    'age' : age,
    'installment' : installment,
    'Housing' : Housing,
    'existing_cards' : existing_cards,
    'job' : job,
    'no_people' : no_people,
    'telephn' : telephn,
    'foreign_worker' : foreign_worker,
}

# Some pre-defined refinement heuristics
refineHeuristics = [(chek_acc, True), (month_duration, True), (credit_history, True), 
                    (purpose, True), (Credit_amo, True),
                    (saving_amo, True), (present_employmment, True),
                    (instalrate, True), (p_status, True), (guatan, True),
                    (present_resident, True), (property, True), (age, True),
                    (installment, True), (Housing, True), (existing_cards, True),
                    (job, True), (no_people, True), (telephn, True),
                    (foreign_worker, True)
                    ]
########################################################

def parse_args():
    parser = eutil.create_base_parser(
        description='Patch German dataset.',
        sensitive_attrs_default = "['age']",
        dataset_default = 'german_SMT.data',
        fairness_thresh_default = 0.8)
        #-
    args = parser.parse_args()
    evalu = eutil.EvalUtil(args)
    random.seed(args.random_seed)
    return evalu


if __name__ == '__main__':
    evalu = parse_args()
    patch.patch(evalu, cols, refineHeuristics, attr_map,classes)
    evalu.save_vals()
