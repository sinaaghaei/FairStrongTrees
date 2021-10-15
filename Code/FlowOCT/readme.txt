
In order to run the code you simply need to pass the following parameters to the main function in FlowOCTReplication.py as follows:


"train_file_reg=" :: This is the name of the training file where the columns are not one-hot encoded
"train_file_enc=" :: This is the name of the training file where the columns are     one-hot encoded
"test_file_reg="  :: This is the name of the test file where the columns are not one-hot encoded
"test_file_enc="  :: This is the name of the test file where the columns are     one-hot encoded
"calibration_file_reg=" :: This is the name of the calibration file where the columns are not one-hot encoded
 "calibration_file_enc=" :: This is the name of the calibration file where the columns are  one-hot encoded
"depth=" :: depth of the tree
"timelimit=" :: Gurobi Timelimit for solving the MIO
 "-i=" ::  a parameter penalizing number of branching nodes in the objective
"fairness_type=" :: Type of fairness you want to enforce. Currennt;y the code support the following fairness types
statistical parity = 'SP'
conditional statistical parity = 'CSP'
Predictive equality = 'PE'
equal opportunity = 'EOpp'
equalized odds = 'EOdds'

please read the paper for the definition of these metrics.


 "fairness_bound=" :: the bound for the fairness constraint. should between [0,1]
"protected_feature=" :: the protected attribute, e.g, race
'positive_class=' :: the value of positive class in your data
"conditional_feature=" :: the conditional feature used in CSP definition
"calibration_mode=" :: whether you want to calibrate a parameter or not.



If you don't have a calibration dataset just pass the test data as the calibration. You shouldn't leave that empty.




import FlowOCTReplication
FlowOCTReplication.main(["--train_file_reg", 'compas_train_reg.csv',"--train_file_enc", 'compas_train_enc.csv',
                                                 "--test_file_reg", 'compas_test_reg.csv',"--test_file_enc", 'compas_test_enc.csv',
                                                 "--calibration_file_reg", 'compas_calibration_reg.csv',"--calibration_file_enc", 'compas_calibration_enc.csv',
                                                 "--depth", 2, "--timelimit", 600, "-i", 0,
                                                 "--fairness_type",'EOpp', "--fairness_bound", 0.1,
                                                 "--protected_feature", 'race', "--positive_class", 2,"--conditional_feature", 'priors_count',"--calibration_mode", 0])
