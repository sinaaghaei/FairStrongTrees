import FlowOCTReplication

timelimit = 20
calibration_mode = 1

for data_set in ['compas']:
    for sample in [1]:
        if calibration_mode == 1:
                train_file_reg = f'compas_train_calibration_{sample}.csv'
                train_file_enc = f'compas_train_calibration_enc_{sample}.csv'
        else:
                train_file_reg = f'compas_train_{sample}.csv'
                train_file_enc = f'compas_train_enc_{sample}.csv'
        test_file_reg = f'compas_test_{sample}.csv'
        test_file_enc = f'compas_test_enc_{sample}.csv'
        calibration_file_reg = f'compas_calibration_{sample}.csv'
        calibration_file_enc = f'compas_calibration_enc_{sample}.csv'
        for depth in [2]:
                for l in [0]:
                        for fairness_type_bound in [('CSP',0.1)]:#('None',1),('SP',0.1),('CSP',0.1),('PE',0.1),('EOpp',0.1),('EOdds',0.1)
                                FlowOCTReplication.main(["--train_file_reg", train_file_reg,"--train_file_enc", train_file_enc,
                                                         "--test_file_reg", test_file_reg,"--test_file_enc", test_file_enc,
                                                         "--calibration_file_reg", calibration_file_reg,"--calibration_file_enc", calibration_file_enc,
                                                         "--depth", depth, "--timelimit", timelimit, "-i", l,
                                                         "--fairness_type",fairness_type_bound[0], "--fairness_bound", fairness_type_bound[1],
                                                         "--protected_feature", 'race', "--positive_class", 2,"--conditional_feature", 'priors_count',"--calibration_mode", calibration_mode, "--sample", 1])
