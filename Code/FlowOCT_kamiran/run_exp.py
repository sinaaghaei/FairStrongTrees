import FlowOCTReplication

timelimit = 10
calibration_mode = 1

dataset_dict = {('compas','protected_feature'):'race',
 ('compas','positive_class'):1,
 ('compas','deprived_group'):1,
 ('german','protected_feature'):'age',
 ('german','positive_class'):2,
 ('german','deprived_group'):1,
 ('adult','protected_feature'):'sex',
 ('adult','positive_class'):2,
 ('adult','deprived_group'):1,
 ('default','protected_feature'):'SEX',
 ('default','positive_class'):1,
 ('default','deprived_group'):2}

for data_set in ['adult']:
    for sample in [1]:
        if calibration_mode == 1:
                train_file_reg = f'{data_set}_train_calibration_{sample}.csv'
                train_file_enc = f'{data_set}_train_calibration_enc_{sample}.csv'
        else:
                train_file_reg = f'{data_set}_train_{sample}.csv'
                train_file_enc = f'{data_set}_train_enc_{sample}.csv'
        test_file_reg = f'{data_set}_test_{sample}.csv'
        test_file_enc = f'{data_set}_test_enc_{sample}.csv'
        calibration_file_reg = f'{data_set}_calibration_{sample}.csv'
        calibration_file_enc = f'{data_set}_calibration_enc_{sample}.csv'
        for depth in [1]:
                for l in [0]:
                        for fairness_type_bound in [('SP',0.1)]:#('None',1),('SP',0.1),('CSP',0.1),('PE',0.1),('EOpp',0.1),('EOdds',0.1)
                                FlowOCTReplication.main(["--train_file_reg", train_file_reg,"--train_file_enc", train_file_enc,
                                                         "--test_file_reg", test_file_reg,"--test_file_enc", test_file_enc,
                                                         "--calibration_file_reg", calibration_file_reg,"--calibration_file_enc", calibration_file_enc,
                                                         "--depth", depth, "--timelimit", timelimit, "-i", l,
                                                         "--fairness_type",fairness_type_bound[0], "--fairness_bound", fairness_type_bound[1],
                                                         "--protected_feature", dataset_dict[(data_set,'protected_feature')], "--positive_class", dataset_dict[(data_set,'positive_class')],"--conditional_feature", 'priors_count',
                                                         "--calibration_mode", calibration_mode, "--sample", 1, "--deprived", dataset_dict[(data_set,'deprived_group')]])
