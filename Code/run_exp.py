import FlowOCTReplication

depths = [1, 2, 3, 4, 5]
datasets = ["monk1.csv", 'monk1_enc.csv', 'monk2_enc.csv', 'monk3_enc.csv', 'car_evaluation_enc.csv',
            'balance-scale_enc.csv', 'kr-vs-kp_enc.csv', 'house-votes-84_enc.csv', 'tic-tac-toe_enc.csv',
            'breast-cancer_enc.csv', 'hayes-roth_enc.csv', 'spect_enc.csv', 'soybean-small_enc.csv',
            'compas.csv', 'compas_enc.csv','compas__short.csv','compas__short_enc.csv', 'compas_trial.csv',
            'compas_trial_enc.csv','compas_trial2.csv','compas_trial2_enc.csv','compas_trial3.csv','compas_trial3_enc.csv']
samples = [1, 2, 3, 4, 5]

FlowOCTReplication.main(["-r", 'compas.csv', "-f", 'compas_enc.csv', "-d", 2, "-t", 60, "-l", 0, "-i", 2, "-c", 1, "-a", 'PE', "-b", 0.2, "-e", 'race_factor', "-g", 2, "-h", 'priors_buckets'])


# for s in [2]:
#     for fairness_type in ["SP","None"]:
#         if fairness_type == "None":
#             FlowOCTReplication.main(["-r", "compas.csv", "-f", 'compas_enc.csv', "-d", 2, "-t", 2100, "-l", 0, "-i", s, "-c", 1, "-a", fairness_type, "-b", 0, "-e", 'race_factor', "-g", 2])
#         else:
#             for delta in [0.01, 0.05, 0.1 , 0.2, 0.3]:
#                 FlowOCTReplication.main(["-r", "compas.csv", "-f", 'compas_enc.csv', "-d", 2, "-t", 2100, "-l", 0, "-i", s, "-c", 1, "-a", fairness_type, "-b", delta, "-e", 'race_factor', "-g", 2])
