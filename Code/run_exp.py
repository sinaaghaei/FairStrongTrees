import FlowOCTReplication

depths = [1, 2, 3, 4, 5]
datasets = ["monk1.csv", 'monk1_enc.csv', 'monk2_enc.csv', 'monk3_enc.csv', 'car_evaluation_enc.csv',
            'balance-scale_enc.csv', 'kr-vs-kp_enc.csv', 'house-votes-84_enc.csv', 'tic-tac-toe_enc.csv',
            'breast-cancer_enc.csv', 'hayes-roth_enc.csv', 'spect_enc.csv', 'soybean-small_enc.csv',
            'compas.csv', 'compas_enc.csv','compas__short.csv','compas__short_enc.csv']
samples = [1, 2, 3, 4, 5]

FlowOCTReplication.main(["-r", "compas__short.csv", "-f", 'compas__short_enc.csv', "-d", 1, "-t", 60, "-l", 0, "-i", 1, "-c", 1, "-a", 'SP', "-b", 1/50, "-e", 'race_factor', "-g", 2])

# for s in samples:
#     for fairness_type in ["SP","None"]:
#         if fairness_type == "None":
#             FlowOCTReplication.main(["-r", "compas.csv", "-f", 'compas_enc.csv', "-d", 2, "-t", 600, "-l", 0, "-i", s, "-c", 1, "-a", fairness_type, "-b", 0, "-e", 'race_factor', "-g", 2])
#         else:
#             for delta in [0.1,0.2,0.3]:
#                 FlowOCTReplication.main(["-r", "compas.csv", "-f", 'compas_enc.csv', "-d", 2, "-t", 600, "-l", 0, "-i", s, "-c", 1, "-a", fairness_type, "-b", delta, "-e", 'race_factor', "-g", 2])
