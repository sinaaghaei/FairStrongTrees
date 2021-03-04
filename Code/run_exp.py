import FlowOCTReplication

depths = [1, 2, 3, 4, 5]
datasets = ["monk1.csv", 'monk1_enc.csv', 'monk2_enc.csv', 'monk3_enc.csv', 'car_evaluation_enc.csv',
            'balance-scale_enc.csv', 'kr-vs-kp_enc.csv', 'house-votes-84_enc.csv', 'tic-tac-toe_enc.csv',
            'breast-cancer_enc.csv', 'hayes-roth_enc.csv', 'spect_enc.csv', 'soybean-small_enc.csv',
            'compas.csv', 'compas_enc.csv']
samples = [1, 2, 3, 4, 5]

FlowOCTReplication.main(["-r", "compas.csv", "-f", 'compas_enc.csv', "-d", 2, "-t", 3600, "-l", 0, "-i", 1, "-c", 1, "-a", 'SP', "-b", 0.01, "-e", 'race_factor'])
# BendersOCTReplication.main(["-f", 'monk1_enc.csv', "-d", 2, "-t", 3600, "-l", 0.8, "-i", 1, "-c", 1])
