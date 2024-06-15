from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds, FalsePositiveRateParity, TruePositiveRateParity
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.preprocessing import CorrelationRemover
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, true_positive_rate_difference, false_positive_rate_difference, equalized_odds_difference


def corr_remover(classifier, metric, X_cols, y_col, l_col, df_train, df_test, alpha, seed):
    l_train = df_train[l_col]
    X_train = df_train[X_cols]
    y_train = df_train[y_col]

    l_test = df_test[l_col]
    X_test = df_test[X_cols]
    y_test = df_test[y_col]
    
    np.random.seed(seed)
    
    c = CorrelationRemover(sensitive_feature_ids=[l_col], alpha=alpha)
    c.fit(X_train, y_train)
    X_train_trans = pd.DataFrame(c.transform(X_train), columns=X_cols[1:])
    X_test_trans = pd.DataFrame(c.transform(X_test), columns=X_cols[1:])
    
    classifier.fit(X_train_trans, y_train)
    y_test_pred = classifier.predict(X_test_trans)
    
    return accuracy_score(y_test, y_test_pred), metric(y_true=y_test, y_pred=y_test_pred, sensitive_features=l_test)

def run_exp_gradient(classifier, constraint, metric, X_cols, y_col, l_col, df_train, df_test, eps, seed):
    l_train = df_train[l_col]
    X_train = df_train[X_cols]
    y_train = df_train[y_col]

    l_test = df_test[l_col]
    X_test = df_test[X_cols]
    y_test = df_test[y_col]
    
    np.random.seed(seed)
    
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    
    mitigator = ExponentiatedGradient(classifier, constraint, eps=eps)
    mitigator.fit(X_train, y_train, sensitive_features=l_train)
    
    y_pred_mitigated = mitigator.predict(X_test)
    
    return accuracy_score(y_test, y_pred_mitigated), metric(y_true=y_test, y_pred=y_pred_mitigated, sensitive_features=l_test)
    
def run_threshold(classifier, constraint, metric, prefit, X_cols, y_col, l_col, df_train, df_test, seed):
    l_train = df_train[l_col]
    X_train = df_train[X_cols]
    y_train = df_train[y_col]

    l_test = df_test[l_col]
    X_test = df_test[X_cols]
    y_test = df_test[y_col]
    
    np.random.seed(seed)
    
    classifier.fit(X_train, y_train)
    postprocess_est = ThresholdOptimizer(
                   estimator=classifier,
                   constraints=constraint,
                   objective="accuracy_score",
                   prefit=prefit,
                   predict_method='predict')
    postprocess_est.fit(X_train, y_train, sensitive_features=l_train)
    
    y_pred_mitigated = postprocess_est.predict(X_test, sensitive_features=l_test)
    
    return accuracy_score(y_test, y_pred_mitigated), metric(y_true=y_test, y_pred=y_pred_mitigated, sensitive_features=l_test)
    
dataset = 'german_binary' # 'compas', 'adult'
splits = [1, 2, 3, 4, 5]

if dataset == 'compas':
    X_cols = ['race', 'age_cat', 'sex', 'priors_count', 'c_charge_degree', 'length_of_stay']
    y_col = 'target'
    l_col = 'race'
elif dataset == 'adult':
    X_cols = ['workclass', 'education', 'marital_status', 'occupation',
       'relationship', 'race', 'native_country', 'fnlwgt', 'sex',
       'hours_per_week', 'capital', 'age_group']
    y_col = 'target'
    l_col = 'sex'
elif dataset == 'german_binary':
    X_cols = ['chek_acc', 'month_duration', 'credit_history', 'purpose',
       'Credit_amo', 'saving_amo', 'present_employmment', 'instalrate',
       'p_status', 'guatan', 'present_resident', 'property', 'age',
       'installment', 'Housing', 'existing_cards', 'job', 'no_people',
       'telephn', 'foreign_worker']
    y_col = 'target'
    l_col = 'age'

classifiers = {'RF': RandomForestClassifier(), 'LR': LogisticRegression(), 'DT': DecisionTreeClassifier(),
              'SVM': SVC(), 'MLP': MLPClassifier(), 'KNN': KNeighborsClassifier()}
metrics = {'DP': demographic_parity_difference, 'TPR': true_positive_rate_difference,
          'FPR': false_positive_rate_difference, 'EO': equalized_odds_difference}

df = pd.DataFrame(columns=['dataset', 'split', 'method', 'seed', 'classifier', 'metric', 'acc', 'disp', 'hyp'])


# run correlation remover
alphas = np.linspace(0, 1, 10)
seeds = [1, 100, 9, 50, 1111]
for split in tqdm(splits, desc=' splits', position=0):
    df_train = pd.read_csv(f'../DataSets/{dataset}_train_{split}.csv')
    df_test = pd.read_csv(f'../DataSets/{dataset}_test_{split}.csv')
    df_train['target'] -= 1
    df_test['target'] -= 1
    for seed in tqdm(seeds, desc=' seeds', position=1, leave=False):
        for cl, classifier in classifiers.items():
            for met, metric in metrics.items():
                for alpha in alphas:
                    acc, disp = corr_remover(classifier, metric, X_cols, y_col, l_col, df_train, df_test, alpha, seed)
                    df.loc[len(df)] = [dataset, split, 'correlation_remover', seed, cl, met, acc, disp, alpha]


metrics_constraints = {'DP': [demographic_parity_difference, 'demographic_parity'], 
                      'TPR': [true_positive_rate_difference, 'true_positive_rate_parity'],
                      'FPR': [false_positive_rate_difference, 'false_positive_rate_parity'], 
                      'EO': [equalized_odds_difference, 'equalized_odds']}

# run threshold optimizer
seeds = [1, 100, 9, 50, 1111]
for split in tqdm(splits, desc=' splits', position=0):
# for split in splits:
    df_train = pd.read_csv(f'../DataSets/{dataset}_train_{split}.csv')
    df_test = pd.read_csv(f'../DataSets/{dataset}_test_{split}.csv')
    df_train['target'] -= 1
    df_test['target'] -= 1
    for seed in tqdm(seeds, desc=' seeds', position=1, leave=False):
        for cl, classifier in classifiers.items():
            for met, metric in metrics_constraints.items():
                for prefit, suffix in zip([True, False], ['prefit', 'no_prefit']):
                    acc, disp = run_threshold(classifier, metric[1], metric[0], prefit, X_cols, y_col, l_col, df_train, df_test, seed)
                    df.loc[len(df)] = [dataset, split, f'threshold_optimizer', seed, cl, met, acc, disp, suffix]


metrics_constraints = {'DP': [demographic_parity_difference, DemographicParity()], 
                      'TPR': [true_positive_rate_difference, TruePositiveRateParity()],
                      'FPR': [false_positive_rate_difference, FalsePositiveRateParity()], 
                      'EO': [equalized_odds_difference, EqualizedOdds()]}

classifiers = {'RF': RandomForestClassifier(), 'LR': LogisticRegression(), 'DT': DecisionTreeClassifier(),}
#               'SVM': SVC()}

# run exponentiated gradient
seeds = [100, 9, 50, 1111]
epsilons = np.linspace(0.001, 0.1, 10)
for split in tqdm(splits[2:], desc=' splits', position=0):
    df_train = pd.read_csv(f'../DataSets/{dataset}_train_{split}.csv')
    df_test = pd.read_csv(f'../DataSets/{dataset}_test_{split}.csv')
    df_train['target'] -= 1
    df_test['target'] -= 1
    for seed in tqdm(seeds, desc=' seeds', position=1, leave=False):
        for cl, classifier in tqdm(classifiers.items(), desc=' classifiers', position=2, leave=False):
            for met, metric in metrics_constraints.items():
                for eps in epsilons:
                    if met == 'DP':
                        constraint = DemographicParity()
                    elif met == 'TPR':
                        constraint = TruePositiveRateParity()
                    elif met == 'FPR':
                        constraint = FalsePositiveRateParity()
                    elif met == 'EO':
                        constraint = EqualizedOdds()
                    
                    acc, disp = run_exp_gradient(classifier, constraint, metric[0], X_cols, y_col, l_col, df_train, df_test, eps, seed)
                    df.loc[len(df)] = [dataset, split, 'exp_gradient', seed, cl, met, acc, disp, eps]

df.to_csv(f'../Results/fairlearn/{dataset}_results.csv', index=False)

