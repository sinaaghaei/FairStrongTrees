'''
This module formulate the FlowOCT problem in gurobipy.
'''

from gurobipy import *
import numpy as np
from itertools import combinations


class FlowOCT:
    def __init__(self, data_enc, data_reg, label, tree, _lambda, time_limit, fairness_type, fairness_bound, protected_feature, positive_class, conditional_feature):
        '''

        :param data_enc: The encoded training data
        :param data_reg: The original training data
        :param label: Name of the column representing the class label
        :param tree: Tree object
        :param _lambda: The regularization parameter in the objective
        :param time_limit: The given time limit for solving the MIP
        :param fairness_type: The fairness constraint you wish to deploy
        :param fairness_bound: The bias bound
        :param protected_feature: The feature where fairness is evaluated
        '''

        self.data_enc = data_enc
        self.data_reg = data_reg
        self.datapoints = data_enc.index
        self.label = label

        self.labels = data_enc[label].unique()

        self.fairness_type = fairness_type
        self.fairness_bound = fairness_bound
        self.protected_feature = protected_feature
        self.positive_class = positive_class
        self.conditional_feature = conditional_feature

        '''
        cat_features is the set of all categorical features.
        reg_features is the set of all features used for the linear regression prediction model in the leaves.
        '''
        self.cat_features = self.data_enc.columns[self.data_enc.columns != self.label]
        # self.reg_features = None
        # self.num_of_reg_features = 1

        self.tree = tree
        self._lambda = _lambda

        # Decision Variables
        self.b = 0
        self.p = 0
        self.beta = 0
        self.zeta = 0
        self.z = 0

        # Gurobi model
        self.model = Model('FlowOCT')
        '''
        To compare all approaches in a fair setting we limit the solver to use only one thread to merely evaluate
        the strength of the formulation.
        '''
        self.model.params.Threads = 1
        self.model.params.TimeLimit = time_limit



    ###########################################################
    # Create the MIP formulation
    ###########################################################
    def create_primal_problem(self):
        '''
        This function create and return a gurobi model formulating the FlowOCT problem
        :return:  gurobi model object with the FlowOCT formulation
        '''
        ############################### define variables
        # b[n,f] ==1 iff at node n we branch on feature f
        self.b = self.model.addVars(self.tree.Nodes, self.cat_features, vtype=GRB.BINARY, name='b')
        # p[n] == 1 iff at node n we do not branch and we make a prediction
        self.p = self.model.addVars(self.tree.Nodes + self.tree.Leaves, vtype=GRB.BINARY, name='p')
        '''
        For classification beta[n,k]=1 iff at node n we predict class k
        For the case regression beta[n,1] is the prediction value for node n
        '''
        self.beta = self.model.addVars(self.tree.Nodes + self.tree.Leaves, self.labels, vtype=GRB.CONTINUOUS, lb=0,
                                       name='beta')
        # zeta[i,n,k] is the amount of flow through the edge connecting node n to sink node t,k for datapoint i
        self.zeta = self.model.addVars(self.datapoints, self.tree.Nodes + self.tree.Leaves, self.labels, vtype=GRB.BINARY, lb=0,
                                       name='zeta')
        # z[i,n] is the incoming flow to node n for datapoint i to terminal node k
        self.z = self.model.addVars(self.datapoints, self.tree.Nodes + self.tree.Leaves, vtype=GRB.BINARY, lb=0,
                                    name='z')

        # if self.fairness_type == 'SP':
        #     self.absolute = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name='absolute')

        ############################### define constraints

        # Constraint 9d
        # z[i,n] = z[i,l(n)] + z[i,r(n)] + (zeta[i,n,k] for all k in Labels)    forall i, n in Nodes
        for n in self.tree.Nodes:
            n_left = int(self.tree.get_left_children(n))
            n_right = int(self.tree.get_right_children(n))
            self.model.addConstrs(
                (self.z[i, n] == self.z[i, n_left] + self.z[i, n_right] + quicksum(self.zeta[i, n, k] for k in self.labels)) for i in self.datapoints)

        # Constraint 9g
        # z[i,l(n)] <= sum(b[n,f], f if x[i,f]=0)    forall i, n in Nodes
        for i in self.datapoints:
            self.model.addConstrs(self.z[i, int(self.tree.get_left_children(n))] <= quicksum(
                self.b[n, f] for f in self.cat_features if self.data_enc.at[i, f] == 0) for n in self.tree.Nodes)

        # Constraint 9h
        # z[i,r(n)] <= sum(b[n,f], f if x[i,f]=1)    forall i, n in Nodes
        for i in self.datapoints:
            self.model.addConstrs((self.z[i, int(self.tree.get_right_children(n))] <= quicksum(
                self.b[n, f] for f in self.cat_features if self.data_enc.at[i, f] == 1)) for n in self.tree.Nodes)

        # Constraint 9b
        # sum(b[n,f], f) + p[n] + sum(p[m], m in A(n)) = 1   forall n in Nodes
        self.model.addConstrs(
            (quicksum(self.b[n, f] for f in self.cat_features) + self.p[n] + quicksum(
                self.p[m] for m in self.tree.get_ancestors(n)) == 1) for n in
            self.tree.Nodes)

        # Constraint 9c
        # p[n] + sum(p[m], m in A(n)) = 1   forall n in Leaves
        self.model.addConstrs(
            (self.p[n] + quicksum(
                self.p[m] for m in self.tree.get_ancestors(n)) == 1) for n in
            self.tree.Leaves)

        # sum(sum(b[n,f], f), n) <= branching_limit
        # self.model.addConstr(
        #     (quicksum(
        #         quicksum(self.b[n, f] for f in self.cat_features) for n in self.tree.Nodes)) <= self.branching_limit)

        # Constraint 9i
        # sum(beta[n,k], k in labels) = p[n]
        for n in self.tree.Nodes + self.tree.Leaves:
            self.model.addConstrs(
                self.zeta[i, n, k] <= self.beta[n, k] for i in self.datapoints for k in self.labels)

        # Constraint 9j
        # sum(beta[n,k] for k in labels) == p[n]
        self.model.addConstrs(
            (quicksum(self.beta[n, k] for k in self.labels) == self.p[n]) for n in
            self.tree.Nodes + self.tree.Leaves)

        # Constraint 9e
        # z[i,n] == sum(zeta[i,n,k], k in labels)
        for n in self.tree.Leaves:
            self.model.addConstrs(quicksum(self.zeta[i, n, k] for k in self.labels) == self.z[i, n] for i in self.datapoints)

        # Constraint 9f
        # z[i,1] = 1 for all i datapoints
        self.model.addConstrs(self.z[i, 1] == 1 for i in self.datapoints)
        

        # Constraint Statistical Parity

        # if self.fairness_type == "SP":
        #
        #     # Loop through all possible combinations of the protected feature
        #     for combo in combinations(self.data_reg[self.protected_feature].unique(), 2):
        #         protected = combo[0]
        #         protected_prime = combo[1]
        #
        #         # Count how many samples correspond to each protected feature
        #         countProtected = np.count_nonzero(self.data_reg[self.protected_feature] == protected)
        #         countProtected_prime = np.count_nonzero(self.data_reg[self.protected_feature] == protected_prime)
        #
        #         # Sum(Sum(zeta(i,n,positive_class) for n in nodes) for i in datapoints) * 1 / (Count of Protected)
        #         self.model.addConstr(self.absolute >= (1/countProtected) * quicksum(quicksum(self.zeta[i,n, self.positive_class] for n in
        #                                                                  self.tree.Leaves + self.tree.Nodes)
        #                                                         for i in self.datapoints if self.data_reg.at[i, self.protected_feature] == protected) -
        #                               (1/countProtected_prime) * quicksum(quicksum(self.zeta[i,n,self.positive_class] for n in
        #                                                                  (self.tree.Leaves + self.tree.Nodes))
        #                                                         for i in self.datapoints if self.data_reg.at[i, self.protected_feature] == protected_prime))
        #         self.model.addConstr(self.absolute >= (-1/countProtected) * quicksum(quicksum(self.zeta[i,n,self.positive_class] for n in
        #                                                                  (self.tree.Leaves + self.tree.Nodes))
        #                                                         for i in self.datapoints if self.data_reg.at[i, self.protected_feature] == protected) +
        #                               (1/countProtected_prime) * quicksum(quicksum(self.zeta[i,n,self.positive_class] for n in
        #                                                                  (self.tree.Leaves + self.tree.Nodes))
        #                                                         for i in self.datapoints if self.data_reg.at[i, self.protected_feature] == protected_prime))
        #
        #         # Ensure absolute value is linearized
        #         self.model.addConstr(self.absolute <= self.fairness_bound)

        if self.fairness_type == "SP":

            # Loop through all possible combinations of the protected feature
            for combo in combinations(self.data_reg[self.protected_feature].unique(), 2):
                    protected = combo[0]
                    protected_prime = combo[1]

                    # Count how many samples correspond to each protected feature
                    countProtected = self.data_reg[self.data_reg[self.protected_feature] == protected].count()[self.label]
                    countProtected_prime = self.data_reg[self.data_reg[self.protected_feature] == protected_prime].count()[self.label]

                    protected_df = self.data_reg[self.data_reg[self.protected_feature] == protected]
                    protected_prime_df = self.data_reg[self.data_reg[self.protected_feature] == protected_prime]

                    # Sum(Sum(zeta(i,n,positive_class) for n in nodes) for i in datapoints) * 1 / (Count of Protected)
                    self.model.addConstr(((1/countProtected) * quicksum(quicksum(self.zeta[i,n, self.positive_class] for n in
                                                                             self.tree.Leaves + self.tree.Nodes)
                                                                    for i in protected_df.index) -
                                          ((1/countProtected_prime) * quicksum(quicksum(self.zeta[i,n,self.positive_class] for n in
                                                                             self.tree.Leaves + self.tree.Nodes)
                                                                    for i in protected_prime_df.index))) <= self.fairness_bound)

                    self.model.addConstr(((1/countProtected) * quicksum(quicksum(self.zeta[i,n,self.positive_class] for n in
                                                                             (self.tree.Leaves + self.tree.Nodes))
                                                                    for i in protected_df.index)) - (
                                          (1/countProtected_prime) * quicksum(quicksum(self.zeta[i,n,self.positive_class] for n in
                                                                             self.tree.Leaves + self.tree.Nodes)
                                                                    for i in protected_prime_df.index)) >= -1*self.fairness_bound)

        if self.fairness_type == "CSP":

            # Loop through all possible combinations of the protected feature
            for combo in combinations(self.data_reg[self.protected_feature].unique(), 2):
                for feature_value in self.data_reg[self.conditional_feature].unique():

                    protected = combo[0]
                    protected_prime = combo[1]

                    # Let's make our dataframe
                    protected_df_old = self.data_reg[self.data_reg[self.protected_feature] == protected]
                    protected_prime_df_old = self.data_reg[self.data_reg[self.protected_feature] == protected_prime]
                    protected_df = protected_df_old[protected_df_old[self.conditional_feature] == feature_value]
                    protected_prime_df = protected_prime_df_old[protected_prime_df_old[self.conditional_feature] == feature_value]

                    # Count how many samples correspond to each protected feature
                    countProtected = protected_df.count()[self.label]
                    countProtected_prime = protected_prime_df.count()[self.label]

                    if countProtected != 0 and countProtected_prime != 0:
                        # Sum(Sum(zeta(i,n,positive_class) for n in nodes) for i in datapoints) * 1 / (Count of Protected)
                        self.model.addConstr(((1/countProtected) * quicksum(quicksum(self.zeta[i,n, self.positive_class] for n in
                                                                                 self.tree.Leaves + self.tree.Nodes)
                                                                        for i in protected_df.index) -
                                              ((1/countProtected_prime) * quicksum(quicksum(self.zeta[i,n,self.positive_class] for n in
                                                                                 self.tree.Leaves + self.tree.Nodes)
                                                                        for i in protected_prime_df.index))) <= self.fairness_bound)

                        self.model.addConstr(((1/countProtected) * quicksum(quicksum(self.zeta[i,n,self.positive_class] for n in
                                                                                 (self.tree.Leaves + self.tree.Nodes))
                                                                        for i in protected_df.index)) - (
                                              (1/countProtected_prime) * quicksum(quicksum(self.zeta[i,n,self.positive_class] for n in
                                                                                 self.tree.Leaves + self.tree.Nodes)
                                                                        for i in protected_prime_df.index)) >= -1*self.fairness_bound)

                    elif countProtected == 0:
                        self.model.addConstr((0 -
                                              ((1/countProtected_prime) * quicksum(quicksum(self.zeta[i,n,self.positive_class] for n in
                                                                                 self.tree.Leaves + self.tree.Nodes)
                                                                        for i in protected_prime_df.index))) <= self.fairness_bound)

                        self.model.addConstr((0 - (1/countProtected_prime) * quicksum(quicksum(self.zeta[i,n,self.positive_class] for n in
                                                                                 self.tree.Leaves + self.tree.Nodes)
                                                                        for i in protected_prime_df.index)) >= -1*self.fairness_bound)

                    elif countProtected_prime == 0:
                        self.model.addConstr(((1/countProtected) * quicksum(quicksum(self.zeta[i,n, self.positive_class] for n in
                                                                                 self.tree.Leaves + self.tree.Nodes)
                                                                        for i in protected_df.index) -
                                              0) <= self.fairness_bound)

                        self.model.addConstr(((1/countProtected) * quicksum(quicksum(self.zeta[i,n,self.positive_class] for n in
                                                                                 (self.tree.Leaves + self.tree.Nodes))
                                                                        for i in protected_df.index)) - 
                                              0 >= -1*self.fairness_bound)


        if self.fairness_type == "PE":

            # Loop through all possible combinations of the protected feature
            for combo in combinations(self.data_reg[self.protected_feature].unique(), 2):

                protected = combo[0]
                protected_prime = combo[1]

                # Let's make our dataframe
                protected_df_old = self.data_reg[self.data_reg[self.protected_feature] == protected]
                protected_prime_df_old = self.data_reg[self.data_reg[self.protected_feature] == protected_prime]
                protected_df = protected_df_old[protected_df_old[self.label] != self.positive_class]
                protected_prime_df = protected_prime_df_old[protected_prime_df_old[self.label] != self.positive_class]

                # Count how many samples correspond to each protected feature
                countProtected = protected_df.count()[self.label]
                countProtected_prime = protected_prime_df.count()[self.label]

                if countProtected != 0 and countProtected_prime != 0:
                    # Sum(Sum(zeta(i,n,positive_class) for n in nodes) for i in datapoints) * 1 / (Count of Protected)
                    self.model.addConstr(((1/countProtected) * quicksum(quicksum(self.zeta[i,n, self.positive_class] for n in
                                                                             self.tree.Leaves + self.tree.Nodes)
                                                                    for i in protected_df.index) -
                                          ((1/countProtected_prime) * quicksum(quicksum(self.zeta[i,n,self.positive_class] for n in
                                                                             self.tree.Leaves + self.tree.Nodes)
                                                                    for i in protected_prime_df.index))) <= self.fairness_bound)

                    self.model.addConstr(((1/countProtected) * quicksum(quicksum(self.zeta[i,n,self.positive_class] for n in
                                                                             (self.tree.Leaves + self.tree.Nodes))
                                                                    for i in protected_df.index)) - (
                                          (1/countProtected_prime) * quicksum(quicksum(self.zeta[i,n,self.positive_class] for n in
                                                                             self.tree.Leaves + self.tree.Nodes)
                                                                    for i in protected_prime_df.index)) >= -1*self.fairness_bound)

                elif countProtected == 0:
                    self.model.addConstr((0 -
                                              ((1/countProtected_prime) * quicksum(quicksum(self.zeta[i,n,self.positive_class] for n in
                                                                                 self.tree.Leaves + self.tree.Nodes)
                                                                        for i in protected_prime_df.index))) <= self.fairness_bound)

                    self.model.addConstr((0 - (1/countProtected_prime) * quicksum(quicksum(self.zeta[i,n,self.positive_class] for n in
                                                                                 self.tree.Leaves + self.tree.Nodes)
                                                                        for i in protected_prime_df.index)) >= -1*self.fairness_bound)

                elif countProtected_prime == 0:
                    self.model.addConstr(((1/countProtected) * quicksum(quicksum(self.zeta[i,n, self.positive_class] for n in
                                                                                 self.tree.Leaves + self.tree.Nodes)
                                                                        for i in protected_df.index) -
                                              0) <= self.fairness_bound)

                    self.model.addConstr(((1/countProtected) * quicksum(quicksum(self.zeta[i,n,self.positive_class] for n in
                                                                                 (self.tree.Leaves + self.tree.Nodes))
                                                                        for i in protected_df.index)) - 
                                              0 >= -1*self.fairness_bound)

        if self.fairness_type == "EOdds":

            # We need to identify the non-positive class
            for i in self.labels:
                if i == self.positive_class:
                    continue
                else:
                    nonpositive = i

            # Loop through all possible combinations of the protected feature
            for combo in combinations(self.data_reg[self.protected_feature].unique(), 2):

                protected = combo[0]
                protected_prime = combo[1]
                label_list = [nonpositive, self.positive_class]

                for label_ in label_list:

                    # Let's make our dataframe
                    protected_df_old = self.data_reg[self.data_reg[self.protected_feature] == protected]
                    protected_prime_df_old = self.data_reg[self.data_reg[self.protected_feature] == protected_prime]
                    protected_df = protected_df_old[protected_df_old[self.label] == label_]
                    protected_prime_df = protected_prime_df_old[protected_prime_df_old[self.label] == label_]

                    # Count how many samples correspond to each protected feature
                    countProtected = protected_df.count()[self.label]
                    countProtected_prime = protected_prime_df.count()[self.label]

                    if countProtected != 0 and countProtected_prime != 0:
                        # Sum(Sum(zeta(i,n,positive_class) for n in nodes) for i in datapoints) * 1 / (Count of Protected)
                        self.model.addConstr(((1/countProtected) * quicksum(quicksum(self.zeta[i,n, self.positive_class] for n in
                                                                                 self.tree.Leaves + self.tree.Nodes)
                                                                        for i in protected_df.index) -
                                              ((1/countProtected_prime) * quicksum(quicksum(self.zeta[i,n,self.positive_class] for n in
                                                                                 self.tree.Leaves + self.tree.Nodes)
                                                                        for i in protected_prime_df.index))) <= self.fairness_bound)

                        self.model.addConstr(((1/countProtected) * quicksum(quicksum(self.zeta[i,n,self.positive_class] for n in
                                                                                 (self.tree.Leaves + self.tree.Nodes))
                                                                        for i in protected_df.index)) - (
                                              (1/countProtected_prime) * quicksum(quicksum(self.zeta[i,n,self.positive_class] for n in
                                                                                 self.tree.Leaves + self.tree.Nodes)
                                                                        for i in protected_prime_df.index)) >= -1*self.fairness_bound)

                    elif countProtected == 0:
                        self.model.addConstr((0 -
                                                  ((1/countProtected_prime) * quicksum(quicksum(self.zeta[i,n,self.positive_class] for n in
                                                                                     self.tree.Leaves + self.tree.Nodes)
                                                                            for i in protected_prime_df.index))) <= self.fairness_bound)

                        self.model.addConstr((0 - (1/countProtected_prime) * quicksum(quicksum(self.zeta[i,n,self.positive_class] for n in
                                                                                     self.tree.Leaves + self.tree.Nodes)
                                                                            for i in protected_prime_df.index)) >= -1*self.fairness_bound)

                    elif countProtected_prime == 0:
                        self.model.addConstr(((1/countProtected) * quicksum(quicksum(self.zeta[i,n, self.positive_class] for n in
                                                                                     self.tree.Leaves + self.tree.Nodes)
                                                                            for i in protected_df.index) -
                                                  0) <= self.fairness_bound)

                        self.model.addConstr(((1/countProtected) * quicksum(quicksum(self.zeta[i,n,self.positive_class] for n in
                                                                                     (self.tree.Leaves + self.tree.Nodes))
                                                                            for i in protected_df.index)) - 
                                                  0 >= -1*self.fairness_bound)

        if self.fairness_type == "EOpp":

            # Loop through all possible combinations of the protected feature
            for combo in combinations(self.data_reg[self.protected_feature].unique(), 2):

                protected = combo[0]
                protected_prime = combo[1]

                # Let's make our dataframe
                protected_df_old = self.data_reg[self.data_reg[self.protected_feature] == protected]
                protected_prime_df_old = self.data_reg[self.data_reg[self.protected_feature] == protected_prime]
                protected_df = protected_df_old[protected_df_old[self.label] == self.positive_class]
                protected_prime_df = protected_prime_df_old[protected_prime_df_old[self.label] == self.positive_class]

                # Count how many samples correspond to each protected feature
                countProtected = protected_df.count()[self.label]
                countProtected_prime = protected_prime_df.count()[self.label]

                if countProtected != 0 and countProtected_prime != 0:
                    # Sum(Sum(zeta(i,n,positive_class) for n in nodes) for i in datapoints) * 1 / (Count of Protected)
                    self.model.addConstr(((1/countProtected) * quicksum(quicksum(self.zeta[i,n, self.positive_class] for n in
                                                                             self.tree.Leaves + self.tree.Nodes)
                                                                    for i in protected_df.index) -
                                          ((1/countProtected_prime) * quicksum(quicksum(self.zeta[i,n,self.positive_class] for n in
                                                                             self.tree.Leaves + self.tree.Nodes)
                                                                    for i in protected_prime_df.index))) <= self.fairness_bound)

                    self.model.addConstr(((1/countProtected) * quicksum(quicksum(self.zeta[i,n,self.positive_class] for n in
                                                                             (self.tree.Leaves + self.tree.Nodes))
                                                                    for i in protected_df.index)) - (
                                          (1/countProtected_prime) * quicksum(quicksum(self.zeta[i,n,self.positive_class] for n in
                                                                             self.tree.Leaves + self.tree.Nodes)
                                                                    for i in protected_prime_df.index)) >= -1*self.fairness_bound)

                elif countProtected == 0:
                    self.model.addConstr((0 -
                                              ((1/countProtected_prime) * quicksum(quicksum(self.zeta[i,n,self.positive_class] for n in
                                                                                 self.tree.Leaves + self.tree.Nodes)
                                                                        for i in protected_prime_df.index))) <= self.fairness_bound)

                    self.model.addConstr((0 - (1/countProtected_prime) * quicksum(quicksum(self.zeta[i,n,self.positive_class] for n in
                                                                                 self.tree.Leaves + self.tree.Nodes)
                                                                        for i in protected_prime_df.index)) >= -1*self.fairness_bound)

                elif countProtected_prime == 0:
                    self.model.addConstr(((1/countProtected) * quicksum(quicksum(self.zeta[i,n, self.positive_class] for n in
                                                                                 self.tree.Leaves + self.tree.Nodes)
                                                                        for i in protected_df.index) -
                                              0) <= self.fairness_bound)

                    self.model.addConstr(((1/countProtected) * quicksum(quicksum(self.zeta[i,n,self.positive_class] for n in
                                                                                 (self.tree.Leaves + self.tree.Nodes))
                                                                        for i in protected_df.index)) - 
                                              0 >= -1*self.fairness_bound)


        # define objective function
        # Max sum(sum(zeta[i,n,y(i)]))

        # Add negative one #
        obj = LinExpr(0)
        for i in self.datapoints:
            for n in self.tree.Nodes + self.tree.Leaves:
                obj.add((1 - self._lambda) * (self.zeta[i, n, self.data_enc.at[i, self.label]]))

        for n in self.tree.Nodes:
            for f in self.cat_features:
                obj.add(-1 * self._lambda * self.b[n, f])

        self.model.setObjective(obj, GRB.MAXIMIZE)
