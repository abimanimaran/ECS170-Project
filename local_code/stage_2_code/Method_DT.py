'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

class Method_DT(method):
    c = None
    data = None
    
    def train(self, X, y):
        # check here for the decision tree classifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
        #model = tree.DecisionTreeClassifier(
            #criterion='entropy',
            #max_depth=40,
            #min_samples_split=10,
            #min_samples_leaf=10,
            #random_state=1
        #)

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=30,
            min_samples_split=10,
            min_samples_leaf=10,
            random_state=1,
            n_jobs=-1

        )
        # check here for decision tree fit doc: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.fit
        model.fit(X, y)
        return model
    
    def test(self, model, X):
        # check here for decision tree predict doc: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.predict
        return model.predict(X)
    
    def run(self):
        print('method running...')
        print('--start training...')
        model = self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(model, self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
            