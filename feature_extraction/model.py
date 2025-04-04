from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegressionCV
import lightgbm as lgb
import xgboost as xgb
import warnings
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore", category=UserWarning)

rkf = RepeatedKFold(n_splits = 5, n_repeats = 3)

class cross_model(object):
    def rf(X, y, X_v, y_v):
        score_rf_add = []
        for train_index, test_index in rkf.split(X):
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]
            model_rf = RandomForestClassifier(n_estimators = 5).fit(X_train,y_train)
            y_pred_t = model_rf.predict(X_test)
            y_probs_t = model_rf.predict_proba(X_test)
            score_rf_add_add = model_rf.score(X_test,y_test)
            score_rf_add.append(score_rf_add_add)
        score = np.mean(score_rf_add)
        score_v = model_rf.score(X_v,y_v)
        y_pred = model_rf.predict(X_v)
        y_probs = model_rf.predict_proba(X_v)
        print(confusion_matrix(y_v, y_pred))
        print(classification_report(y_v, y_pred))
        return score, score_v,y_pred,y_probs,y_pred_t,y_probs_t,y_test
    
    def SVM(X, y, X_v, y_v):
        score_svm_add = []
        for train_index, test_index in rkf.split(X):
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]
            Cs = np.logspace(-1,3,10,base = 2)
            gammas = np.logspace(-4,1,50,base = 2)
            param_grid = dict(C = Cs, gamma = gammas)
            grid = GridSearchCV(SVC(kernel = 'rbf'),param_grid = param_grid, cv = 10).fit(X,y)
            C = grid.best_params_['C']
            gamma = grid.best_params_['gamma']
            model_svm = SVC(kernel = 'rbf',C = C, gamma = gamma,probability = True).fit(X_train,y_train)
            score_svm_add_add = model_svm.score(X_test,y_test)
            score_svm_add.append(score_svm_add_add)
            y_pred_t = model_svm.predict(X_test)
            y_probs_t = model_svm.predict_proba(X_test)
        score = np.mean(score_svm_add)
        score_v = model_svm.score(X_v,y_v)
        y_pred = model_svm.predict(X_v)
        y_probs = model_svm.predict_proba(X_v)
        
        print(confusion_matrix(y_v, y_pred))
        print(classification_report(y_v, y_pred))
        return score, score_v,y_pred,y_probs,y_pred_t,y_probs_t,y_test

    def dbt(X, y, X_v, y_v):
        score_dbt_add = []
        for train_index, test_index in rkf.split(X):
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]
            dbt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5), algorithm="SAMME", n_estimators=400, learning_rate=0.8)
            model_dbt = dbt.fit(X_train,y_train)
            score_dbt_add_add = model_dbt.score(X_test,y_test)
            score_dbt_add.append(score_dbt_add_add)
            y_pred_t = model_dbt.predict(X_test)
            y_probs_t = model_dbt.predict_proba(X_test)
        score = np.mean(score_dbt_add)
        score_v = model_dbt.score(X_v,y_v)
        y_pred = model_dbt.predict(X_v)
        y_probs = model_dbt.predict_proba(X_v)
        print(confusion_matrix(y_v, y_pred))
        print(classification_report(y_v, y_pred))
        return score, score_v,y_pred,y_probs,y_pred_t,y_probs_t,y_test


    def knn(X, y, X_v, y_v):
        score_knn_add = []
        for train_index, test_index in rkf.split(X):
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]
            model_knn = KNeighborsClassifier().fit(X_train, y_train)
            score_knn_add_add = model_knn.score(X_test, y_test)
            score_knn_add.append(score_knn_add_add)
            y_pred_t = model_knn.predict(X_test)
            y_probs_t = model_knn.predict_proba(X_test)
        score = np.mean(score_knn_add)
        score_v = model_knn.score(X_v,y_v)
        y_pred = model_knn.predict(X_v)
        y_probs = model_knn.predict_proba(X_v)
    
        print(confusion_matrix(y_v, y_pred))
        print(classification_report(y_v, y_pred))
        return score, score_v,y_pred,y_probs,y_pred_t,y_probs_t,y_test


    def NB(X, y, X_v ,y_v):
        score_NB_add = []
        for train_index, test_index in rkf.split(X):
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]
            model_NB = GaussianNB().fit(X_train , y_train)
            score_NB_add_add = model_NB.score(X_test, y_test)
            score_NB_add.append(score_NB_add_add)
            y_pred_t = model_NB.predict(X_test)
            y_probs_t = model_NB.predict_proba(X_test)
        score = np.mean(score_NB_add)
        score_v = model_NB.score(X_v,y_v)
        y_pred = model_NB.predict(X_v)
        y_probs = model_NB.predict_proba(X_v)
        
        print(confusion_matrix(y_v, y_pred))
        print(classification_report(y_v, y_pred))
        return score, score_v,y_pred,y_probs,y_pred_t,y_probs_t,y_test


    def BNB(X, y, X_v, y_v):
        score_BNB_add = []
        for train_index, test_index in rkf.split(X):
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]
            model_BNB = BernoulliNB().fit(X_train , y_train)
            score_BNB_add_add = model_BNB.score(X_test, y_test)
            score_BNB_add.append(score_BNB_add_add)
        score = np.mean(score_BNB_add)
        score_v = model_BNB.score(X_v,y_v)
        y_pred = model_BNB.predict(X_v)
        y_probs = model_BNB.predict_proba(X_v)
        y_pred_t = model_BNB.predict(X_test)
        y_probs_t = model_BNB.predict_proba(X_test)
        print(confusion_matrix(y_v, y_pred))
        print(classification_report(y_v, y_pred))
        return score, score_v,y_pred,y_probs,y_pred_t,y_probs_t,y_test


    def Dtree(X, y, X_v, y_v):
        score_Dtree_add = []
        for train_index, test_index in rkf.split(X):
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]
            model_Dtree = DecisionTreeClassifier().fit(X_train,y_train)
            score_Dtree_add_add = model_Dtree.score(X_test, y_test)
            score_Dtree_add.append(score_Dtree_add_add)
        score = np.mean(score_Dtree_add)
        score_v = model_Dtree.score(X_v,y_v)
        y_pred = model_Dtree.predict(X_v)
        y_probs = model_Dtree.predict_proba(X_v)
        y_pred_t = model_Dtree.predict(X_test)
        y_probs_t = model_Dtree.predict_proba(X_test)
        print(confusion_matrix(y_v, y_pred))
        print(classification_report(y_v, y_pred))
        return score, score_v,y_pred,y_probs,y_pred_t,y_probs_t,y_test


    def Gtree(X, y, X_v, y_v):
        score_Gtree_add = []
        for train_index, test_index in rkf.split(X):
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]
            model_Gtree = GradientBoostingClassifier().fit(X_train,y_train)
            score_Gtree_add_add = model_Gtree.score(X_test, y_test)
            score_Gtree_add.append(score_Gtree_add_add)
        score = np.mean(score_Gtree_add)
        score_v = model_Gtree.score(X_v,y_v)
        y_pred = model_Gtree.predict(X_v)
        y_probs = model_Gtree.predict_proba(X_v)
        y_pred_t = model_Gtree.predict(X_test)
        y_probs_t = model_Gtree.predict_proba(X_test)
        print(confusion_matrix(y_v, y_pred))
        print(classification_report(y_v, y_pred))
        return score, score_v,y_pred,y_probs,y_pred_t,y_probs_t,y_test


    def mlp(X, y, X_v, y_v):
        score_mlp_add = []
        for train_index, test_index in rkf.split(X):
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]
            model_mlp = MLPClassifier(solver="lbfgs",random_state=1,max_iter=10000).fit(X_train,y_train)
            score_mlp_add_add = model_mlp.score(X_test, y_test)
            score_mlp_add.append(score_mlp_add_add)
        score = np.mean(score_mlp_add)
        score_v = model_mlp.score(X_v,y_v)
        y_pred = model_mlp.predict(X_v)
        y_probs = model_mlp.predict_proba(X_v)
        y_pred_t = model_mlp.predict(X_test)
        y_probs_t = model_mlp.predict_proba(X_test)
        print(confusion_matrix(y_v, y_pred))
        print(classification_report(y_v, y_pred))
        return score, score_v,y_pred,y_probs,y_pred_t,y_probs_t,y_test


    def gbm(X, y, X_v, y_v):
        score_gbm_add = []
        for train_index, test_index in rkf.split(X):
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]
            model_gbm = lgb.LGBMClassifier().fit(X_train, y_train)
            score_gbm_add_add = model_gbm.score(X_test, y_test)
            score_gbm_add.append(score_gbm_add_add)
        score = np.mean(score_gbm_add)
        score_v = model_gbm.score(X_v,y_v)
        y_pred = model_gbm.predict(X_v)
        y_probs = model_gbm.predict_proba(X_v)
        y_pred_t = model_gbm.predict(X_test)
        y_probs_t = model_gbm.predict_proba(X_test)
        print(confusion_matrix(y_v, y_pred))
        print(classification_report(y_v, y_pred))
        return score, score_v,y_pred,y_probs,y_pred_t,y_probs_t,y_test



    def xgb(X, y, X_v, y_v):
        score_xgb_add = []
        for train_index, test_index in rkf.split(X):
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]
            model_xgb = xgb.XGBClassifier(eval_metric=['logloss','auc','error']).fit(X_train,y_train)
            score_xgb_add_add = model_xgb.score(X_test, y_test)
            score_xgb_add.append(score_xgb_add_add)
        score = np.mean(score_xgb_add)
        score_v = model_xgb.score(X_v,y_v)
        y_pred = model_xgb.predict(X_v)
        y_probs = model_xgb.predict_proba(X_v)
        y_pred_t = model_xgb.predict(X_test)
        y_probs_t = model_xgb.predict_proba(X_test)
        print(confusion_matrix(y_v, y_pred))
        print(classification_report(y_v, y_pred))
        return score, score_v,y_pred,y_probs,y_pred_t,y_probs_t,y_test


    def lore(X, y, X_v, y_v):
        score_lore_add = []
        for train_index, test_index in rkf.split(X):
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]
            model_lore = LogisticRegressionCV(multi_class='ovr',fit_intercept=True, Cs=np.logspace(-2, 2, 20), cv=2, 
                                penalty='l2', solver='lbfgs', tol=0.01).fit(X_train, y_train)
            score_lore_add_add = model_lore.score(X_test, y_test)
            score_lore_add.append(score_lore_add_add)
        score = np.mean(score_lore_add)
        score_v = model_lore.score(X_v,y_v)
        y_pred = model_lore.predict(X_v)
        y_probs = model_lore.predict_proba(X_v)
        y_pred_t = model_lore.predict(X_test)
        y_probs_t = model_lore.predict_proba(X_test)
        print(confusion_matrix(y_v, y_pred))
        print(classification_report(y_v, y_pred))
        return score, score_v,y_pred,y_probs,y_pred_t,y_probs_t,y_test





