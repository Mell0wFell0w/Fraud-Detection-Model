def import_data(path, messages=True):
  import pandas as pd
  df = pd.read_csv(path)
  if messages: print(df.shape)
  return df

def bin_groups(df, features=[], cutoff=0.05, replace_with='Other', messages=True):
  import pandas as pd
  if len(features) == 0: features = df.columns
  for feat in features:
    if feat in df.columns:  # Make sure they don't accidentally enter a feature name that doesn't exist
      if not pd.api.types.is_numeric_dtype(df[feat]):
        other_list = df[feat].value_counts()[df[feat].value_counts() / df.shape[0] < cutoff].index
        if len(other_list) > 0:
            df.loc[df[feat].isin(other_list), feat] = replace_with
            if messages and len(other_list) > 0: print(f'{feat} has been binned by setting {other_list.values} to {replace_with}')
    else:
      if messages: print(f'{feat} not found in the DataFrame provided. No binning performed')
  return df

def missing_drop(df, label, row_thresh=0.7, col_thresh=0.9, drop_all=False):
  df.dropna(axis='rows', subset=[label], inplace=True)
  df.dropna(axis='columns', thresh=1, inplace=True)
  df.dropna(axis='rows', thresh=1, inplace=True)
  df.dropna(axis='columns', thresh=round(df.shape[0] * row_thresh), inplace=True)
  df.dropna(axis='rows', thresh=round(df.shape[1] * col_thresh), inplace=True)
  df.drop(columns=['date', 'Unnamed: 0', 'customer_id'], inplace=True)
  if drop_all: df.dropna(axis='rows', inplace=True)
  return df

def Xandy(df, label):
  import pandas as pd
  y = df[label]
  X = df.drop(columns=[label])
  return X, y

def dummy_code(X):
  import pandas as pd
  X = pd.get_dummies(X, drop_first=True)
  return X

def minmax(X):
  import pandas as pd
  from sklearn.preprocessing import MinMaxScaler
  X = pd.DataFrame(MinMaxScaler().fit_transform(X.copy()), columns=X.columns, index=X.index)
  return X

def impute_KNN(df, label, neighbors=5):
  from sklearn.impute import KNNImputer
  import pandas as pd
  X, y = Xandy(df, label)
  X = dummy_code(X.copy())
  X = minmax(X.copy())
  imp = KNNImputer(n_neighbors=neighbors, weights="uniform")
  X = pd.DataFrame(imp.fit_transform(X), columns=X.columns, index=X.index)
  return X.merge(y, left_index=True, right_index=True)


def select_features(df, label, model, max='auto'):
  from sklearn.feature_selection import SelectFromModel
  import pandas as pd
  X, y = Xandy(df, label)
  if max != 'auto':
    sel = SelectFromModel(model, prefit=True, max_features=round(max*df.drop(columns=[label]).shape[1]))
  else:
    sel = SelectFromModel(model, prefit=True)
  sel.transform(X)
  columns = list(X.columns[sel.get_support()])
  columns.append(label)
  return df[columns]



def dump_pickle(model, file_name):
  import pickle
  pickle.dump(model, open(file_name, "wb"))

def load_pickle(file_name):
  import pickle
  model = pickle.load(open(file_name, "rb"))
  return model

def fit_cv_classification(df, k, label, repeat=True, algorithm='ensemble', random_state=1, messages=True):
  from sklearn.model_selection import KFold, RepeatedKFold, cross_val_score
  import pandas as pd
  from numpy import mean
  X, y = Xandy(df, label)
  X = dummy_code(X)
  if repeat:  cv = RepeatedKFold(n_splits=k, n_repeats=5, random_state=12345)
  else:       cv = KFold(n_splits=k, random_state=12345, shuffle=True)
  if algorithm == 'linear':
    from sklearn.linear_model import RidgeClassifier, SGDClassifier
    model1 = RidgeClassifier(random_state=random_state)
    model2 = SGDClassifier(random_state=random_state)
    score1 = mean(cross_val_score(model1, X, y, scoring='accuracy', cv=cv, n_jobs=-1))
    score2 = mean(cross_val_score(model2, X, y, scoring='accuracy', cv=cv, n_jobs=-1))
  elif algorithm == 'ensemble':
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    model1 = RandomForestClassifier(random_state=random_state)
    model2 = GradientBoostingClassifier(random_state=random_state)
    score1 = mean(cross_val_score(model1, X, y, scoring='accuracy', cv=cv, n_jobs=-1))
    score2 = mean(cross_val_score(model2, X, y, scoring='accuracy', cv=cv, n_jobs=-1))
  else:
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    model1 = MLPClassifier(random_state=random_state, max_iter=10000)
    model2 = KNeighborsClassifier()
    score1 = mean(cross_val_score(model1, X, y, scoring='accuracy', cv=cv, n_jobs=-1))
    score2 = mean(cross_val_score(model2, X, y, scoring='accuracy', cv=cv, n_jobs=-1))
  if messages:
    print('Accuracy', '{: <25}'.format(type(model1).__name__), round(score1, 4))
    print('Accuracy', '{: <25}'.format(type(model2).__name__), round(score2, 4))
  if score1 > score2: return model1.fit(X, y)
  else:               return model2.fit(X, y)

def fit_cv_classification_expanded(df, label, k=10, r=5, repeat=True, random_state=1):
    import sklearn.linear_model as lm
    import pandas as pd
    import sklearn.ensemble as se
    import numpy as np
    from sklearn.model_selection import KFold, RepeatedKFold, cross_val_score
    from numpy import mean, std
    from sklearn import svm
    from sklearn import gaussian_process
    from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn import svm
    from sklearn.naive_bayes import CategoricalNB
    from xgboost import XGBClassifier
    from sklearn import preprocessing
    from sklearn.neural_network import MLPClassifier

    X, y = Xandy(df, label)

    if repeat:
        cv = RepeatedKFold(n_splits=k, n_repeats=r, random_state=random_state)
    else:
        cv = KFold(n_splits=k, random_state=random_state, shuffle=True)

    fit = {}    # Use this to store each of the fit metrics
    models = {} # Use this to store each of the models

    # Create the model objects
    # model_log = lm.LogisticRegression(max_iter=100)
    # model_logcv = lm.RidgeClassifier()
    # model_sgd = lm.SGDClassifier(max_iter=1000, tol=1e-3)
    # model_pa = lm.PassiveAggressiveClassifier(max_iter=1000, random_state=random_state, tol=1e-3)
    # model_per = lm.Perceptron(fit_intercept=False, max_iter=10, tol=None, shuffle=False)
    # model_knn = KNeighborsClassifier(n_neighbors=3)
    # model_svm = svm.SVC(decision_function_shape='ovo') # Remove the parameter for two-class model

    # model_bag = se.BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
    model_ada = se.AdaBoostClassifier(n_estimators=100, random_state=random_state)
    # model_ext = se.ExtraTreesClassifier(n_estimators=100, random_state=random_state)
    # model_rf = se.RandomForestClassifier(n_estimators=10)
    # model_hgb = se.HistGradientBoostingClassifier(max_iter=100)
    # model_vot = se.VotingClassifier(estimators=[('lr', model_log), ('rf', model_ext), ('gnb', model_hgb)], voting='hard')
    # model_gb = se.GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    # estimators = [('ridge', lm.RidgeCV()), ('lasso', lm.LassoCV(random_state=random_state)), ('knr', KNeighborsRegressor(n_neighbors=20, metric='euclidean'))]
    # final_estimator = se.GradientBoostingRegressor(n_estimators=25, subsample=0.5, min_samples_leaf=25, max_features=1, random_state=random_state)
    # model_st = se.StackingRegressor(estimators=estimators, final_estimator=final_estimator)
    # model_nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=random_state)
    # model_xgb = XGBClassifier()

    # # Fit a cross-validated R squared score and add it to the dict
    # fit['Logistic'] = mean(cross_val_score(model_log, X, y, scoring='accuracy', cv=cv, n_jobs=-1))
    # fit['Ridge'] = mean(cross_val_score(model_logcv, X, y, scoring='accuracy', cv=cv, n_jobs=-1))
    # fit['SGD'] = mean(cross_val_score(model_sgd, X, y, scoring='accuracy', cv=cv, n_jobs=-1))
    # fit['PassiveAggressive'] = mean(cross_val_score(model_pa, X, y, scoring='accuracy', cv=cv, n_jobs=-1))
    # fit['Perceptron'] = mean(cross_val_score(model_per, X, y, scoring='accuracy', cv=cv, n_jobs=-1))
    # fit['KNN'] = mean(cross_val_score(model_knn, X, y, scoring='accuracy', cv=cv, n_jobs=-1))
    # fit['SVM'] = mean(cross_val_score(model_svm, X, y, scoring='accuracy', cv=cv, n_jobs=-1))

    # fit['Bagging'] = mean(cross_val_score(model_bag, X, y, scoring='accuracy', cv=cv, n_jobs=-1))
    fit['AdaBoost'] = mean(cross_val_score(model_ada, X, y, scoring='accuracy', cv=cv, n_jobs=-1))
    # fit['ExtraTrees'] = mean(cross_val_score(model_ext, X, y, scoring='accuracy', cv=cv, n_jobs=-1))
    # fit['RandomForest'] = mean(cross_val_score(model_rf, X, y, scoring='accuracy', cv=cv, n_jobs=-1))
    # fit['HistGradient'] = mean(cross_val_score(model_hgb, X, y, scoring='accuracy', cv=cv, n_jobs=-1))
    # fit['Voting'] = mean(cross_val_score(model_vot, X, y, scoring='accuracy', cv=cv, n_jobs=-1))
    # fit['GradBoost'] = mean(cross_val_score(model_gb, X, y, scoring='accuracy', cv=cv, n_jobs=-1))
    # fit['NeuralN'] = mean(cross_val_score(model_nn, X, y, scoring='accuracy', cv=cv, n_jobs=-1))

    # # XGBoost needs to LabelEncode the y before fitting the model
    # from sklearn.preprocessing import LabelEncoder
    # le = LabelEncoder().fit(y)
    # y_encoded = le.transform(y)
    # fit['XGBoost'] = mean(cross_val_score(model_xgb, X, y_encoded, scoring='accuracy', cv=cv, n_jobs=-1))

    # # Add the model to another dict; make sure the keys have the same names as the list above
    # models['Logistic'] = model_log
    # models['Ridge'] = model_logcv
    # models['SGD'] = model_sgd
    # models['PassiveAggressive'] = model_pa
    # models['Perceptron'] = model_per
    # models['KNN'] = model_knn
    # models['SVM'] = model_svm

    # models['Bagging'] = model_bag
    models['AdaBoost'] = model_ada
    # models['ExtraTrees'] = model_ext
    # models['RandomForest'] = model_rf
    # models['HistGradient'] = model_hgb
    # models['Voting'] = model_vot
    # models['GradBoost'] = model_gb
    # models['XGBoost'] = model_xgb
    # models['NeuralN'] = model_nn

    # Add the fit dictionary to a new DataFrame, sort, extract the top row, use it to retrieve the model object from the models dictionary
    df_fit = pd.DataFrame({'Accuracy': fit})
    df_fit.sort_values(by=['Accuracy'], ascending=False, inplace=True)
    best_model = df_fit.index[0]
    print(df_fit)

    return models[best_model].fit(X, y)
