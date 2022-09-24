from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
optuna.logging.disable_default_handler

import utils.feature_selections as fs
import utils.get_top_n_features as top_features
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import pandas as pd

from sklearn.decomposition import PCA

def pipeline(
                features_df, train_df, val_df,
                left_on='CLIENT_ID', right_on='CLIENT_ID',
                tech_cols=['CLIENT_ID'], target_column='TARGET',
                drop_singular_cut_off=0.9,
                drop_corr_cut_off=0.75,
                n_trials=100,
                model_filename = 'model_classic.txt'
            ):
    '''
    ==== Аргументы ====
    • features_df           - DataFrame с ID элемента (CLIENT_ID), остальные столбца - произвольные фичи
    • train_df и val_df     - DataFrame`ы с ID элемента (CLIENT_ID) и таргет столбцом (TARGET)
    • left_on и right_on    - параметры для pd.merge между train_df/val_df и features_df
    • tech_cols             - список ID элементов
    • target_column         - таргет столбец
    • drop_singular_cut_off и drop_corr_cut_off
                            - параметры для функций из utils
    • n_trials              - кол-во итераций при поиске оптимальных параметров
    • model_filename        - название файла для сохранения модели

    ==== Функционал ====
    Pipeline:
        • Удаление фичей:
            • drop_singular()
            • drop_corr()
        • Соединение train_df/val_df и features_df
            • Заполнеие fillna средними значениями фичей
        • Base level
        • Поиск оптимальный гиперпараметров
        • Обучение модели
        • Потсроение ROC AUC

    Функция возвращает объект lgb.LGBMClassifier() (обученную)

    '''
    line_str = '='*72 + '\n'

    ## CUTTING features_df
    print(line_str + 'CUTTING features_df')
    drop_list_singular = drop_singular(features_df,  cut_off=drop_singular_cut_off)
    print(f'Длина DataFrame после drop_singular(features_df, {drop_singular_cut_off}): {len(drop_list_singular)}')
    features_df.drop(drop_list_singular, axis = 1, inplace = True)

    drop_list_corr = drop_corr(features_df, tech_cols, drop_corr_cut_off)
    print(f'Длина DataFrame после drop_corr(features_df, {tech_cols}, {drop_corr_cut_off}): {len(drop_list_corr)}')
    features_df.drop(drop_list_corr, axis = 1, inplace = True)

    ## MERGING into train and validations dataframes
    print(line_str + 'MERGING into train and validations dataframes')
    train_df = train_df.merge(features_df, how='left', left_on=left_on, right_on=right_on)
    val_df   =   val_df.merge(features_df, how='left', left_on=left_on, right_on=right_on)

    ## filling with mean values
    train_df = train_df.apply(lambda x: x.fillna(x.mean()),axis=0)
    val_df   =   val_df.apply(lambda x: x.fillna(x.mean()),axis=0)

    x_train = train_df.drop(tech_cols + [target_column], axis = 1)
    x_test  = val_df.drop(tech_cols + [target_column], axis = 1)
    y_train = train_df[target_column]
    y_test  = val_df[target_column]

    ## BASE level
    print(line_str + 'ASE level')
    model = lgb.LGBMClassifier()
    model.fit(x_train, y_train)

    roc_value = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
    print(f'roc_auc_score для LGBMClassifier: {roc_value}')

    ## SEARCH for best params
    print(line_str + 'SEARCH for best params')
    study_lgb = optuna.create_study(direction = 'maximize')
    study_lgb.optimize(lambda trial: objective(trial, x_train, y_train, x_test, y_test, 'LightGBM'), n_trials=n_trials)

    print('-'*20)
    print('Best params:')
    print(study_lgb.best_trial.params)

    params = study_lgb.best_trial.params

    print('\n')
    argList = get_top_features('LightGBM', target_column, params, x_train, y_train, x_test, y_test)
    print(f'len(argList): {len(argList)}')
    print(f'argList: {argList}')

    print('\n')
    argList2 = fs.backward_selection(argList, model, x_train, y_train, x_test, y_test)
    print(f'len(argList2): {len(argList2)}')
    print(f'argList2: {argList2}')

    study_lgb2 = optuna.create_study(direction = 'maximize')
    study_lgb2.optimize(lambda trial: objective(trial, x_train[argList2], y_train, x_test[argList2], y_test, 'LightGBM'), n_trials=n_trials)

    ## Training a model
    print(line_str + 'Training a model')
    model = lgb.LGBMClassifier(**study_lgb.best_trial.params)
    model.fit(x_train[argList2], y_train)

    probs_train = model.predict_proba(x_train[argList2])[:, 1]
    probs_test = model.predict_proba(x_test[argList2])[:, 1]

    roc_train = roc_auc_score(y_train, probs_train)
    roc_test = roc_auc_score(y_test, probs_test)

    tpr_train, fpr_train, _ = roc_curve(y_train, probs_train)
    tpr_test, fpr_test, _ = roc_curve(y_test, probs_test)

    fig = plt.figure(figsize = (5, 5))
    plt.plot(tpr_train, fpr_train, label = f'ROC AUC Train: {roc_train:.2f}')
    plt.plot(tpr_test, fpr_test, label = f'ROC AUC Test: {roc_test:.2f}')
    plt.plot([0, 1], [0,1], label = 'Baseline', ls = '--')
    plt.title('ROC AUC Score')
    plt.legend()
    plt.show()

    print(y_train.sum()/y_train.shape[0], y_test.sum()/y_test.shape[0])
    model.booster_.save_model(model_filename)
    
    return {
        'model': model,
        'argList2': argList2,
        'probs_train': probs_train,
        'probs_test': probs_test,
        'dataframes': {
            'x_train': x_train,
            'x_test': x_test,
            'y_train': y_train,
            'y_test': y_test
        }

    }

def drop_singular(df, cut_off):
    delColList = []
    for col in df.columns:
        if max(df[col].value_counts() / df.shape[0]) > cut_off:
            delColList.append(col)
    return delColList

def drop_corr(df, tech_cols, cut_off):
    delColList = []
    df_ = df.sample(frac = 0.1)
    argList = list(df.columns)
    for i in tech_cols:
        argList.remove(i)
    while 1:
        if (len(argList) <= 1):
            break
        
        feature = argList[0]
        
        for i in argList[1:]:
            try:
                if abs(df_[[i, feature]].corr().iloc[0][1]) >= cut_off:
                    delColList.append(feature)
                    break
            except Exception:
                print("Не рассчиталась корреляция между фичами " + i + " и " + feature)
        argList.remove(feature)
    return delColList
                
def backward_selection(argList, model, x_train, y_train, x_test, y_test):
    argList2 = argList.copy()
    model.fit(x_train[argList2], y_train)
    probs_test = model.predict_proba(x_test[argList2])[:, 1]
    roc = roc_auc_score(y_test, probs_test)
    while 1:
        best_roc_dif = 0
        for i in argList2:
            model.fit(x_train[argList2].drop(i, axis = 1), y_train)
            probs_test = model.predict_proba(x_test[argList2].drop(i, axis = 1))[:, 1]
            roc_new = roc_auc_score(y_test, probs_test)
            if roc_new / roc >= 0.99:
                if (roc_new / roc > best_rock_did):
                    best_rock_did = roc_new / roc
                    perem_del = i
        if (best_roc_dif == 0):
            break
        else:
            argList2.remove(perem_del)
            model.fit(x_train[argList2], y_train)
            probs_test = model.predict_proba(x_test[argList2])[:, 1]
    return argList2

def get_top_features(alg, target_name, params, x_train, y_train, x_test, y_test):
    if alg == 'XGB':
        model = xgb.XGBClassifier(n_jobs = 4)
    elif alg == 'LightGBM':
        model = lgb.LGBMClassifier(n_jobs = 4)
    else:
        cb.CatboostClassifier()
        
    model.set_params(**params)
    model.fit(x_train, y_train, verbose = False)
    
    fi = pd.DataFrame({'feature': list(x_train.columns),
                       'importance': model.feature_importances_}).sort_values('importance', ascending = True)
    
    rocTest = []
    rocTrain = []
    rocPredictors = [50, 45, 40, 35, 30, 25, 24, 23, 22, 21, 20,
                     19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3]
    argList = []
    
    for i in rocPredictors:
        model.fit(x_train[list(fi['feature'][-i:])], y_train)
        
        probs_test = model.predict_proba(x_test[list(fi['feature'][-i:])])[:, 1]
        rocTest.append(roc_auc_score(y_test, probs_test))
        
        probs_train = model.predict_proba(x_train[list(fi['feature'][-i:])])[:, 1]
        rocTrain.append(roc_auc_score(y_train, probs_train))
        
        argList.append(list(fi['feature'][-i:]))
        
    return argList[len(rocTest) - rocTest[::-1].index(max(rocTest)) - 1]

def objective(trial, x_train, y_train, x_test, y_test, alg):
    target_share = y_train.mean()
    if target_share >= 0.5:
        scale_left = (1 - target_share) / target_share
        scale_right = 1.0
    else:
        scale_left = 1.0
        scale_right = (1 - target_share) / target_share
    if alg == 'LightGBM':
        param = {'n_estimators': trial.suggest_int('n_estimators', 8, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5002),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
                'subsample': trial.suggest_float('subsample', 0.4, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.05, 0.85),
                'num_leaves': trial.suggest_int('num_leaves', 4, 256),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', scale_left, scale_right)}
        model = lgb.LGBMClassifier(n_jobs = 8)
    else:
        param = {'n_estimators': trial.suggest_int('n_estimators', 8, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5002),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
                'subsample': trial.suggest_float('subsample', 0.4, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.05, 0.85),
                'gamma': trial.suggest_float('gamma', 0.01, 0.91),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', scale_left, scale_right)}
        model = xgb.XGBClassifier(n_jobs = 8)
    model.set_params(**param)
    model.fit(x_train, y_train)
    roc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
    return roc

def group_columns(df, ids='CLIENT_ID', target='TARGET', n_components=5):
    columns_groups = []
    columns_groups_names = []

    long_i_cols = [col for col in list(df.columns) if 'i' in col and len(col) == 5]
    starts = list(set(map(lambda x: x[:2], long_i_cols)))
    for start in sorted(starts):
        columns_groups.append([col for col in long_i_cols if start == col[:2]])
        columns_groups_names.append(start + '000')

    short_i_cols = [col for col in list(df.columns) if 'i' in col and len(col) == 4]
    starts = list(set(map(lambda x: x[:2], short_i_cols)))
    for start in sorted(starts):
        columns_groups.append([col for col in short_i_cols if start == col[:2]])
        columns_groups_names.append(start + '00')

    u_cols = [col for col in list(df.columns) if 'u' in col]
    starts = list(set(map(lambda x: x[:2], u_cols)))
    for start in sorted(starts):
        columns_groups.append([col for col in u_cols if start == col[:2]])
        columns_groups_names.append(start)
        
    new_features_df = df[[ids]].copy()

    for columns, group_name in zip(columns_groups, columns_groups_names):
        step_df = df[columns].copy()
        columns = step_df.columns

        ## PCA
        if n_components < len(step_df.columns):
            # pca = PCA(n_components=n_components)
            # pca.fit(step_df)
            # step_df = pd.DataFrame(pca.components_.T)
            pca = PCA(n_components=n_components)
            principalComponents = pca.fit_transform(step_df)
            columns = [f'{group_name}_{i}' for i in range(n_components)]
            step_df = pd.DataFrame(data = principalComponents, columns = columns)

        ## taking mean
        # step_df=(step_df-step_df.min())/(step_df.max()-step_df.min())
        # new_features_df[group_name] = step_df[columns].mean(axis=1)

        new_features_df[columns] = step_df

    return new_features_df