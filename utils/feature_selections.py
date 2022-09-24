from itertools import combinations
from sklearn.metrics import roc_auc_score, mean_absolute_error
from collections import Counter

def forward_selection_regression(argList, model, x_train, y_train, x_test, y_test, max_add_features_per_iter,
                                 quality_improve, max_features_to_select):
    
    if max_features_to_select > len(argList):
        max_features_to_select = len(argList)
        
    if len(quality_improve) != max_add_features_per_iter:
        quality_improve = [0.4 * i for i in range(1, max_add_features_per_iter + 1)]
        
    forward_argList = []
    
    best_metric = 1e10
    
    for k in range(1, max_add_features_per_iter + 1):
        print(f'Add {k} features per iter')
        
        while True:
            if len(forward_argList) + k > max_features_to_select:
                return forward_argList
            
            best_diff = 1e10
            for features in list(combinations(set(argList) - set(forward_argList), k)):
                model.fit(x_train[forward_argList + list(features)], y_train)
                preds_test = model.predict(x_test[forward_argList + list(features)])
                metric = mean_absolute_error(y_test, preds_test)
                
                if metric / best_metric <= 1 - quality_improve[k-1]/100:
                    if metric/best_metric < best_dif:
                        best_dif = metric/best_metric
                        feature_to_add = list(features)
                        
            if (best_dif == 1e10):
                break
            else:
                best_metric *= best_dif
                forward_argList += feature_to_add
                print(f'Фичей в наборе: {len(forward_argList)}, MAE = {best_metric}')
    return forward_argList

def forward_selection(argList, model, x_train, y_train, x_test, y_test, max_add_features_per_iter, quality_improve,
                      max_features_to_select, n_features_to_select = -1):
    
    if max_features_to_select > len(argList):
        max_features_to_select = len(argList)
        
    if len(quality_improve) != max_add_features_per_iter:
        quality_improve = [0.4 * i for i in range(1, max_add_features_per_iter + 1)]
        
    forward_argList = []
    
    best_roc_auc = 1.e-2
    if n_features_to_select == -1:
        for k in range(1, max_add_features_per_iter + 1):
            print(f'Add {k} features per iter')
            
            while True:
              if len(forward_argList) + k > max_features_to_select:
                  return forward_argList
              
              best_diff = 0
              for features in list(combinations(set(argList) - set(forward_argList), k)):
                  model.fit(x_train[forward_argList + list(features)], y_train)
                  preds_test = model.predict_proba(x_test[forward_argList + list(features)])[:, 1]
                  roc_auc = roc_auc_score(y_test, preds_test)
                  
                  if roc_auc / best_roc_auc >= 1 + quality_improve[k-1]/100:
                      if roc_auc / best_roc_auc > best_diff:
                          best_diff = roc_auc / best_roc_auc
                          feature_to_add = list(features)
                          
              if (best_diff == 0):
                  break
              else:
                  best_roc_auc *= best_diff
                  forward_argList += feature_to_add
                  print(f'Фичей в наборе: {len(forward_argList)}, ROC AUC = {best_roc_auc}')
        return forward_argList
    else:
        print(f'Adding 1 feature per iter while n_features != {n_features_to_select}')
        
        while len(forward_argList) < n_features_to_select:
            best_roc = 0
            for feature in list(set(argList) - set(forward_argList)):
                model.fit(x_train[forward_argList + [feature]], y_train)
                probs_test = model.predict_proba(x_test[forward_argList + [feature]])[:, 1]
                roc_auc = roc_auc_score(y_test, probs_test)
                
                if roc_auc > best_roc:
                    best_roc = roc_auc
                    feature_to_add = feature
                    
            forward_argList.append(feature_to_add)
            print(f'Фичей в наборе: {len(forward_argList)}, ROC AUC = {best_roc}')
        return forward_argList

def backward_selection_regression(argList, model, x_train, y_train, x_test, y_test, 
                                  quality_loss = 0.5, n_features_to_select = -1):
    backward_argList = argList.copy()
    
    model.fit(x_train[backward_argList], y_train)
    probs_train = model.predict(x_test[backward_argList])
    metric = mean_absolute_error(y_test, probs_test)
    print('MAE Test initial = ', metric)
    
    if n_features_to_select == -1:
        while True:
            best_metric_dif = 1e10
            
            for i in backward_argList:
                model.fit(x_train[backward_argList].drop(i, axis = 1), y_train)
                probs_test = model.predict(x_test[backward_argList].drop(i, axis = 1))
                metric_new = mean_absolute_error(y_test, probs_test)
                
                if (metric_new/metric <= 1 + quality_loss/100):
                    if metric_new/metric < best_metric_dif:
                        best_metric_dif = metric_new/metric
                        perem_del = i
                        
            if best_metric_dif == 1e10:
                break
            else:
                backward_argList.remove(perem_del)
                print(f'Осталось фичей: {len(backward_argList)}, MAE = {best_metric_dif*metric}, удалена фича {perem_del}')
        return backward_argList
    else:
        while len(backward_argList) > n_features_to_select:
            best_metric = 1e10
            
            for i in backward_argList:
                model.fit(x_train[backward_argList].drop(i, axis = 1), y_train)
                probs_test = model.predict(x_test[backward_argList].drop(i, axis = 1))[:, 1]
                metric_new = mean_absolute_error(y_test, probs_test)
                
                if metric_new < best_metric:
                    best_metric = metric_new
                    perem_del = i
            backward_argList.remove(perem_del)
            
            model.fit(x_train[backward_argList], y_train)
            probs_test = model.predict(x_test[backward_argList])
            metric_new = mean_absolute_error(y_test, probs_test)
            print(f'Осталось фичей: {len(backward_argList)}, MAE = {metric_new}, удалена фича {perem_del}')
        return backward_argList
    
def backward_selection(argList, model, x_train, y_train, x_test, y_test, quality_loss = 0.5, n_features_to_select = -1):
    backward_argList = argList.copy()
    model.fit(x_train[backward_argList], y_train)
    probs_test = model.predict_proba(x_test[backward_argList])[:, 1]
    roc = roc_auc_score(y_test, probs_test)
    print('ROC AUC Test initial =', roc)
    
    if n_features_to_select == -1:
        while True:
            best_roc_dif = 0
            
            for i in backward_argList:
                model.fit(x_train[backward_argList].drop(i, axis = 1), y_train)
                probs_test = model.predict_proba(x_test[backward_argList].drop(i, axis = 1))[:, 1]
                roc_new = roc_auc_score(y_test, probs_test)
                
                if (roc_new/roc >= 1-quality_loss/100):
                    if (roc_new/roc > best_roc_dif):
                        best_roc_dif = roc_new/roc
                        perem_del = i
                        
            if (best_roc_dif == 0):
                break
            else:
                backward_argList.remove(perem_del)
                print(f'Осталось фичей: {len(backward_argList)}, ROC AUC = {best_roc_dif*roc}, удалена фича {perem_del}')
                
        return backward_argList
    else:
        while len(backward_argList) > n_features_to_select:
            best_roc = 0
            
            for i in backward_argList:
                model.fit(x_train[backward_argList].drop(i, axis = 1), y_train)
                probs_test = model.predict_proba(x_test[backward_argList].drop(i, axis = 1))[:, 1]
                roc_new = roc_auc_score(y_test, probs_test)
                
                if roc_new > best_roc:
                    best_roc = roc_new
                    perem_del = i
            backward_argList.remove(perem_del)
            model.fit(x_train[backward_argList], y_train)
            probs_test = model.predic_proba(x_test[backward_argList])[:, 1]
            roc_new = roc_auc_score(y_test, probs_test)
            print(f'Осталось фичей: {len(backward_argList)}, ROC AUC = {best_roc_dif*roc}, удалена фича {perem_del}')
        return backward_argList
    
def bidirectional_selection(argList, model, x_train, y_train, x_test, y_test, max_features_to_select):
    
    forward_argList = []
    backward_argList = argList.copy()
    iter_num = 0
    while len(backward_argList) > max_features_to_select:
        forward_argList = []
        print(f'Итерация {iter_num}')
        iter_num += 1
        while len(backward_argList) != len(forward_argList):
            best_roc_auc = 0
            feature_to_add = []
            for feature in list(set(backward_argList) - set(forward_argList)):
                model.fit(x_train[forward_argList + [feature]], y_train)
                probs_test = model.predict_proba(x_test[forward_argList + [feature]])[:, 1]
                roc_auc = roc_auc_score(y_test, probs_test)
                
                if roc_auc > best_roc_auc:
                    best_roc_auc = roc_auc
                    feature_to_add = feature
            forward_argList.append(feature_to_add)
            print(f'Фичей в forward: {len(forward_argList)}, ROC AUC = {best_roc_auc}')
            
            best_roc_auc = 0
            feature_to_remove = ''
            for feature in list(set(backward_argList) - set(forward_argList)):
                model.fit(x_train[backward_argList].drop(feature, axis = 1), y_train)
                probs_test = model.predict_proba(x_train[backward_argList].drop(feature, axis = 1))[:, 1]
                roc_auc = roc_auc_score(y_test, probs_test)
                
                if roc_auc > best_roc_auc:
                    best_roc_auc = roc_auc
                    feature_to_remove = feature
            backward_argList = list(set(backward_argList) - set([feature_to_remove]))
            print(f'Фичей в backward: {len(backward_argList)}, ROC AUC = {best_roc_auc}')
    return backward_argList

def SFFS(argList, model, x_train, y_train, x_test, y_test, quality_improve, quality_loss):
    best_argList = []
    model.fit(x_train[argList], y_train)
    probs_test = model.predict_proba(x_test[argList])[:, 1]
    roc_auc = roc_auc_score(y_test, probs_test)
    print('ROC AUC Test initial =', roc_auc)
    best_roc_auc = 0.01
    deleted_features = []
    cash_best_argList = Counter()
    
    while True:
        best_roc_dif = 0
        for feature in set(argList) - set(best_argList) - set(deleted_features):
            model.fit(x_train[best_argList + [feature]], y_train)
            probs_test = model.predict_proba(x_test[best_argList + [feature]])[:, 1]
            roc_auc = roc_auc_score(y_test, probs_test)
            if roc_auc / best_roc_auc > (1 + quality_improve/100) and roc_auc / best_roc_auc > best_roc_dif:
                best_roc_dif = roc_auc/best_roc_auc
                feature_to_add = feature
        if best_roc_dif == 0:
            break
        else:
            best_argList.append(feature_to_add)
            best_roc_auc *= best_roc_dif
            print(f'Добавлена фича: {feature_to_add}, ROC AUC = {best_roc_auc}')
            
            cash_best_argList[tuple(sorted(best_argList))] += 1
            if len(cash_best_argList) != sum(cash_best_argList.values()):
                print(f'Комбинация фичей {sorted(best_argList)} уже была получена ранее, остановка алгоритма')
                break
        if len(best_argList) > 2:
            print('Начало этапа исключения фичей')
            deleted_features = []
            while True:
                best_roc_dif = 0
                for feature in set(best_argList) - set([feature_to_add]):
                    model.fit(x_train[best_argList].drop(feature, axis = 1), y_train)
                    probs_test = model.predict_proba(x_test[best_argList].drop(feature, axis = 1))[:, 1]
                    roc_auc = roc_auc_score(y_test, probs_test)
                    if roc_auc/best_roc_auc > (1-quality_loss/100) and roc_auc/best_roc_auc > best_roc_dif:
                        best_roc_dif = roc_auc/best_roc_auc
                        feature_to_remove = feature
                if best_roc_dif == 0:
                    break
                else:
                    best_argList.remove(feature_to_remove)
                    deleted_features.append(feature_to_remove)
                    best_roc_auc *= best_roc_dif
                    print(f'Удалена фича {feature_to_remove}, ROC AUC = {best_roc_auc}')
            print('Конец этапа исключения фичей')
    return best_argList

def SFBS(argList, model, x_train, y_train, x_test, y_test, quality_improve, quality_loss):
    best_argList = argList.copy()
    model.fit(x_train[argList], y_train)
    probs_test = model.predict_proba(x_test[argList])[:, 1]
    roc = roc_auc_score(y_test, probs_test)
    print(f'ROC AUC Test initial {roc}')
    best_roc_auc = 0.01
    added_features = []
    
    cash_best_argList = Counter()
    
    while True:
        best_roc_dif = 0
        for feature in set(best_argList) - set(added_features):
            model.fit(x_train[best_argList].drop(feature, axis = 1), y_train)
            probs_test = model.predict_proba(x_test[best_argList].drop(feature, axis = 1))[:, 1]
            roc_auc = roc_auc_score(y_test, probs_test)
            if roc_auc/roc > (1-quality_loss/100) and roc_auc/roc > best_roc_dif:
                best_roc_dif = roc_auc/roc
                feature_to_remove = feature
        if best_roc_dif == 0:
            break
        else:
            best_argList.remove(feature_to_remove)
            best_roc_auc = best_roc_dif * roc
            print(f'Удалена фича {feature_to_remove}, ROC AUC = {best_roc_auc}')
            
        if len(argList) - len(best_argList) > 2:
            print('Начало этапа добавления фичей')
            while True:
                best_roc_dif = 0
                for feature in set(argList) - set(best_argList) - set([feature_to_remove]):
                    model.fit(x_train[best_argList + [feature]], y_train)
                    probs_test = model.predict_proba(x_test[best_argList + [feature]])[:, 1]
                    roc_auc = roc_auc_score(y_test, probs_test)
                    if roc_auc/best_roc_auc > (1+quality_improve/100) and roc_auc/best_roc_auc > best_roc_dif:
                        best_roc_dif = roc_auc/best_roc_auc
                        feature_to_add = feature
                if best_roc_dif == 0:
                    break
                else:
                    best_argList.append(feature_to_add)
                    added_features.append(feature_to_add)
                    best_roc_auc *= best_roc_dif
                    print(f'Добавлена фича {feature_to_add}')
            print('Конец этапа добавления фичей')
            cash_best_argList[tuple(sorted(best_argList))] += 1
            if len(cash_best_argList) != sum(cash_best_argList.values()):
                print(f'Комбинация фичей {sorted(best_argList)} уже была получена ранее, остановка алгоритма')
                break
    return best_argList