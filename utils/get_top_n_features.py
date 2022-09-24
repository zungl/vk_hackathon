def get_top_features(alg, target_name, params, x_train, y_train, x_test, y_test):
    if alg == 'XGB':
        model = XGBClassifier(n_jobs = 4)
    elif alg == 'LightGBM':
        model = lgb.LGBMClassifier(n_jobs = 4)
    else:
        cb.CatboostClassifier()
        
    model.set_params(random_state = 121, **params)
    model.fit(x_train, y_train[target_name], verbose = False)
    
    fi = pd.DataFrame({'feature': list(x_train.columns),
                       'importance': model.feature_importances_}).sort_values('importance', ascending = True)
    
    rocTest = []
    rocTrain = []
    rocPredictors = [100, 90, 80, 70, 60, 50, 45, 40, 35, 30, 25, 24, 23, 22, 21, 20,
                     19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3]
    argList = []
    
    for i in rocPredictors:
        model.fit(x_train[list(fi['feature'][-i:])], y_train[target_name])
        
        probs_test = model.predict_proba(x_test[list(fi['feature'][-i:])])[:, 1]
        rocTest.append(roc_auc_score(y_test[target_name], probs_test))
        
        probs_train = model.predict_proba(x_train[list(fi['feature'][-i:])])[:, 1]
        rocTrain.append(roc_auc_score(y_train[target_name], probs_train))
        
        argList.append(list(fi['feature'][-i:]))
        
    return (model, argList[len(rocTest) - rocTest[::-1].index(max(rocTest)) - 1],
            rocTrain[len(rocTest) - rocTest[::-1].index(max(rocTest)) - 1], max(rocTest))

# model, argList, roc_tr, roc_test = get_top_features(alg, target, params, x_train, y_train, x_test, y_test)