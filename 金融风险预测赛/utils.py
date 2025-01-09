import numpy as np
from sklearn.preprocessing import MinMaxScaler


def find_outliers_by_3segama(data, fea):
    data_std = np.std(data[fea])
    data_mean = np.mean(data[fea])
    outliers_cut_off = data_std * 3
    lower_rule = data_mean - outliers_cut_off
    upper_rule = data_mean + outliers_cut_off
    data = data[(data[fea] >= lower_rule) & (data[fea] <= upper_rule)]
    # data[fea+'_outliers'] = data[fea].apply(lambda x:str('异常值') if x > upper_rule or x < lower_rule else '正常值')
    print(data.shape)
    return data


def cv_model(clf, train_x, train_y, test_x, clf_name):
    folds = 5
    seed = 2020
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    train = np.zeros(train_x.shape[0])
    test = np.zeros(test_x.shape[0])

    cv_scores = []

    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('************************************ {} ************************************'.format(str(i+1)))
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], train_y[valid_index]

        print("连续数值进行归一化")
        mms1 = MinMaxScaler(feature_range=(0, 1))
        trn_x = mms1.fit_transform(trn_x)
        val_x = mms1.transform(val_x)
        mms2 = MinMaxScaler(feature_range=(0, 1))
        test_x = mms2.fit_transform(test_x)

        if clf_name == "lgb":
            train_matrix = clf.Dataset(trn_x, label=trn_y)
            valid_matrix = clf.Dataset(val_x, label=val_y)

            # params = {
            #     'boosting_type': 'gbdt',
            #     'objective': 'binary',
            #     'metric': 'auc',
            #     'min_child_weight': 5,
            #     'num_leaves': 2 ** 5,
            #     'lambda_l2': 10,
            #     'feature_fraction': 0.8,
            #     'bagging_fraction': 0.8,
            #     'bagging_freq': 4,
            #     'learning_rate': 0.1,
            #     'seed': 2020,
            #     'nthread': 28,
            #     'n_jobs':24,
            #     'silent': True,
            #     'verbose': -1,
            # }
            params = {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': 'auc',
                'min_child_weight': 5,
                'num_leaves': 116,
                'lambda_l2': 10,
                'feature_fraction': 0.8,
                'bagging_fraction': 1.0,
                'bagging_freq': 33,
                'learning_rate': 0.1,
                'seed': 2020,
                'nthread': 28,
                'n_jobs':24,
                'silent': True,
                'verbose': -1,
            }
            model = clf.train(params, train_matrix, 1000, valid_sets=[train_matrix, valid_matrix], 
                              callbacks = [log_evaluation(period=20), early_stopping(stopping_rounds=50)])
            # model = clf.train(params, train_matrix, 50000, valid_sets=[train_matrix, valid_matrix])
            val_pred = model.predict(val_x, num_iteration=model.best_iteration)
            test_pred = model.predict(test_x, num_iteration=model.best_iteration)
            
            # print(list(sorted(zip(features, model.feature_importance("gain")), key=lambda x: x[1], reverse=True))[:20])

                
        if clf_name == "xgb":
            train_matrix = clf.DMatrix(trn_x , label=trn_y)
            valid_matrix = clf.DMatrix(val_x , label=val_y)
            if i == 0:
                # 如果重复给test更改为DMatrix，会报错。因为DMatrix不支持输入DMatrix类型的数据。
                test_x = clf.DMatrix(data=test_x)
            
            
            params = {'booster': 'gbtree',
                      'objective': 'binary:logistic',
                      'eval_metric': 'auc',
                      'gamma': 1,
                      'min_child_weight': 1.5,
                      'max_depth': 5,
                      'lambda': 10,
                      'subsample': 0.7,
                      'colsample_bytree': 0.7,
                      'colsample_bylevel': 0.7,
                      'eta': 0.04,
                      'tree_method': 'exact',
                      'seed': 2020,
                      'nthread': 36,
                      "silent": True,
                      }
            
            watchlist = [(train_matrix, 'train'),(valid_matrix, 'eval')]
            
            model = clf.train(params, train_matrix, num_boost_round=50000, evals=watchlist, verbose_eval=200, early_stopping_rounds=200)
            val_pred  = model.predict(valid_matrix)
            test_pred = model.predict(test_x)
                 
        if clf_name == "cat":
            params = {'learning_rate': 0.05, 'depth': 5, 'l2_leaf_reg': 10, 'bootstrap_type': 'Bernoulli',
                      'od_type': 'Iter', 'od_wait': 50, 'random_seed': 11, 'allow_writing_files': False}
            
            model = clf(iterations=20000, **params)
            model.fit(trn_x, trn_y, eval_set=(val_x, val_y),
                      cat_features=[], use_best_model=True, verbose=500)
            
            val_pred  = model.predict(val_x)
            test_pred = model.predict(test_x)
            
        train[valid_index] = val_pred
        test += test_pred / kf.n_splits
        cv_scores.append(roc_auc_score(val_y, val_pred))
        
        print(cv_scores)
        
    print("%s_scotrainre_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    print("%s_score_std:" % clf_name, np.std(cv_scores))
    return train, test, model

def lgb_model(x_train, y_train, x_test):
    # https://www.heywhale.com/mw/project/6585325cdcad99bb0a1f4686
    lgb_train, lgb_test, model = cv_model(lgb, x_train, y_train, x_test, "lgb")
    return lgb_train, lgb_test, model

def xgb_model(x_train, y_train, x_test):
    # https://www.cnblogs.com/Mephostopheles/p/18397154
    xgb_train, xgb_test, model = cv_model(xgb, x_train, y_train, x_test, "xgb")
    return xgb_train, xgb_test, model

def cat_model(x_train, y_train, x_test):
    # https://mp.weixin.qq.com/s/xloTLr5NJBgBspMQtxPoFA
    cat_train, cat_test, model = cv_model(CatBoostRegressor, x_train, y_train, x_test, "cat")
    return cat_train, cat_test, model