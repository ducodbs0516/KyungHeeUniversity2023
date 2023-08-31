import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

MODEL = 'LightGBM'
c = ['a','b','c','d','e']
nn=['1','5','10','15','20']
for cs in range(len(c)):
    for ns in range(len(nn)):
            
        ## 1. data load
        a ='./'+c[cs]+'/N_'+nn[ns]+'.csv'
        df=pd.read_csv(a)
        #df = df.dropna()
        X = df.iloc[:,:-1]
        y = df.iloc[:,-1]
        #print(a)

        ## 2. data preprocessing: abnormal data check (NaN, outlier check!!!)
        
        ## 3. train / test data split
        from sklearn.model_selection import train_test_split
        X_train,X_test,y_train,y_test = \
                    train_test_split(X,y,random_state=42,test_size=0.2,stratify=y)
                    
        ## 4. scaling
        from sklearn.preprocessing import StandardScaler
        ss = StandardScaler()
        ss.fit(X_train)
        X_train_std = ss.transform(X_train)
        X_test_std = ss.transform(X_test)

        ## 5. algorithm training
        import lightgbm as lgb 
        from sklearn.model_selection import GridSearchCV

        mdl = lgb.LGBMClassifier(random_state=42, device = 'gpu')
        
        params = {'max_depth':[9,11,13,15],
                'n_estimators':[500,1000,2000,3000,4000]
                }
        
        grid_mdl = GridSearchCV(mdl, 
                                param_grid=params, 
                                cv=5, 
                                refit = True) #n_jobs = -1
        
        grid_mdl.fit(X_train_std, y_train)

        print('best validation score: %.3f' %grid_mdl.best_score_)
        print(grid_mdl.best_params_)
        print(grid_mdl.best_estimator_)

        scores_df = pd.DataFrame(grid_mdl.cv_results_)
        scores_df[['params', 'mean_test_score', 'rank_test_score', 
                'split0_test_score', 'split1_test_score',
                'split2_test_score','split3_test_score','split4_test_score']]
        scores_df.to_csv('./'+ MODEL +'/'+c[cs] + '/'+ nn[ns] +'_params.csv')

        lgbm = grid_mdl.best_estimator_

        from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix
        y_train_pred = lgbm.predict(X_train_std)
        y_test_pred = lgbm.predict(X_test_std)
        y_test_prob = lgbm.predict_proba(X_test_std)

        ## 6. performance measure
        print("==========================================================")
        print("train data accuracy: %.3f" %accuracy_score(y_train, y_train_pred))
        print("test data accuracy: %.3f" %accuracy_score(y_test, y_test_pred))
        print("test data recall (macro): %.3f" %recall_score(y_test, y_test_pred, average='macro'))
        print("test data precison (macro): %.3f" %precision_score(y_test, y_test_pred, average='macro'))
        print("test data f1 score (macro): %.3f" %f1_score(y_test, y_test_pred, average='macro'))
        print("test data recall (micro): %.3f" %recall_score(y_test, y_test_pred, average='micro'))
        print("test data precison (micro): %.3f" %precision_score(y_test, y_test_pred, average='micro'))
        print("test data f1 score (micro): %.3f" %f1_score(y_test, y_test_pred, average='micro'))
        print("test data recall (weighted): %.3f" %recall_score(y_test, y_test_pred, average='weighted'))
        print("test data precison (weighted): %.3f" %precision_score(y_test, y_test_pred, average='weighted'))
        print("test data f1 score (weighted): %.3f" %f1_score(y_test, y_test_pred, average='weighted'))
        print("test data AUC: %.3f" %roc_auc_score(y_test, y_test_prob, multi_class='ovr'))
        print("test data Confusion matrix:")
        print(confusion_matrix(y_test, y_test_pred))
        print("==========================================================")


MODEL = 'XGBoost'
c = ['a','b','c','d','e']
nn=['1','5','10','15','20']
for cs in range(len(c)):
    for ns in range(len(nn)):
            
        ## 1. data load
        a ='./'+c[cs]+'/N_'+nn[ns]+'.csv'
        df=pd.read_csv(a)
        #df = df.dropna()
        X = df.iloc[:,:-1]
        y = df.iloc[:,-1]
        #print(a)

        ## 2. data preprocessing: abnormal data check (NaN, outlier check!!!)

        ## 3. train / test data split
        from sklearn.model_selection import train_test_split
        X_train,X_test,y_train,y_test = \
                    train_test_split(X,y,random_state=42,test_size=0.2,stratify=y)
                    
        ## 4. scaling
        from sklearn.preprocessing import StandardScaler
        ss = StandardScaler()
        ss.fit(X_train)
        X_train_std = ss.transform(X_train)
        X_test_std = ss.transform(X_test)
        
        ## 5. algorithm training
        from xgboost import XGBClassifier
        from sklearn.model_selection import GridSearchCV
        
        xgb = XGBClassifier(tree_method='gpu_hist', gpu_id=0, learning_rate=0.1, random_state=42)
        
        params = {'max_depth':[ 3, 5, 7, 9,11,13,15],
                 'n_estimators':[500,1000,2000,3000,4000]}

        xgb_ = GridSearchCV(xgb, param_grid=params, cv=5, refit = True)
        xgb_.fit(X_train_std, y_train)
        xgb = xgb_.best_estimator_
        print(xgb_.best_estimator_)
        print('best validation score: %.3f' %xgb_.best_score_)
        print(xgb_.best_params_)

        scores_df = pd.DataFrame(xgb_.cv_results_)

        scores_df[['params', 'mean_test_score', 'rank_test_score', 
                'split0_test_score', 'split1_test_score',
                'split2_test_score','split3_test_score','split4_test_score']]
        scores_df.to_csv('./'+ MODEL +'/'+c[cs] + '/'+ nn[ns] +'_params.csv')
        
        ## 6. performance measure
        from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix
        y_train_pred = xgb.predict(X_train_std)
        y_test_pred = xgb.predict(X_test_std)
        y_test_prob = xgb.predict_proba(X_test_std)
        print("==========================================================")
        print("train data accuracy: %.3f" %accuracy_score(y_train, y_train_pred))
        print("test data accuracy: %.3f" %accuracy_score(y_test, y_test_pred))
        print("test data recall (macro): %.3f" %recall_score(y_test, y_test_pred, average='macro'))
        print("test data precison (macro): %.3f" %precision_score(y_test, y_test_pred, average='macro'))
        print("test data f1 score (macro): %.3f" %f1_score(y_test, y_test_pred, average='macro'))
        print("test data recall (micro): %.3f" %recall_score(y_test, y_test_pred, average='micro'))
        print("test data precison (micro): %.3f" %precision_score(y_test, y_test_pred, average='micro'))
        print("test data f1 score (micro): %.3f" %f1_score(y_test, y_test_pred, average='micro'))
        print("test data recall (weighted): %.3f" %recall_score(y_test, y_test_pred, average='weighted'))
        print("test data precison (weighted): %.3f" %precision_score(y_test, y_test_pred, average='weighted'))
        print("test data f1 score (weighted): %.3f" %f1_score(y_test, y_test_pred, average='weighted'))
        print("test data AUC: %.3f" %roc_auc_score(y_test, y_test_prob, multi_class='ovr'))
        print("test data Confusion matrix:")
        print(confusion_matrix(y_test, y_test_pred))
        print("==========================================================")


MODEL = 'Random Forest'
c = ['a','b','c','d','e']
nn=['1','5','10','15','20']
for cs in range(len(c)):
    for ns in range(len(nn)):   
        ## 1. data load
        a ='./'+c[cs]+'/N_'+nn[ns]+'.csv'
        df=pd.read_csv(a)
        #df = df.dropna()
        X = df.iloc[:,:-1]
        y = df.iloc[:,-1]
        #print(a)
        
        ## 2. data preprocessing: abnormal data check (NaN, outlier check!!!)
        
        ## 3. train / test data split
        from sklearn.model_selection import train_test_split
        X_train,X_test,y_train,y_test = \
                    train_test_split(X,y,random_state=42,test_size=0.2,stratify=y)
            
        ## 4. scaling
        from sklearn.preprocessing import StandardScaler
        ss = StandardScaler()
        ss.fit(X_train)
        X_train_std = ss.transform(X_train)
        X_test_std = ss.transform(X_test)
        
        ## 5. algorithm training    
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV
        RF = RandomForestClassifier(criterion='gini',random_state=42)

        param_grid = {'n_estimators':[500, 1000, 2000, 3000, 4000]}
        grid_cv = GridSearchCV(estimator = RF,param_grid = param_grid,scoring='accuracy',cv=5,refit = True,n_jobs = -1)
        grid_cv.fit(X_train_std, y_train)

        print('best validation score: %.3f' %grid_cv.best_score_)
        print(grid_cv.best_params_)
        print(grid_cv.best_estimator_)

        bestModel = grid_cv.best_estimator_
        bestModel.fit(X_train_std, y_train)

        scores_df = pd.DataFrame(grid_cv.cv_results_)
        scores_df[['params', 'mean_test_score', 'rank_test_score', 
                'split0_test_score', 'split1_test_score',
                'split2_test_score','split3_test_score','split4_test_score']]
        scores_df.to_csv('./'+ MODEL +'/'+c[cs] + '/'+ nn[ns] +'_params.csv')
        
        ## 6. performance measure
        from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix
        y_test_pred =bestModel.predict(X_test_std)
        y_test_prob = bestModel.predict_proba(X_test_std)
        y_train_pred = bestModel.predict(X_train_std)
        y_train_prob = bestModel.predict_proba(X_train_std)
        print("==========================================================")
        print("train data accuracy: %.3f" %accuracy_score(y_train, y_train_pred))
        print("test data accuracy: %.3f" %accuracy_score(y_test, y_test_pred))
        print("test data recall (macro): %.3f" %recall_score(y_test, y_test_pred, average='macro'))
        print("test data precison (macro): %.3f" %precision_score(y_test, y_test_pred, average='macro'))
        print("test data f1 score (macro): %.3f" %f1_score(y_test, y_test_pred, average='macro'))
        print("test data recall (micro): %.3f" %recall_score(y_test, y_test_pred, average='micro'))
        print("test data precison (micro): %.3f" %precision_score(y_test, y_test_pred, average='micro'))
        print("test data f1 score (micro): %.3f" %f1_score(y_test, y_test_pred, average='micro'))
        print("test data recall (weighted): %.3f" %recall_score(y_test, y_test_pred, average='weighted'))
        print("test data precison (weighted): %.3f" %precision_score(y_test, y_test_pred, average='weighted'))
        print("test data f1 score (weighted): %.3f" %f1_score(y_test, y_test_pred, average='weighted'))
        print("test data AUC: %.3f" %roc_auc_score(y_test, y_test_prob, multi_class='ovr'))
        print("test data Confusion matrix:")
        print(confusion_matrix(y_test, y_test_pred))
        print("==========================================================")


MODEL = 'CatBoost'
c = ['a','b','c','d', 'e']
nn=['1','5','10','15','20']
for cs in range(len(c)):
    for ns in range(len(nn)):   
        ## 1. data load
        a ='./'+c[cs]+'/N_'+nn[ns]+'.csv'
        df=pd.read_csv(a)
        #df = df.dropna()
        X = df.iloc[:,:-1]
        y = df.iloc[:,-1]
        #print(a)
        
        ## 2. data preprocessing: abnormal data check (NaN, outlier check!!!)
        
        ## 3. train / test data split
        from sklearn.model_selection import train_test_split
        X_train,X_test,y_train,y_test = \
                    train_test_split(X,y,random_state=42,test_size=0.2,stratify=y)
            
        ## 4. scaling
        from sklearn.preprocessing import StandardScaler
        ss = StandardScaler()
        ss.fit(X_train)
        X_train_std = ss.transform(X_train)
        X_test_std = ss.transform(X_test)
     
        ## 5. algorithm training
        from sklearn.model_selection import GridSearchCV
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(random_state=42,iterations = 500, task_type="GPU")
        # 매개변수 그리드 설정
        param_grid = {'depth': [4,5,6,7,8,9],
                    'l2_leaf_reg': [1, 3, 5,]
                        }
        cat_ = GridSearchCV( model, param_grid=param_grid, cv=5, scoring='accuracy')

        cat_.fit(X_train_std, y_train)

        cat = cat_.best_estimator_
        print(cat_.best_estimator_)
        cat.fit(X_train_std, y_train)
        scores_df = pd.DataFrame(cat_.cv_results_)

        scores_df[['params', 'mean_test_score', 'rank_test_score', 
                'split0_test_score', 'split1_test_score',
                'split2_test_score','split3_test_score','split4_test_score']]
        scores_df.to_csv('./'+ MODEL +'/'+c[cs] + '/'+ nn[ns] +'_params.csv')
        
        ## 6. performance measure
        from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix
        y_test_pred =cat.predict(X_test_std)
        y_test_prob = cat.predict_proba(X_test_std)
        y_train_pred = cat.predict(X_train_std)
        y_train_prob = cat.predict_proba(X_train_std)
        print("==========================================================")
        print("train data accuracy: %.3f" %accuracy_score(y_train, y_train_pred))
        print("test data accuracy: %.3f" %accuracy_score(y_test, y_test_pred))
        print("test data recall (macro): %.3f" %recall_score(y_test, y_test_pred, average='macro'))
        print("test data precison (macro): %.3f" %precision_score(y_test, y_test_pred, average='macro'))
        print("test data f1 score (macro): %.3f" %f1_score(y_test, y_test_pred, average='macro'))
        print("test data recall (micro): %.3f" %recall_score(y_test, y_test_pred, average='micro'))
        print("test data precison (micro): %.3f" %precision_score(y_test, y_test_pred, average='micro'))
        print("test data f1 score (micro): %.3f" %f1_score(y_test, y_test_pred, average='micro'))
        print("test data recall (weighted): %.3f" %recall_score(y_test, y_test_pred, average='weighted'))
        print("test data precison (weighted): %.3f" %precision_score(y_test, y_test_pred, average='weighted'))
        print("test data f1 score (weighted): %.3f" %f1_score(y_test, y_test_pred, average='weighted'))
        print("test data AUC: %.3f" %roc_auc_score(y_test, y_test_prob, multi_class='ovr'))
        print("test data Confusion matrix:")
        print(confusion_matrix(y_test, y_test_pred))
        print("==========================================================")


MODEL = 'SVM'
c = ['a','b','c','d', 'e']
nn=['1','5','10','15','20']
for cs in range(len(c)):
    for ns in range(len(nn)):   
        ## 1. data load
        a ='./'+c[cs]+'/N_'+nn[ns]+'.csv'
        df=pd.read_csv(a)
        #df = df.dropna()
        X = df.iloc[:,:-1]
        y = df.iloc[:,-1]
        #print(a)
        
        ## 2. data preprocessing: abnormal data check (NaN, outlier check!!!)
        
        ## 3. train / test data split
        from sklearn.model_selection import train_test_split
        X_train,X_test,y_train,y_test = \
                    train_test_split(X,y,random_state=42,test_size=0.2,stratify=y)
            
        ## 4. scaling
        from sklearn.preprocessing import StandardScaler
        ss = StandardScaler()
        ss.fit(X_train)
        X_train_std = ss.transform(X_train)
        X_test_std = ss.transform(X_test)
     
                
        ## 5. algorithm training
        from sklearn.svm import SVC
        from sklearn.model_selection import GridSearchCV
        svm = SVC(random_state=42, C=10.0, gamma=0.01, kernel = 'rbf', probability=True)

        svm.fit(X_train_std, y_train)
        
        ## 6. performance measure
        from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix
        y_test_pred =svm.predict(X_test_std)
        y_test_prob = svm.predict_proba(X_test_std)
        y_train_pred = svm.predict(X_train_std)
        y_train_prob = svm.predict_proba(X_train_std)

        print("==========================================================")
        print("train data accuracy: %.3f" %accuracy_score(y_train, y_train_pred))
        print("test data accuracy: %.3f" %accuracy_score(y_test, y_test_pred))
        print("test data recall (macro): %.3f" %recall_score(y_test, y_test_pred, average='macro'))
        print("test data precison (macro): %.3f" %precision_score(y_test, y_test_pred, average='macro'))
        print("test data f1 score (macro): %.3f" %f1_score(y_test, y_test_pred, average='macro'))
        print("test data recall (micro): %.3f" %recall_score(y_test, y_test_pred, average='micro'))
        print("test data precison (micro): %.3f" %precision_score(y_test, y_test_pred, average='micro'))
        print("test data f1 score (micro): %.3f" %f1_score(y_test, y_test_pred, average='micro'))
        print("test data recall (weighted): %.3f" %recall_score(y_test, y_test_pred, average='weighted'))
        print("test data precison (weighted): %.3f" %precision_score(y_test, y_test_pred, average='weighted'))
        print("test data f1 score (weighted): %.3f" %f1_score(y_test, y_test_pred, average='weighted'))
        print("test data AUC: %.3f" %roc_auc_score(y_test, y_test_prob, multi_class='ovr'))
        print("test data Confusion matrix:")
        print(confusion_matrix(y_test, y_test_pred))
        print("==========================================================")


MODEL = 'Decision Tree'
c = ['a','b','c','d', 'e']
nn=['1','5','10','15','20']
for cs in range(len(c)):
    for ns in range(len(nn)):   
        ## 1. data load
        a ='./'+c[cs]+'/N_'+nn[ns]+'.csv'
        df=pd.read_csv(a)
        #df = df.dropna()
        X = df.iloc[:,:-1]
        y = df.iloc[:,-1]
        #print(a)
        
        ## 3. train / test data split
        from sklearn.model_selection import train_test_split
        X_train,X_test,y_train,y_test = \
                    train_test_split(X,y,random_state=42,test_size=0.2,stratify=y)
            
        ## 4. scaling
        from sklearn.preprocessing import StandardScaler
        ss = StandardScaler()
        ss.fit(X_train)
        X_train_std = ss.transform(X_train)
        X_test_std = ss.transform(X_test)
     
        ## 5. algorithm training
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import GridSearchCV
        dt = DecisionTreeClassifier(criterion = 'gini', random_state=42)
        parameters = { 
                'max_depth':[2,4,6,8,10,12,14,16,18,20,None]
                }   
        gr_dt = GridSearchCV(dt, param_grid=parameters, cv=5, refit=True)
                                            
        gr_dt.fit(X_train_std, y_train)

        dt = gr_dt.best_estimator_
        dt.fit(X_train_std, y_train)
        print(gr_dt.best_estimator_)
        print('best validation score: %.3f' %gr_dt.best_score_)
        print(gr_dt.best_params_)

        scores_df = pd.DataFrame(gr_dt.cv_results_)
        scores_df[['params', 'mean_test_score', 'rank_test_score', 
                'split0_test_score', 'split1_test_score',
                'split2_test_score','split3_test_score','split4_test_score']]
        scores_df.to_csv('./'+ MODEL +'/'+c[cs] + '/'+ nn[ns] +'_params.csv')
        
        ## 6. performance measure
        from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix
        y_test_pred =dt.predict(X_test_std)
        y_test_prob = dt.predict_proba(X_test_std)
        y_train_pred = dt.predict(X_train_std)
        y_train_prob = dt.predict_proba(X_train_std)
        print("==========================================================")
        print("train data accuracy: %.3f" %accuracy_score(y_train, y_train_pred))
        print("test data accuracy: %.3f" %accuracy_score(y_test, y_test_pred))
        print("test data recall (macro): %.3f" %recall_score(y_test, y_test_pred, average='macro'))
        print("test data precison (macro): %.3f" %precision_score(y_test, y_test_pred, average='macro'))
        print("test data f1 score (macro): %.3f" %f1_score(y_test, y_test_pred, average='macro'))
        print("test data recall (micro): %.3f" %recall_score(y_test, y_test_pred, average='micro'))
        print("test data precison (micro): %.3f" %precision_score(y_test, y_test_pred, average='micro'))
        print("test data f1 score (micro): %.3f" %f1_score(y_test, y_test_pred, average='micro'))
        print("test data recall (weighted): %.3f" %recall_score(y_test, y_test_pred, average='weighted'))
        print("test data precison (weighted): %.3f" %precision_score(y_test, y_test_pred, average='weighted'))
        print("test data f1 score (weighted): %.3f" %f1_score(y_test, y_test_pred, average='weighted'))
        print("test data AUC: %.3f" %roc_auc_score(y_test, y_test_prob, multi_class='ovr'))
        print("test data Confusion matrix:")
        print(confusion_matrix(y_test, y_test_pred))
        print("==========================================================")