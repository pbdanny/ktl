def train_model(params):
    
    X_train, y_train, X_val, y_val = broadcast_data.value
    
    with mlflow.start_run(nested=True):
        model = BalancedRandomForestClassifier(**params, random_state=0)
        model.fit(X_train, y_train)
        
        # Get param from fitted model
        fitted_param = model.get_params()
        
        pred = model.predict(X_val)
        pred_prop = model.predict_proba(X_val)[:,1]
        
        bal_acc = balanced_accuracy_score(y_val, pred)
        fpr, tpr, thresholds = roc_curve(y_val, pred_prop)
        roc_auc = auc(fpr, tpr)
        
        rec_scr = recall_score(y_val, pred)
        f1_scr = f1_score(y_val, pred)
        avg_prc = average_precision_score(y_val, pred_prop)

        mlflow.log_metric('val_bal_acc', bal_acc)
        mlflow.log_metric('val_roc_auc', roc_auc)
        mlflow.log_metric('val_rec_scr', rec_scr)
        mlflow.log_metric('val_f1_scr', f1_scr)
        mlflow.log_metric('val_avg_prc', avg_prc)

        loss = -1.0*avg_prc
        
        return {'loss': loss, 'status': STATUS_OK}
     
class Model():
    '''
        Model class to wrap the model and provide some common functions
    '''

    def __init__(self, model, params: dict, is_classif: bool = True):
        '''
            :param:
                model: sklearn model, model to be wrapped
                params: dict, parameters for the model
                is_classif: bool, whether the model is a classifier

        '''
        from sklearn.multiclass import OneVsRestClassifier
        import logging
        if isinstance(model, OneVsRestClassifier):
            self.model = model
        else:
            self.model = model()
        try:
            self.model.set_params(**params)
        except Exception as e:
            self.model.set_params(**params)
            logging.warning(
                'The model is successfully initiated without n_jobs')
        self.is_fitted = False
        self.is_classif = is_classif

    def __pad_column(self, df, ref_cols):
        '''
            Pad the column of the dataframe to match the reference columns
            :param:
                df: pandas dataframe, dataframe to be padded
                ref_cols: list, reference columns
            :return:
                df: pandas dataframe, padded dataframe
        '''
        pad_cols = set(ref_cols) - set(df.columns)
        for col in pad_cols:
            df[col] = 0
        return df[ref_cols]

    def fit(self, X, y, class_weight=None):
        '''
            Fit the model
            :param:
                X: pandas dataframe, features
                y: pandas series, target
                class_weight: dict, class weight for classifier
        '''
        self.ref_columns = list(X.columns)
        from xgboost import XGBRegressor
        if isinstance(self.model, XGBRegressor):
            self.model.fit(X.to_numpy(), y)
        elif self.is_classif and class_weight is not None:
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weight = compute_sample_weight(class_weight, y)
            self.model.fit(X, y, sample_weight)
        else:
            self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X):
        '''
            Predict the target
            :param:
                X: pandas dataframe, features
            :return:
                y_pred: numpy array, predicted target
        '''
        from xgboost import XGBRegressor
        X = self.__pad_column(X, self.ref_columns)
        if self.is_fitted:
            if isinstance(self.model, XGBRegressor):
                y_pred = self.model.predict(X.to_numpy())
            else:
                y_pred = self.model.predict(X)
            return y_pred
        else:
            raise ValueError('Model is not fitted.')

    def predict_proba(self, X):
        '''
            Predict the target probability for classifier model 
            :param:
                X: pandas dataframe, features
            :return:
                y_pred: numpy array, predicted target probability
        '''
        X = self.__pad_column(X, self.ref_columns)
        if self.is_fitted:
            if self.is_classif:
                y_pred = self.model.predict_proba(X)
                return y_pred
            else:
                raise ValueError('Model is not a classifier.')
        else:
            raise ValueError('Model is not fitted.')

    def save(self, path):
        '''
            Save the model to disk
            :param:
                path: str, path to save the model
        '''
        import pickle
        pickle.dump(self, open(path, 'wb'))

    @classmethod
    def load(cls, path):
        '''
            Load the model from disk
            :param:
                path: str, path to load the model
            :return:
                model: Model, the loaded model
        '''
        import pickle
        return pickle.load(open(path, 'rb'))

    def get_model(self):
        '''
            Get the model
            :return:
                model: sklearn model, the model
        '''
        return self.model
