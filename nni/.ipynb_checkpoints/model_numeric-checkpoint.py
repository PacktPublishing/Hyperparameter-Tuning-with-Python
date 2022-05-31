import nni
import pandas as pd
import numpy as np
import logging
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

LOG = logging.getLogger('nni_sklearn')

non_numeric_mapping = params = {
        'model__criterion': ['gini','entropy'],
        'model__class_weight': ['balanced','balanced_subsample'],
    }

def load_data():
    '''Load dataset'''
    df = pd.read_csv(f"{Path(__file__).parent.parent}/train.csv",sep=";")

    #Convert the target variable to integer
    df['y'] = df['y'].map({'yes':1,'no':0})

    #Split full data into train and test data
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=0) 

    #Get list of categorical and numerical features
    numerical_feats = list(df_train.drop(columns='y').select_dtypes(include=np.number).columns)
    categorical_feats = list(df_train.drop(columns='y').select_dtypes(exclude=np.number).columns)

    X_train = df_train.drop(columns=['y'])
    y_train = df_train['y']
    X_test = df_test.drop(columns=['y'])
    y_test = df_test['y']

    return X_train, X_test, y_train, y_test, numerical_feats, categorical_feats

def get_default_parameters():
    '''get default parameters'''
    params = {
        'model__n_estimators': 5,
        'model__criterion': 0,
        'model__class_weight': 0,
        'model__min_samples_split': 0.01,
    }

    return params

def get_model(PARAMS, numerical_feats, categorical_feats):
    '''Get model according to parameters'''

    # Initiate the Normalization Pre-processing for Numerical Features
    numeric_preprocessor = StandardScaler()

    # Initiate the One-Hot-Encoding Pre-processing for Categorical Features
    categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")

    #Create the ColumnTransformer Class to delegate each preprocessor to the corresponding features
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_preprocessor, numerical_feats),
            ("cat", categorical_preprocessor, categorical_feats),
        ]
    )

    #Create a Pipeline of preprocessor and model
    pipe = Pipeline(
        steps=[("preprocessor", preprocessor), 
               ("model", RandomForestClassifier(random_state=0))]
    )

    #Non-numeric hyperparameter mapping
    for key in non_numeric_mapping:
        PARAMS[key] = non_numeric_mapping[key][PARAMS[key]]

    #Set hyperparmeter values
    pipe = pipe.set_params(**PARAMS)

    return pipe

def run(X_train, y_train, model):
    '''Train model and predict result'''
    model.fit(X_train, y_train)
    score = np.mean(cross_val_score(model,X_train, y_train, 
                    cv=5, scoring='f1')
            )
    LOG.debug('score: %s', score)
    nni.report_final_result(score)

if __name__ == '__main__':
    X_train, _, y_train, _, numerical_feats, categorical_feats = load_data()

    try:
        # get parameters from tuner
        RECEIVED_PARAMS = nni.get_next_parameter()
        LOG.debug(RECEIVED_PARAMS)
        PARAMS = get_default_parameters()
        PARAMS.update(RECEIVED_PARAMS)
        LOG.debug(PARAMS)
        model = get_model(PARAMS, numerical_feats, categorical_feats)
        run(X_train, y_train, model)
    except Exception as exception:
        LOG.exception(exception)
        raise
