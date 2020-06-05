from make_data import *
from make_model import *

from sklearn.metrics import roc_auc_score


def train_model(file_name,):
    d = from_excel(file_name)

    features, target = make_data(d, 'Y')
    features = make_feature(features, target)
    
    select_best_features(features, target)
    best_features = ['p9', 'p13', 'p3', 'p5', 'p7']
    features_train, features_test, target_train, target_test = train_test_split(features[best_features], 
                                                                                target,
                                                                                test_size=.25,
                                                                                random_state=21)

    select_best_estimator(features_train, target_train,)
    n_estimators = 60
    model = make_model(features_train, target_train, n_estimators)

    return model, best_features

def predict(features_file, model, target=None,):
    features = from_excel(features_file)

    model, best_features = train_model('dataset.xlsx')
    features = features[best_features]

    predictions = model.predict(features)

    if target is not None:
        report = classification_report(target, predictions)
        auc = roc_auc_score(target, predictions)

        print(f'AUC: {auc}')
        print(f'Classification report: {report}')
    
    return predictions