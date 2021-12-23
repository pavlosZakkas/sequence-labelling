import sklearn_crfsuite
import scipy.stats
from data_extraction import sent2features, sent2labels, consecutive
from sklearn.model_selection import PredefinedSplit
from sklearn_crfsuite import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer

RANDOMIZED_SEARCH_ITERATIONS = 50


def train_model(X_train, y_train, c1=0.1, c2=0.1):
    print(c1, c2)
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=c1,
        c2=c2,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)
    return crf


def randomized_search(X_train, X_dev, y_train, y_dev, labels):

    # define fixed parameters and parameters to search
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        all_possible_transitions=True
    )
    params_space = {
        'c1': scipy.stats.loguniform(a=1e-7,b=1e2),
        'c2': scipy.stats.loguniform(a=1e-7,b=1e2),
    }

    # use the same metric for evaluation
    f1_scorer = make_scorer(metrics.flat_f1_score,
                            average='weighted', labels=labels)

    X = X_train + X_dev
    # Create a list where train data indices are -1 and validation data indices are 0
    indices = [-1 if x in range(len(X_train)) else 0 for x in range(len(X))]
    # Use the list to create PredefinedSplit
    pds = PredefinedSplit(test_fold=indices)

    y = y_train + y_dev

    rs = RandomizedSearchCV(
        crf, params_space,
        cv=pds,
        verbose=1,
        n_jobs=-1,
        n_iter=RANDOMIZED_SEARCH_ITERATIONS,
        scoring=f1_scorer,
        refit=False
    )
    rs.fit(X, y)

    print('best params:', rs.best_params_)
    print('best CV score:', rs.best_score_)

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=rs.best_params_['c1'],
        c2=rs.best_params_['c2'],
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)

    return crf


def get_reformatted_labels(crf):
    labels = list(crf.classes_)
    labels.remove('O')
    return labels


def sort(labels):
    return sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )


def run_baseline(train_sents, test_sents, y_train, y_test):
    print('Running baseline model')
    X_train = [sent2features(s) for s in train_sents]
    crf = train_model(X_train, y_train)

    # Model Evalutation
    labels = get_reformatted_labels(crf)
    sorted_labels = sort(labels)
    X_test = [sent2features(s) for s in test_sents]
    y_pred = crf.predict(X_test)
    print('F1 scores: \n', metrics.flat_f1_score(
        y_test, y_pred, average='weighted', labels=labels))
    print('Classification Report: \n', metrics.flat_classification_report(
        y_test, y_pred, labels=sorted_labels, digits=3))

    print('\n')
    return crf, labels



def run_baseline_with_hparam_optimization(
    train_sents, dev_sents, test_sents, y_train, y_dev, y_test, labels
):
    print('Running randomized search on baseline model..')
    X_train = [sent2features(s) for s in train_sents]
    X_dev = [sent2features(s) for s in dev_sents]

    rs_crf = randomized_search(X_train, X_dev, y_train, y_dev, labels)

    # Model Evalutation
    sorted_labels = sort(labels)
    X_test = [sent2features(s) for s in test_sents]
    y_pred = rs_crf.predict(X_test)
    print('F1 scores: \n', metrics.flat_f1_score(
        y_test, y_pred, average='weighted', labels=labels))
    print('Classification Report: \n', metrics.flat_classification_report(
        y_test, y_pred, labels=sorted_labels, digits=3))

    print('\n')
    return rs_crf


def run_extra_features_with_hparam_optimization(
    train_sents, dev_sents, test_sents, y_train, y_dev, y_test, labels, extra_features=[], search_depth=1
):
    extra_features_log = f'extra features {", ".join(extra_features)}' if len(extra_features) else 'no extra features'
    print(f'Running randomized search on model with search depth {search_depth} and {extra_features_log}')

    X_train = [sent2features(s, extra_features, search_depth)for s in train_sents]
    X_dev = [sent2features(s, extra_features, search_depth) for s in dev_sents]

    rs_crf = randomized_search(X_train, X_dev, y_train, y_dev, labels)

    # Model Evalutation
    sorted_labels = sort(labels)
    X_test = [sent2features(s, extra_features, search_depth)
              for s in test_sents]
    y_pred = rs_crf.predict(X_test)
    print('F1 scores: \n', metrics.flat_f1_score(
        y_test, y_pred, average='weighted', labels=labels))
    print('Classification Report: \n', metrics.flat_classification_report(
        y_test, y_pred, labels=sorted_labels, digits=3))

    print('\n')
    return rs_crf
