from data_extraction import sent2labels
from parser import read_file
import models
import nltk

import matplotlib.pyplot as plt

# Download pos tagger
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

TRAIN_PATH = './W-NUT_data/wnut17train.conll'
DEV_PATH = './W-NUT_data/emerging.dev.conll'
TEST_PATH = './W-NUT_data/emerging.test.annotated'

EXTRA_FEATURES = ['url', 'emoji', 'tag', 'hashtag', 'consecutive']

'''
For the creation of train, test and dev sentences sets
'''
def create_set_sent(doc):
    return [[word.to_tuple() for word in sent.words] for sent in doc.sentences]


def run_models_with_optimization(train_sents, dev_sents, test_sents, y_train, y_dev, y_test):
    models.run_baseline_with_hparam_optimization(
        train_sents, dev_sents, test_sents, y_train, y_dev, y_test, labels)

    models.run_extra_features_with_hparam_optimization(
        train_sents, dev_sents, test_sents, y_train, y_dev, y_test, labels,
        extra_features=[],
        search_depth=2,
    )

    models.run_extra_features_with_hparam_optimization(
        train_sents, dev_sents, test_sents, y_train, y_dev, y_test, labels,
        extra_features=[],
        search_depth=3,
    )

    for extra_feature in EXTRA_FEATURES:
        models.run_extra_features_with_hparam_optimization(
            train_sents, dev_sents, test_sents, y_train, y_dev, y_test, labels,
            extra_features=[extra_feature],
            search_depth=1,
        )

    models.run_extra_features_with_hparam_optimization(
        train_sents, dev_sents, test_sents, y_train, y_dev, y_test, labels,
        extra_features=EXTRA_FEATURES,
        search_depth=1,
    )

if __name__ == "__main__":
    # Reading the IOB files
    doc_train, doc_dev, doc_test = [
        read_file(path) for path in [TRAIN_PATH, DEV_PATH, TEST_PATH]]

    # Creating the respective sets
    train_sents, dev_sents, test_sents = [create_set_sent(doc)
                                          for doc in [doc_train, doc_dev, doc_test]]

    # Parsing labels
    y_train = [sent2labels(s) for s in train_sents]
    y_dev = [sent2labels(s) for s in dev_sents]
    y_test = [sent2labels(s) for s in test_sents]

    # Running models
    crf, labels = models.run_baseline(train_sents, test_sents, y_train, y_test)
    run_models_with_optimization(train_sents, dev_sents, test_sents, y_train, y_dev, y_test)        