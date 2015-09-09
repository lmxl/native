__author__ = 'ymo'

import matplotlib
matplotlib.use('Agg')
import json
import htmlparser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
import cPickle as pickle
from os.path import exists
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import xgboost as xgb

def xgboost_pred(train,labels,test):
    params = {}
    params["objective"] = "binary:logistic"
    params["eta"] = 0.15
    params["min_child_weight"] = 32
    params["subsample"] = 0.7
    params["colsample_bytree"] = 1
    params["scale_pos_weight"] = 1
    params["silent"] = 1
    params["max_depth"] = 3
    params["eval_metric"] = "auc"
    params["nthread"] = 12

    plst = list(params.items())

    #Using 5000 rows for early stopping.
    offset = 8000

    num_rounds = 10000
    xgtest = xgb.DMatrix(test)

    #create a train and validation dmatrices
    xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
    xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

    #train using early stopping and predict
    #watchlist = [(xgval, 'val')]
    watchlist = [(xgtrain, 'train'),(xgval, 'val')]
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)
    preds1 = model.predict(xgtest,ntree_limit=model.best_iteration)

    return preds1

def getVector(redo = False):
    homepath = htmlparser.savejson()
    if exists(homepath + 'jar.pickle') and redo == False:
        with open(homepath + 'jar.pickle') as fp:
            print ('Loading features from cache')
            feature_train = pickle.load(fp)
            feature_test = pickle.load(fp)
            testfile = pickle.load(fp)
            labels = pickle.load(fp)
        return feature_train, feature_test,testfile, labels
    print ('Loading features from summary file')
    summarypath = homepath + 'summary.txt'
    labels = []
    text = []
    text2 = []
    testfile = []
    eof = False
    for line in open(summarypath).readlines():
        try:
            obj = json.loads(line)
            if obj['label'] >= 0 and 'failure' not in obj:
                labels.append(obj['label'])
                text.append(obj['text'])
            if obj['label'] < 0:
                if 'failure' not in obj:
                    text2.append(obj['text'])
                    testfile.append(obj['filename'])
        except:
            print('Early EOF received.')
            eof = True
            continue

    print ("Total # of valid training example: {num}".format(num=len(text)))
    #vectorizer = TfidfVectorizer(encoding='ascii', stop_words='english', lowercase=True, binary=True, use_idf=False, norm=None)
    vectorizer = TfidfVectorizer(encoding='ascii', stop_words='english', lowercase=True, sublinear_tf=True, max_df=3000, min_df=5, binary=True)
    feature_train = vectorizer.fit_transform(text)
    feature_test = vectorizer.transform(text2)
    indices = SelectKBest(chi2, k=100).fit(feature_train, labels).get_support(indices=True)
    names = vectorizer.get_feature_names()
    print [names[i] for i in indices]
    print ('Done with feature extraction')
    if not eof:
        with open(homepath + 'jar.pickle', 'w') as fp:
            pickle.dump(feature_train, fp)
            pickle.dump(feature_test, fp)
            pickle.dump(testfile, fp)
            pickle.dump(labels, fp)
    return feature_train, feature_test,testfile, labels

def train_test():
    feature_train, feature_test, testfile, labels = getVector(redo=False)

    #from sklearn.decomposition import TruncatedSVD
    #svd = TruncatedSVD(n_components=2000, random_state=42)

    #feature_train = svd.fit_transform(feature_train)
    #feature_test = svd.transform(feature_test)
    # experimental
    x_train, x_test, y_train, y_test = train_test_split(feature_train, labels, test_size=0.20, random_state=42)
    classifiers = {}
    for c in [1,5,8,10,15, 20,30,100]:
        for class_weight in ['auto', None]:
            classifiers['C={C}\tweight={weight}'.format(C=c,weight=class_weight)]= \
                LogisticRegression(penalty='l2',class_weight=class_weight,C=c)
    plt.figure()
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(x_train, y_train)
        y_score = classifier.decision_function(x_test)
        fpr, tpr, thr = roc_curve(y_test, y_score)
        area = auc(fpr, tpr)
        print '{learner}\t{area:.3f}'.format(learner=name, area=area)
        plt.plot(fpr, tpr, label='{learner}\t{area:.3f}'.format(learner=name, area=area))
        plt.legend(loc=4)
    plt.savefig('sub3.jpg')

    print ('Updating spreadsheet...')
    # production
    clf =  LogisticRegression(class_weight='auto', C=10, penalty='l2')
    clf.fit(feature_train, labels)
    y_score = clf.predict_proba(feature_test)[:,1]
    res_tb = {testfile[i]: y_score[i] for i in range(len(y_score))}

    missing = 0
    filled = 0
    fpo = open('sub.csv', 'w')
    fpi = open("sampleSubmission.csv")
    fpo.write(fpi.readline())
    for line in fpi.readlines():
        [file, sval] = line.strip().split(',')
        val = float(sval)
        if file in res_tb:
            val = res_tb[file]
            filled += 1
        else:
            missing += 1
        fpo.write('{file},{score:.3f}\n'.format(file=file, score=val))
    fpo.close()
    fpi.close()
    print('Done! Missing = {missing}/{total}'.format(missing=missing, total=filled+missing))

    print ('Updating spreadsheet2...')
    # production
    x_train_new = x_train
    x_test_new = x_test
    print ('tranformation from {shape1} to {shape2}'.format(shape1=x_train.shape, shape2=x_train_new.shape))
    y_score = xgboost_pred(x_train_new, y_train, x_test_new)
    fpr, tpr, thr = roc_curve(y_test, y_score)
    area = auc(fpr, tpr)
    print '{learner}\t{area:.3f}'.format(learner="xgboost", area=area)

    return

    y_score = clf.predict_proba(feature_test)[:,1]
    res_tb = {testfile[i]: y_score[i] for i in range(len(y_score))}
    missing = 0
    filled = 0
    fpo = open('sub.csv', 'w')
    fpi = open("sampleSubmission.csv")
    fpo.write(fpi.readline())
    for line in fpi.readlines():
        [file, sval] = line.strip().split(',')
        val = float(sval)
        if file in res_tb:
            val = res_tb[file]
            filled += 1
        else:
            missing += 1
        fpo.write('{file},{score:.3f}\n'.format(file=file, score=val))
    fpo.close()
    fpi.close()
    print('Done! Missing = {missing}/{total}'.format(missing=missing, total=filled+missing))

if __name__ == "__main__":
    train_test()