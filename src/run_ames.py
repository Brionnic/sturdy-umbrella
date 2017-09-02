import simplejson

from model import AmesModel
from data_pipeline import GetData

import numpy as np

# model functionality and validation
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

def main():
    ames_data = GetData(adjust_inflation=True)
    ames_data.fit()
    X, y = ames_data.transform()

    # because the number of model evaluations would be high make a standard
    # final evaluation split with a seed so that the models can't overfit
    # for evaluation.  Only score the models on evaluation set rarely
    X_TRAIN, X_eval, y_TRAIN, y_eval = train_test_split(X, y, random_state=42, test_size=0.2)

    # split the data for train and test like normal
    # X_train1, X_train2, X_test, y_train1, y_train2, y_test = make_data_splits(X_train, y_train)

    RMSEs = []

    for idx in range(31):
        X_train, X_test, y_train, y_test = train_test_split(X_TRAIN, y_TRAIN, test_size=0.3)

        model = AmesModel()
        model.fit(X_train, y_train)

        y_preds = model.transform(X_test)

        info = y_test.values

        # print "predicted:  actual:"
        # for idx in range(5):
        #     print np.exp(y_preds[idx]), np.exp(info[idx]), y_preds[idx], info[idx]
        # print y_preds[0:5]
        # print y_test[0:5]

        # errors = np.sum((y_preds - info)**2) / (len(y_preds) * 1.0)
        # print "Custom error:", errors

        error = np.sqrt(mean_squared_error(y_preds, y_test))

        print "RSMLE: {:2.4f}".format(error)
        RMSEs.append(error)

    boink = sorted(RMSEs)[15]
    mean = sum(RMSEs) / (len(RMSEs) * 1.0)
    print "Median RMSLE: {:2.4f} Mean RMSLE: {:2.4f}".format(boink, mean)

    with open("../data/rmsle_log.txt", "r") as in_file:
        read_file = in_file.read()

    parent_list = simplejson.loads(read_file)
    # print "parent_list", parent_list
    parent_list.append(RMSEs)
    with open("../data/rmsle_log.txt", "w") as out_file:
        out_file.write(simplejson.dumps(parent_list))

    #######################################
    ### Run Eval, do about 1 in 10 runs ###
    #######################################

    # eval_preds = model.transform(X_eval)
    # eval_error = np.sqrt(mean_squared_error(eval_preds, y_eval))
    # print
    # print "testing Eval Set: RMSLE: {:2.4f}".format(eval_error)
    #
    # print "RMSE: {}".format(np.sqrt(mean_squared_error(np.exp(y_preds), np.exp(y_test))))




def make_data_splits(X, Y):
    '''
    Take data in and split it into 3 sets for a 2 stage stacked ensemble
    return 3 X_set splits and 3 y_set splits like:
    X_train1, X_train2, X_test, y_train1, y_train2, y_test
    '''
    # split the data for train and test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)

#     print "y_train", y_train.shape
#     print "X_train", X_train.shape
#     print "y_test", y_test.shape
#     print "X_test", X_test.shape

    # subdivide again for 2 stage ensemble
    X_train1, X_train2, y_train1, y_train2 = train_test_split(X_train, y_train, test_size=0.5)

    y_train1 = y_train1.reshape(-1, 1)
    y_train2 = y_train2.reshape(-1, 1)

    return X_train1, X_train2, X_test, y_train1, y_train2, y_test

if __name__ == "__main__":
    main()
