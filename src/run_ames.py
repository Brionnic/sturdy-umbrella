import simplejson

from model import AmesModel
from data_pipeline import GetData

import numpy as np
import matplotlib.pyplot as plt

# model functionality and validation
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

def main():
    # for models exept LinearRegression types
    ames_data = GetData(log_price=True, adjust_inflation=True, one_hots=True)
    # To allow LinearRegression to converge
    # ames_data = GetData(log_price=True, adjust_inflation=True, one_hots=False)
    ames_data.fit()
    X, y = ames_data.transform()

    # because the number of model evaluations would be high make a standard
    # final evaluation split with a seed so that the models can't overfit
    # for evaluation.  Only score the models on evaluation set rarely
    X_TRAIN, X_eval, y_TRAIN, y_eval = train_test_split(X, y, random_state=42, test_size=0.2)

    # split the data for train and test like normal
    # X_train1, X_train2, X_test, y_train1, y_train2, y_test = make_data_splits(X_train, y_train)

    # import ipdb; ipdb.set_trace()

    RMSEs = []

    runs = 31

    for idx in range(runs):
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

        # print_coefs(model, X)

        # if the error is big then it blew up in the predictions
        # for some reason
        # if error > 1.0:
        #     plot_residuals(y_preds, y_test, X_test, model)

        print "RSMLE: {:2.4f}".format(error)
        RMSEs.append(error)

    boink = sorted(RMSEs)[15]
    mean = sum(RMSEs) / (len(RMSEs) * 1.0)
    print "Median RMSLE: {:2.4f} Mean RMSLE: {:2.4f}".format(boink, mean)

    # with open("../data/rmsle_log.txt", "r") as in_file:
    #     read_file = in_file.read()
    #
    # parent_list = simplejson.loads(read_file)
    # # print "parent_list", parent_list
    # parent_list.append(RMSEs)
    # with open("../data/rmsle_log.txt", "w") as out_file:
    #     out_file.write(simplejson.dumps(parent_list))

    #######################################
    ### Run Eval, do about 1 in 10 runs ###
    #######################################

    # eval_preds = model.transform(X_eval)
    # eval_error = np.sqrt(mean_squared_error(eval_preds, y_eval))
    # print
    # print "testing Eval Set: RMSLE: {:2.4f}".format(eval_error)
    #
    # print "RMSE: {}".format(np.sqrt(mean_squared_error(np.exp(y_preds), np.exp(y_test))))

def print_coefs(model, X):
    '''
    Try to print out the coefficients in an interpretable manner
    '''
    coefs = list(model.model_1.coef_)
    keys = X.keys()

    for idx, key in enumerate(keys):
        print "Feature: {:<20} Coef:{:4.4f}".format(key, coefs[idx])

def plot_residuals(y_preds, y_actuals, X, model):
    '''
    Try to troubleshoot why linear regression is exploding
    '''
    y_actuals = np.array(y_actuals)

    # perfect predictioon would be 0
    residuals = y_preds - y_actuals

    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(y_actuals, y_preds, alpha=0.4, s=100, color="blue")
    ax.set_title("Difference in Predicted Price vs Actual Price (log)")
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Error in Predicted Price")
    # plt.show()

    print "                          len residuals:", len(residuals), "len preds:", len(y_preds), "len actuals:", len(y_actuals)
    print "                          type y actuals:", type(y_actuals)
    _max = residuals.max()
    print "                          max:", _max
    print "                          argmax (index?):", residuals.argmax()

    index = list(residuals).index(_max)
    print "                          index:", index
    print "                          y_actuals:", y_actuals[index]

    print "                          X?:", X.iloc[index, :]
    # print "PID:", X.iloc[index, "PID"]

    # print "###########################"
    # print X.keys()
    # keys = X.keys()
    #
    # print "basement poor", X.loc[:,"bsmt_qual_po"].describe()

    # keys = [u'Bedroom AbvGr', u'TotRms AbvGrd', u'Garage Cars', u'Overall Qual',
    #    u'Lot Area', u'Lot Frontage', u'Pool Area', u'Fireplaces', u'home_sf',
    #    u'bsmt_qual_ex', u'bsmt_qual_fa', u'bsmt_qual_gd', u'bsmt_qual_po',
    #    u'bsmt_qual_ta', u'bsmt_qual_nan', u'total_baths', u'bed_to_room_ratio',
    #    u'yrs_since_update', u'oc_1.0', u'oc_2.0', u'oc_3.0', u'oc_4.0',
    #    u'oc_5.0', u'oc_6.0', u'oc_7.0', u'oc_8.0', u'oc_9.0', u'oc_nan',
    #    u'yrs_since_sold']

    # for idx, key in enumerate(keys):
    #     print key, X.iloc[_max_pos, idx]

    # residuals = sorted(residuals)
    # print "res 0:10", residuals[0:10]
    # print
    # print
    # print "res last 10", residuals[::-10]

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
