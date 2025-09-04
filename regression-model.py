import os
import sys
import copy
import argparse
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results', type=str,
                        help='path to the output location for storage')
    return parser.parse_args()


args = parse_args()
warnings.filterwarnings("ignore")

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')))

if not os.path.exists(args.output):
    os.mkdir(args.output)

print(f"\ncommute time to work regression problem by comprehensive census data\n")

input, output = "data", args.output

''' preprocessing / model selection '''

print(f"\npreprocessing / model selection\n")


def read(file):
    file = os.path.join(os.path.dirname(__file__), "data", file)
    dataset = pd.read_csv(file, header=0, index_col="Id")
    return dataset
# -> csv file reader - return: pandas dataframe


development = read("development.csv")
# -> pandas dataframe

print(
    f"-> detect missing values (NaN) per column - return: {development.isna().any(axis = 0).sum()}\n")
# -> detect missing values (NaN) per column - return: 0

print(
    f"-> counts the number of duplicate rows - return: {development.duplicated().sum()}\n")
# -> counts the number of duplicate rows - return: 711

development.drop_duplicates(inplace=True)
# -> drops duplicate rows

dataset = copy.deepcopy(development)

if not os.path.exists(f"{output}/data"):
    os.mkdir(f"{output}/data")

for column in dataset.columns:
    if dataset[column].nunique() > 100:
        # quantile-based discretization function
        dataset[column] = pd.cut(x=dataset[column],
                                 bins=np.linspace(dataset[column].min(),
                                                  dataset[column].max(),
                                                  num=100), labels=False,
                                 duplicates='drop')
    sns.catplot(data=dataset, x=column,
                kind="count", palette="ch:.25")
    plt.xticks([], [])
    plt.yticks(fontsize=5)
    plt.xlabel(xlabel=column, fontsize=5)
    plt.ylabel(ylabel=None, fontsize=5)
    plt.savefig(output + f"/data/{column}.png", dpi=300)
    plt.close()
# -> plots the observations distribution for each feature of the development dataset


def ML(dataset, label):  # model selector

    x = dataset.drop(columns=[label], inplace=False)
    y = dataset[label]
    pipeline = Pipeline([
        ('normalizer', StandardScaler()),  # step 1: normalize data
        ('classifier', None)])  # step 2: classifier
    xtrain, xtest, ytrain, ytest = train_test_split(x, y,
                                                    test_size=0.25,
                                                    shuffle=True,
                                                    random_state=42)
    classifiers = [DecisionTreeRegressor(),
                   RandomForestRegressor(),
                   LinearRegression()]
    R2 = {}
    for c in classifiers:
        pipeline.set_params(classifier=c)
        scores = cross_validate(pipeline, xtrain, ytrain,
                                cv=5, scoring='r2')
        R2[c] = scores["test_score"].mean()
        print(f"{c}: {R2[c]}\n")

    return max(R2, key=R2.get)
# -> select the regression model with the best R2 score


print(
    f"-> regression model with the best R2 score: {ML(development, 'JWMNP')}\n")
# -> return: RandomForestRegressor()


def FE(dataset, label, file):  # features evaluation

    x = dataset.drop(columns=[label], inplace=False)
    y = dataset[label]
    xtrain, xtest, ytrain, ytest = train_test_split(x, y,
                                                    test_size=0.25,
                                                    shuffle=True,
                                                    random_state=42)
    regressor = RandomForestRegressor()
    regressor.fit(xtrain, ytrain)
    features = pd.DataFrame({'features': regressor.feature_names_in_,
                             'importances': regressor.feature_importances_}).sort_values(by='importances',
                                                                                         ascending=False)
    sns.barplot(data=features,
                x='importances',
                y='features',
                palette='rocket')
    sns.despine(bottom=True, left=True)
    plt.tight_layout()
    plt.title(
        'Feature importance according to the Random Forest Regressor', fontsize=5.5)
    plt.xlabel(None, fontsize=5.5)
    plt.ylabel(None, fontsize=5.5)
    plt.yticks(fontsize=5.5)
    plt.xticks([])
    for value in sns.barplot(data=features, x='importances',
                             y='features',
                             palette='rocket').containers:
        plt.bar_label(value,
                      padding=2,
                      fontsize=5.5)
    plt.savefig(output + f"/{file}.png", dpi=300)
    plt.close()
    return
# -> plots the importance of each feature according to the Random Forest Regressor model


FE(development, "JWMNP", "RFR features evaluation")
# -> return: JWAP (0.588532), JWDP (0.399011), PINCP (0.0036701), ...


def CTTW(file):  # commute time to work (file: data dictionary)

    file = os.path.join(os.path.dirname(__file__), "data", file)

    def CV(h):  # hours into minutes conversion function
        res = ''.join(filter(lambda x: x != ".", h))
        return (int(res[0:res.index(":")]) * 60) + int(res.replace(".", "")[res.index(":") + 1:])

    def TK(doc):  # tokenization function
        for index, word in enumerate(doc):
            if word == "a.m.":
                if CV(doc[index - 1]) in range(720, 780):
                    doc[index - 1] = CV(doc[index - 1].replace("12", "00"))
                else:
                    doc[index - 1] = CV(doc[index - 1])
            if word == "p.m.":
                doc[index - 1] = CV(doc[index - 1]) + 720
        return doc

    def AF(x, y):  # averaging function (standard mean computation)
        return (x + y) / 2

    with open(file, encoding='utf-8') as file:
        dictionary = file.read()

    JWDP = dictionary.split()[dictionary.split().index(
        "JWDP"):dictionary.split().index("POBP")]
    del JWDP[JWDP.index("JWDP"):JWDP.index("001")]
    doc = TK(JWDP)
    res = {}
    i = 0

    while i != 900:
        res[float(doc[i])] = [doc[i + 1], doc[i + 4]]
        i += 6  # 150 (rows) * 6 (tokens) = 900
    res = pd.DataFrame.from_dict(res, orient='index')
    for i in res.index:
        res.loc[i, "departure"] = AF(res.loc[i, 0], res.loc[i, 1])
    JWDP = res.drop(columns=[0, 1],
                    inplace=False)  # time of departure for work - hour and minute

    JWAP = dictionary.split()[dictionary.split().index(
        "JWAP"):dictionary.split().index("JWDP")]
    del JWAP[JWAP.index("JWAP"):JWAP.index("001")]
    doc = TK(JWAP)
    res = {}
    i = 0

    while i != 1710:
        res[float(doc[i])] = [doc[i + 1], doc[i + 4]]
        i += 6  # 285 (rows) * 6 (tokens) = 1710
    res = pd.DataFrame.from_dict(res, orient='index')
    for i in res.index:
        res.loc[i, "arrival"] = AF(res.loc[i, 0], res.loc[i, 1])
    JWAP = res.drop(columns=[0, 1],
                    inplace=False)  # time of arrival at work - hour and minute

    return JWDP, JWAP
# -> transforms the categorical attributes relating to the time of departure for work
# and the time of arrival at work into numerical attributes (in minutes)


departure, arrival = CTTW("data-dictionary.txt")
# -> implements the CTTW function

for i in development.index:
    development.loc[i, "CTTW"] = arrival.loc[development.loc[i, "JWAP"],
                                             "arrival"] - departure.loc[development.loc[i, "JWDP"], "departure"]
# -> implements the CTTW feature into the development dataset
# (the commute time to work feature from the computation between the JWAP and the JWDP features correlation)

FE(development, "JWMNP", "RFR features evaluation - CTTW")
# -> return: CTTW(0.993403), JWDP (0.00190481) JWAP (0.00187811), ...


def PS(dataset, features, label):  # parameters evaluation

    R2 = {}
    for feature in features:
        x = dataset[["CTTW", feature]]
        y = dataset[label]
        xtrain, xtest, ytrain, ytest = train_test_split(x, y,
                                                        test_size=0.25,
                                                        random_state=42)
        regressor = RandomForestRegressor()
        regressor.fit(xtrain, ytrain)
        R2[feature] = r2_score(ytest,
                               regressor.predict(xtest))

    return max(R2, key=R2.get)
# -> return the attribute that most increases the R2 score in correlation with the CTTW attribute


features = ["CTTW", PS(development, development.drop(
    columns=["JWMNP"]).columns, "JWMNP")]
# -> return: CTTW, JWDP

print(
    f"-> attribute that most increases the R2 score in correlation with the CTTW attribute: {features[1]}\n")

''' hyperparameters tuning '''

print(f"\nhyperparameters tuning\n")

x = development[features]
y = development["JWMNP"]

xtrain, xtest, ytrain, ytest = train_test_split(x, y,
                                                test_size=0.25,
                                                random_state=42)

parameters = {"n_estimators": np.arange(start=100,
                                        stop=250,
                                        step=5,
                                        dtype=int),
              "criterion": ["squared_error"],
              "random_state": [42],
              "n_jobs": [-1]}

gs = GridSearchCV(RandomForestRegressor(),
                  param_grid=parameters,
                  scoring="r2",
                  n_jobs=- 1,
                  cv=5)

gs.fit(xtrain, ytrain)

print(
    f"-> 25% - R2 score: {r2_score(ytest, gs.predict(xtest))} - NE: {gs.best_params_['n_estimators']}\n")

xtrain, xtest, ytrain, ytest = train_test_split(x, y,
                                                test_size=0.005, random_state=42)

parameters = {"n_estimators": np.arange(start=100,
                                        stop=250,
                                        step=5,
                                        dtype=int),
              "criterion": ["squared_error"],
              "random_state": [42],
              "n_jobs": [-1]}

gs = GridSearchCV(RandomForestRegressor(),
                  param_grid=parameters,
                  scoring="r2",
                  n_jobs=- 1,
                  cv=5)

gs.fit(xtrain, ytrain)

print(
    f"-> 0.005% - R2 score: {r2_score(ytest, gs.predict(xtest))} - NE: {gs.best_params_['n_estimators']}\n")

ax = range(ytest.index.size)

plt.plot(ax, ytest, linewidth=1, label="original")
plt.plot(ax, gs.predict(xtest), linewidth=1.05, label="predicted")
plt.title(f"R2 score ({features[0]} and {features[1]}): {r2_score(ytest, gs.predict(xtest)):.3f} - NE: {gs.best_params_['n_estimators']}",
          fontsize=5.5, fontweight='bold')
plt.legend(loc='best', fancybox=True, shadow=True)
plt.xlabel('x axis', fontsize=5)
plt.ylabel('y axis', fontsize=5)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.grid(True)
plt.savefig(output + "/R2.png", dpi=300)

''' fine tuning '''

print(f"\nfine tuning\n")

development = read("development.csv").drop_duplicates()

evaluation = read("evaluation.csv")

for dataset in [development, evaluation]:
    for i in dataset.index:
        dataset.loc[i, "CTTW"] = arrival.loc[dataset.loc[i, "JWAP"],
                                             "arrival"] - departure.loc[dataset.loc[i, "JWDP"], "departure"]

xtrain = development[features]

ytrain = development["JWMNP"]

xtest = evaluation[features]

parameters = {"n_estimators": np.arange(start=100,
                                        stop=250,
                                        step=5,
                                        dtype=int),
              "criterion": ["squared_error"],
              "random_state": [42],
              "n_jobs": [-1]}

gs = GridSearchCV(RandomForestRegressor(),
                  param_grid=parameters,
                  scoring="r2",
                  n_jobs=- 1,
                  cv=5)

gs.fit(xtrain, ytrain)

print(f"-> NE: {gs.best_params_['n_estimators']}\n")

ypred = gs.predict(xtest)

pd.DataFrame(ypred, index=evaluation.index).to_csv(
    output + "/output.csv", index_label="Id", header=["Predicted"])
