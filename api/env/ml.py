from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn import datasets
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
matplotlib.use('Agg')


def split_data(X, y, test_size):
    ''' Descr: Splits data X and target variable Y into training and test sets 
        @Arg X: n-dimensional array containing predictive variable values of varying types. non-strict
        @Arg Y: 1 dimensional array containing target variable data. Values of varying types
        split
        @Arg test_size: float representing percentage of dataset to hold out for test set.
        returns: x_train, x_test, y_train, y_test'''

    return train_test_split(X, y, test_size, cv=10)


def train_random_forest(df, x_train, y_train, x_test, y_test):
    # train model
    model = RandomForestRegressor().fit(x_train, y_train)
    # make predictions
    preds = model.predict(x_test)
    # assign predicitions to new df
    predsDf = pd.DataFrame(preds, columns=["Predictions"])
    # assign true predictions to df for comparison
    predsDf["True"] = [i for i in y_test]
    train_results = model.predict(x_train)
    # np.sqrt as error is squared for negative values
    training_score = np.sqrt(mean_squared_error(train_results, y_train))
    test_score = np.sqrt(mean_squared_error(y_test, preds))
    print("Training Score: ", training_score)
    print("Testing Score: ", test_score)
    return {'model_predictions': preds, 'training_score': training_score, 'test_score': test_score}


def create_scatter(X, y):
    plt.plot(X, y, c=1)


df = pd.read_csv(
    "./temp.csv", ).dropna()
# remove any white space from colNames
df.columns = [i.strip() for i in df.columns]
# For linear regression we will need to select numeric dtype columns only for our model
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
# filters DF based on dtypes defined above
newdf = df.select_dtypes(include=numerics)
# creates X dataset excluding target variable
cols = [i for i in newdf.columns if i != arg]
X = newdf[cols]
# creates target feature
y = newdf[arg]
# create training and test set
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=323)
# assign model and fit
model = RandomForestRegressor().fit(x_train, y_train)
# make predictions
preds = model.predict(x_test)
# assign predicitions to new df
predsDf = pd.DataFrame(preds, columns=["Predictions"])
# assign true predictions to df for comparison
predsDf["True"] = [i for i in y_test]
train_results = model.predict(x_train)
plt.clf()
plt.scatter(train_results, y_train, cmap="RdBu")
plt.savefig("../src/components/Launchpad/results.png",
            facecolor="darkslategray")
training_score = np.sqrt(mean_squared_error(train_results, y_train))
test_score = np.sqrt(mean_squared_error(y_test, preds))
print("Training Score: ", training_score)
print("Testing Score: ", test_score)
