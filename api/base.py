from flask import Flask, request, jsonify, render_template
import pandas as pd
from flask_cors import CORS
import os
# from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn import datasets
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from ml import *
matplotlib.use('Agg')


api = Flask(__name__)
CORS(api)


@api.route('/data', methods=['POST'])
def my_profile():
    # pointer to uploaded file
    uploaded_file = request.files['file']
    # makes sure file exists
    if uploaded_file.filename != '':
        # sets outgoing file path
        file_path = os.path.join(
            "./", 'temp.csv')
       # saves file
        uploaded_file.save(file_path)
        # create dataframe
        df = pd.read_csv(
            "./temp.csv")
    # return df.to_json(orient="records")
        return jsonify({'table': df.head(20).to_html(), 'col_names': df.columns.values.tolist()})


@api.route('/inspect/<arg>', methods=['POST'])
def inspectData(arg):
    df = pd.read_csv(
        "./temp.csv", )
    df.columns = [i.strip() for i in df.columns]
    sns.heatmap(pd.DataFrame(round(df.corr(), 2)),
                annot=True, cmap="RdBu", vmin=-1, vmax=1)
    plt.savefig("../src/components/Launchpad/heatmap.png",
                facecolor="#212529")
    plt.clf()
    sns.distplot(df[arg])
    plt.savefig("../src/components/Launchpad/distplot.png",
                facecolor="#212529")
    plt.clf()
    cols = [i for i in df.columns if i != arg]
    print(df[cols])
    sns.regplot(df[cols[0]], df[arg])
    plt.savefig("../src/components/Launchpad/regplot.png",
                facecolor="#212529")
    return jsonify({'corr': pd.DataFrame(round(df.corr(), 2)).to_html(), 'summary': pd.DataFrame(df[arg]).describe(include="all").to_html()})


@ api.route('/plot', methods=['POST'])
def plot():
    df = pd.read_csv(
        "./temp.csv", )
    return df.plot()


@ api.route('/train/<arg>', methods=['GET'])
def train(arg):
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
    x_train, x_test, y_train, y_test = split_data(X, y, test_size=0.33)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=323)
    # assign model and fit
    model = RandomForestRegressor(
        max_depth=5, n_estimators=500).fit(x_train, y_train)
    # make predictions
    preds = model.predict(x_test)
    # assign predicitions to new df
    predsDf = pd.DataFrame(preds, columns=["Predictions"])
    # assign true predictions to df for comparison
    predsDf["True"] = [i for i in y_test]
    train_results = model.predict(x_train)
    # train_random_forest(newdf, x_train, y_train, x_test, y_test)
    plt.clf()
    plt.plot(np.array(train_results), 'b.', c=None)
    print(len(train_results))
    print(len(y_train))
    plt.plot(np.array(y_train), 'r.')
    plt.savefig("../src/components/Launchpad/results.png",
                facecolor="darkslategray")
    training_score = np.sqrt(mean_squared_error(train_results, y_train))
    test_score = np.sqrt(mean_squared_error(y_test, preds))
    print("Training Score: ", training_score)
    print("Testing Score: ", test_score)
    return jsonify({"training_score": training_score, "test_score": test_score, "sample": predsDf.iloc[:10].to_json()})


if __name__ == '__main__':
    api.run(debug=True)
