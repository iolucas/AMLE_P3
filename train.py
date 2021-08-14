
import argparse
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline

from azureml.core import Workspace, Experiment
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

ws = Workspace.from_config()


ds = TabularDatasetFactory.from_delimited_files(path="https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv")

x, y = clean_data(ds)

#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
x_train, x_test, y_train, y_test = train_test_split(x,y)

run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)
    
    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    
    #Save model for further reference
    model_name = "model_lr_" + str(args.C) + "_" + str(args.max_iter) + ".pkl"
    filename = "outputs/" + model_name
    
    joblib.dump(value=model, filename=filename)
    run.upload_file(name=model_name, path_or_stream=filename)

if __name__ == '__main__':
    main()