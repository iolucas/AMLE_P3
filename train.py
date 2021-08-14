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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier

from azureml.core import Workspace, Experiment
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset-name', required=True, help="Name of the dataset ito be used.")
    parser.add_argument('--output-dir', required=True, type=str, help="Output directory where the trained model will be saved.")
    
    parser.add_argument("--use-stopwords", required=True, action="store_true", help="Use this option if you want to preprocess the text using stopwords.")
    parser.add_argument("--use-binary-count", required=True, action="store_true", help="Use this option if you want to use binary counts of words presence.")
    parser.add_argument("--ngram-range", required=True, type=int, help="Max number of ngrams to use")

    args = parser.parse_args()
    
    
    run = Run.get_context()

    run.log("Stopwords: ", str(args.use_stopwords))
    run.log("Binary count: ", str(args.use_binary_count))
    run.log("Ngram range: ", str(args.ngram_range))

    ws = Workspace.from_config()

    #dataset = dataset = ws.datasets[args.dataset_name]
    dataset = TabularDatasetFactory.from_delimited_files(path="https://github.com/iolucas/AMLE_P3/raw/main/amazon_reviews.csv").to_pandas_dataframe() 

    #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=38)
    
    
    model = make_pipeline(
        CountVectorizer(
            strip_accents="unicode", 
            stop_words=lambda: "english" if args.use_stopwords else None, 
            ngram_range=(1, args.ngram_range), 
            #max_df=1.0, 
            #min_df=1, 
            binary=args.use_binary_count
        ),
        #GradientBoostingClassifier()  
        LogisticRegression()
    )
    
    model.fit(train_data["review"], train_data["rating"])
    
    accuracy = model.score(test_data["review"], test_data["rating"])
    run.log("Accuracy", np.float(accuracy))
    
    model_name = "model_" + "_".join([
           args.use_stopwords,
           args.use_binary_count,
           args.ngram_range
    ]) + ".pkl"
    
    #Save model for further reference
    #model_name = "model_lr_" + str(args.C) + "_" + str(args.max_iter) + ".pkl"
    #filename = "outputs/" + model_name
    filename = os.path.join(args.output_dir, model_name)
    
    joblib.dump(value=model, filename=filename)
    run.upload_file(name=model_name, path_or_stream=filename)

if __name__ == '__main__':
    main()
