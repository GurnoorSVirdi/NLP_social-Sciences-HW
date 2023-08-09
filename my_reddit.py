#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import praw
import pandas as pd
import pytz
from datetime import datetime
import pickle
from utils import *
import joblib

subreddit_channel = 'politics'

with open('my_model.pk', 'rb') as f:
    data = pickle.load(f)
    
with open('my_model_1.pk', 'wb') as f:
    pickle.dump(data, f)

out_path = "/Users/gurnoorvirdi/Desktop/NLP- social sciences/Homework/Homework3/"
reddit = praw.Reddit(
     client_id="x75D-71UkPSmPGL6mCmlQQ",
     client_secret="C-eLhjAzDvdANTPQw1fEJmMdLRMAyw",
     user_agent="testscript by u/fakebot3",
     username="gurnoorvirdi1",
     password="S1ngh02$",
     check_for_async=False
 )
print(reddit.read_only)

def conv_time(var):
    tmp_df = pd.DataFrame()
    tmp_df = tmp_df.append(
        {'created_at': var},ignore_index=True)
    tmp_df.created_at = pd.to_datetime(
        tmp_df.created_at, unit='s').dt.tz_localize(
            'utc').dt.tz_convert('US/Eastern') 
    return datetime.fromtimestamp(var).astimezone(pytz.utc)

def get_reddit_data(var_in):
    import pandas as pd
    tmp_dict = pd.DataFrame()
    tmp_time = None
    try:
        tmp_dict = tmp_dict.append({"created_at": conv_time(
                                        var_in.created_utc)},
                                    ignore_index=True)
        tmp_time = tmp_dict.created_at[0] 
    except:
        print("ERROR")
        pass
    tmp_dict = {'msg_id': str(var_in.id),
                'author': str(var_in.author),
                'body': var_in.body, 'datetime': tmp_time}
    return tmp_dict


def read_pickle(path_o, name_i):
    # reads an object
    import pickle
    the_data_t = pickle.load(open(path_o + name_i + ".pk", 'rb'))
    return the_data_t



for comment in reddit.subreddit(subreddit_channel).stream.comments():
    tmp_df = get_reddit_data(comment)
    tmp_df["body_cleaned"] =  clean_text(tmp_df["body"])
    tmp_df['rem_sw_body_cleaned'] = rem_sw(tmp_df["body_cleaned"])
    tmp_df["stem_body_cleaned"] = stem_fun(tmp_df['rem_sw_body_cleaned'])

    # Load vectorizer model
    vectorizer = read_pickle(out_path, "vectorizer")
    text_xform = vectorizer.transform([tmp_df["stem_body_cleaned"]]).toarray()

    # Load PCA model
    pca_model = read_pickle(out_path, "pca")
    pca_data = pca_model.transform(text_xform)

    # Load classifier model
    classifier_model = joblib.load('my_model.pk')
    pred = classifier_model.predict(text_xform)
    pred_proba = pd.DataFrame(classifier_model.predict_proba(text_xform))
    pred_proba.columns = classifier_model.classes_
    print(pred_proba)