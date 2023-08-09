# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 18:34:13 2023

@author: pathouli
"""

def jd_fun(x, y):
    my_corpus_a_set = set(x.split())
    my_corpus_b_set = set(y.split())
    the_int = my_corpus_a_set.intersection(my_corpus_b_set)
    the_union = my_corpus_a_set.union(my_corpus_b_set)
    jd = len(the_int) / len(the_union)
    return jd

def clean_text(var_in):
    import re
    example_str_clean = re.sub(
        "[^A-Za-z']+", " ", var_in).strip().lower()
    return example_str_clean

def wrd_freq(var_in):
    tok_tmp = var_in.split()
    wrd_dictionary = dict()
    for word in set(tok_tmp):
        wrd_dictionary[word] = tok_tmp.count(word)
    return wrd_dictionary

def file_open(path_in):
    try:
        txt_tmp = ""
        f = open(path_in, "r", encoding="UTF8")
        txt_tmp = f.read() #this reads the entire content at once
        f.close()
    except:
        print ("cant open", path_in)
        pass
    txt_tmp = clean_text(txt_tmp)
    return txt_tmp

def fetch_data(path_in_tmp):
    import os
    import pandas as pd
    my_pd_tmp = pd.DataFrame()
    for root, dirs, files in os.walk(path_in_tmp, topdown=False):
       label = root.split("/")[-1:][0]
       for name in files:
          tmp_t = file_open(root + "/" + name)
          if len(tmp_t) != 0:
              tmp_df = pd.DataFrame({"body": tmp_t, "label": label}, index=[0])
              my_pd_tmp = pd.concat([my_pd_tmp, tmp_df], ignore_index=True)
              #my_pd_tmp = my_pd_tmp.append(
              #    {"body": tmp_t, "label": label}, ignore_index=True)
    return my_pd_tmp

def dictionary_fun(df_in):
    main_dictionary = dict()
    import collections
    for topic in df_in.label.unique():
        tmp_t = df_in[df_in.label == topic]
        tmp_t = tmp_t.body.str.cat(sep=" ")
        main_dictionary[topic] = collections.Counter(tmp_t.split())
    return main_dictionary

def count_tokens(str_in):
    return len(str_in.split())

def count_unique_tokens(str_in):
    return len(set(str_in.split()))

def write_pickle(obj_in, path_o, name_i):
    #writes an object
    import pickle
    pickle.dump(obj_in, open(path_o + name_i + ".pk", 'wb'))

def stem_fun(str_in):
    #stemming, removes affixes of words/tokens
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    stem_txt = [ps.stem(word) for word in str_in.split()]
    stem_txt = ' '.join(stem_txt)
    return stem_txt

#refactor rem_sw to filter tokens < 3 chars in length
def rem_sw(str_in):
    from nltk.corpus import stopwords
    sw = list(stopwords.words('english'))
    txt_clean = [word for word in str_in.split() if word not in sw]
    txt_clean = [word for word in txt_clean if len(word) > 2]
    # txt_clean = [word for word in str_in.split(
    #     ) if word not in sw and len(word) > 2]
    txt_clean = ' '.join(txt_clean)
    return txt_clean

def dictionary_check(str_in):
    #determine if a token exists in the dicttionary
    from nltk.corpus import words
    wrds = set(words.words())
    txt_t = [word for word in str_in.split() if word in wrds]
    txt_t = ' '.join(txt_t)
    return txt_t

def vectorize_text(df_in, m_in, n_in, path_o, name_i):
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    import pandas as pd
    if name_i == "cv":
        cv = CountVectorizer(ngram_range=(m_in, n_in))
    else:
        cv = TfidfVectorizer(ngram_range=(m_in, n_in))
    xform_data = pd.DataFrame(
        cv.fit_transform(df_in).toarray())#be careful memory intensive
    col_names = cv.get_feature_names()
    xform_data.columns = col_names
    wrd_dictionary = xform_data.sum(axis=0)
    write_pickle(cv, path_o, name_i)
    return xform_data, wrd_dictionary

def cosine_fun(df_in_a, df_in_b, label_in):
#create a function that outputs the cosine_similarity of two dataframes
    from sklearn.metrics.pairwise import cosine_similarity
    import pandas as pd
    cos_matrix = pd.DataFrame(cosine_similarity(df_in_a, df_in_a))
    cos_matrix.index = label_in
    cos_matrix.columns = label_in
    t = cos_matrix.mean(axis=0)
    t.index = label_in
    return cos_matrix, t

def pca_fun(df_in, exp_var_in, path_o, name_in):
    #turn this into a function, save off as pickle object
    from sklearn.decomposition import PCA
    import pandas as pd
    dim = PCA(n_components=exp_var_in)
    dim_data = pd.DataFrame(dim.fit_transform(df_in))
    exp_var = sum(dim.explained_variance_ratio_)
    write_pickle(dim, path_o, name_in)
    print (exp_var)
    return dim_data

def chi_fun(df_in, label_in, k_in, o_path, name_in):
    #chi-square feature selection
    from sklearn.feature_selection import chi2
    from sklearn.feature_selection import SelectKBest
    import pandas as pd
    feat_sel = SelectKBest(score_func=chi2, k=k_in)
    dim_data = pd.DataFrame(
        feat_sel.fit_transform(df_in, label_in))
    feat_index = feat_sel.get_support(indices=True)
    feature_names = df_in.columns[feat_index]
    dim_data.columns = list(feature_names)
    write_pickle(feat_sel, o_path, name_in)
    return dim_data

def extract_embeddings_pre(df_in, out_path_i, name_in):
    #https://code.google.com/archive/p/word2vec/
    #https://pypi.org/project/gensim/
    #pip install gensim
    import pandas as pd
    from nltk.data import find
    from gensim.models import KeyedVectors
    import pickle
    def get_score(var):
        import numpy as np
        tmp_arr = list()
        for word in var:
            try:
                tmp_arr.append(list(my_model_t.get_vector(word)))
            except:
                pass
        tmp_arr
        return np.mean(np.array(tmp_arr), axis=0)
    word2vec_sample = str(find(name_in))
    my_model_t = KeyedVectors.load_word2vec_format(
        word2vec_sample, binary=False)
    # word_dict = my_model.key_to_index
    tmp_out = df_in.str.split().apply(get_score)
    tmp_data = tmp_out.apply(pd.Series).fillna(0)
    pickle.dump(my_model_t, open(out_path_i + "embeddings.pkl", "wb"))
    pickle.dump(tmp_data, open(out_path_i + "embeddings_df.pkl", "wb" ))
    return tmp_data, my_model_t

def domain_train(df_in, path_in, name_in):
    #domain specific
    import pandas as pd
    import gensim
    def get_score(var):
        import numpy as np
        tmp_arr = list()
        for word in var:
            try:
                tmp_arr.append(list(model.wv.get_vector(word)))
            except:
                pass
        tmp_arr
        return np.mean(np.array(tmp_arr), axis=0)
    model = gensim.models.Word2Vec(df_in.str.split())
    model.save(path_in + 'body.embedding')
    #call up the model
    #load_model = gensim.models.Word2Vec.load('body.embedding')
    model.wv.similarity('fish','river')
    tmp_data = pd.DataFrame(df_in.str.split().apply(get_score))
    return tmp_data, model

def model_train(df_in, l_in, t_size, sw_in, v_n_in, o_path):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    import pandas as pd
    X_train, X_test, y_train, y_test = train_test_split(
        df_in, l_in, test_size=t_size, random_state=123)
    if sw_in == "rf":
        m = RandomForestClassifier(random_state=123)
    elif sw_in == "gnb":
        m = GaussianNB()
    elif sw_in == "gbc":
        m = GradientBoostingClassifier()
    m.fit(X_train, y_train)
    write_pickle(m, o_path, sw_in)
    try:
        vec_tmp = read_pickle(o_path, v_n_in)
        feat_names = vec_tmp.get_feature_names()
        fi = pd.DataFrame(m.feature_importances_)
        fi.index = feat_names
        fi.columns = ["fi_score"]
        fi.sort_values(by=['fi_score'], ascending=False
                       ).to_csv(o_path + "fi.csv")
        tmp = fi[fi.fi_score > 0.0]
        t_perc = round((len(tmp) / len(fi)*100), 2)
        print (t_perc, "of the data has propensity")
    except:
        print ("Sorry model doesn't support feature importance")
        pass
    return m, X_test, y_test

def model_perf(model_in, test_in, label_in):
    #model evaluation/peformance metrics
    import pandas as pd
    from sklearn.metrics import precision_recall_fscore_support
    y_pred = pd.DataFrame(model_in.predict(test_in))
    y_pred_post_prob = pd.DataFrame(
        model_in.predict_proba(test_in))
    y_pred_post_prob.columns = model_in.classes_
    y_pred.index = label_in
    y_pred_post_prob["true"] = y_pred.index
    
    metrics = pd.DataFrame(precision_recall_fscore_support(
        label_in, y_pred, average='weighted'))
    metrics.index = ["precision", "recall", "f_Score", None]
    print (metrics)

def grid_fun(df_in, label_in, grid_in, cv_in, sw, path_in):
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    if sw == "rf":
        m = RandomForestClassifier(random_state=123)
    elif sw == "gnb":
        m = GaussianNB()
    elif sw == "gbc":
        m = GradientBoostingClassifier()
    elif sw == "lr":
        m = LogisticRegression()
    cv = GridSearchCV(m, grid_in, cv=cv_in, n_jobs=-1)
    cv.fit(df_in, label_in)
    perf = cv.best_score_
    optimal_params = cv.best_params_
    print (perf)
    print (optimal_params)
    if sw == "rf":
        m = RandomForestClassifier(**cv.best_params_, random_state=123)
    elif sw == "gnb":
        m = GaussianNB(**cv.best_params_)
    elif sw == "gbc":
        m = GradientBoostingClassifier(**cv.best_params_)
    elif sw == "lr":
        m = LogisticRegression(**cv.best_params_)
    m.fit(df_in, label_in)
    write_pickle(m, path_in, sw)
    return m