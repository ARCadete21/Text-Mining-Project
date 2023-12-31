import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from nltk.tag import pos_tag
import math
from operator import itemgetter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
import csv
from nltk.sentiment.vader import SentimentIntensityAnalyzer


##### VISUALIZATION
def set_plot_properties(x_label, y_label, y_lim=[]):
    """
    Set properties of a plot axis.

    Args:
        ax (matplotlib.axes.Axes): The axis object of the plot.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        y_lim (list, optional): The limits for the y-axis. Defaults to [].

    Returns:
        None
    """
    plt.xlabel(x_label)  # Set the label for the x-axis
    plt.ylabel(y_label)  # Set the label for the y-axis
    if len(y_lim) != 0:
        plt.ylim(y_lim)  # Set the limits for the y-axis if provided


def plot_bar_chart(data, variable, x_label, y_label='Count', y_lim=[], legend=[], color='cadetblue', annotate=False, top=None, vertical=False):
    """
    Plot a bar chart based on the values of a variable in the given data.

    Args:
        data (pandas.DataFrame): The input data containing the variable.
        variable (str): The name of the variable to plot.
        x_label (str): The label for the x-axis.
        y_label (str, optional): The label for the y-axis. Defaults to 'Count'.
        y_lim (list, optional): The limits for the y-axis. Defaults to [].
        legend (list, optional): The legend labels. Defaults to [].
        color (str, optional): The color of the bars. Defaults to 'cadetblue'.
        annotate (bool, optional): Flag to annotate the bars with their values. Defaults to False.
        top (int or None, optional): The top value for plotting. Defaults to None.
        vertical (bool, optional): Flag to rotate x-axis labels vertically. Defaults to False.

    Returns:
        None
    """
    # Count the occurrences of each value in the variable
    counts = data[variable].value_counts()[:top] if top else data[variable].value_counts()
    x = counts.index  # Get x-axis values
    y = counts.values  # Get y-axis values

    # Plot the bar chart with specified color
    plt.bar(x, y, color=color)
    
    # Set the x-axis tick positions and labels, rotate if vertical flag is True
    plt.xticks(ticks=range(len(x)),
               labels=legend if legend else x,
               rotation=90 if vertical else 0)

    # Annotate the bars with their values if annotate flag is True
    if annotate:
        for i, v in enumerate(y):
            plt.text(i, v, str(v), ha='center', va='bottom', fontsize=12)

    set_plot_properties(x_label, y_label, y_lim) # Set plot properties using helper function


def plot_histogram(data, variable, x_label, y_label='Count', color='cadetblue', log=False):
    """
    Plot a histogram based on the values of a variable in the given data.

    Args:
        ax (matplotlib.axes.Axes): The axis object of the plot.
        data (pandas.DataFrame): The input data containing the variable.
        variable (str): The name of the variable to plot.
        x_label (str): The label for the x-axis.
        y_label (str, optional): The label for the y-axis. Defaults to 'Count'.
        color (str, optional): The color of the histogram bars. Defaults to 'cadetblue'.

    Returns:
        None
    """
    plt.hist(data[variable], bins=50, log=log, color=color)  # Plot the histogram using 50 bins

    set_plot_properties(x_label, y_label)  # Set plot properties using helper function


##### PREPROCESSING
def lower_case(text):
    return text.lower()


def expand_contractions(text):
    contractions_dict = { 
        "ain't": "is not",
        "aren't": "are not",
        "can't": "can not",
        "can't've": "can not have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so is",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": " when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": " who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"
        }

    # Regular expression pattern to find contractions
    contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

    def expand_match(contraction):
        match = contraction.group(0)
        expanded = contractions_dict.get(match)
        return expanded

    expanded_text = contractions_re.sub(expand_match, text)
    return expanded_text


# def sub_remove(text):
#     # Remove noise
#     x = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z ])|(\w+:\/\/\S+)|http.+?", " ", text, flags=re.MULTILINE)
    
#     return x

emoji_pattern = re.compile("["
    u"\U0001F600-\U0001F64F"  # Emoticons
    u"\U0001F300-\U0001F5FF"  # Symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # Transport & map symbols
    u"\U0001F700-\U0001F77F"  # Alchemical symbols
    u"\U0001F780-\U0001F7FF"  # Geometric shapes
    u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    u"\U0001FA00-\U0001FA6F"  # Chess symbols
    u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    u"\u2600-\u26FF"          # Miscellaneous Symbols
    u"\u2700-\u27BF"          # Dingbat Symbols
    "]", flags=re.UNICODE)


def sub_remove(x):
    # Remove noise, respectively...
    # email addresses
    # social media tags
    # characters that are not letters, numbers and (condionally) emojis
    # website links (both www and https)
    # html tags
    x = re.sub(r"(\b[A-Za-z0-9.%+-]+@[A-Za-z0-9.-]+.[A-Z|a-z]{2,}\b)", " ", x)
    x = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z{} ])|(\w+://\S+)|(http.+?)|(<.*?>)".format(emoji_pattern.pattern[1:-1]), 
               " ", x, flags=re.MULTILINE)
 
    return x


def sub_spaces(text):
    x = re.sub(r' +', ' ', text)
    return x


# def tokenization(lyrics):
#     tokenizer = RegexpTokenizer('\w+')
#     return tokenizer.tokenize(lyrics)


def tokenization_emojis(lyrics):

    emoji_pattern = re.compile("["
                    u"\U0001F600-\U0001F64F"  # Emoticons
                    u"\U0001F300-\U0001F5FF"  # Symbols & pictographs
                    u"\U0001F680-\U0001F6FF"  # Transport & map symbols
                    u"\U0001F700-\U0001F77F"  # Alchemical symbols                           
                    u"\U0001F780-\U0001F7FF"  # Geometric shapes
                    u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                    u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                    u"\U0001FA00-\U0001FA6F"  # Chess symbols
                    u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                    "]+", flags=re.UNICODE)

    tokenizer = RegexpTokenizer(r'\w+|' + emoji_pattern.pattern)

    return tokenizer.tokenize(lyrics)

def tokenization(lyrics):

    tokenizer = RegexpTokenizer('\w+')
    return tokenizer.tokenize(lyrics)


def lemmatization(words):
    lemmatizer = WordNetLemmatizer()

    for pos_tag in ['v', 'n', 'a', 'r']:
        words = [lemmatizer.lemmatize(word, pos=pos_tag) for word in words]

    return words


def stopwords_removal(words):
    stopwords = nltk.corpus.stopwords.words('english')
    return [word for word in words if word not in stopwords]


def join_tokens(tokens):
    return ' '.join(tokens)


# def text_preprocessing(data, text_column, target=None):
#     text_data = data[text_column].copy()
    
#     functions = [lower_case, 
#                  expand_contractions, 
#                  sub_remove, 
#                  sub_spaces]

#     if target is not None:
#         additional_functions = [tokenization,
#                                 lemmatization,
#                                 # stopwords_removal,
#                                 join_tokens]

#         functions.extend(additional_functions)

#     for function in functions:
#         text_data = text_data.apply(function)

#     text_data = pd.DataFrame(text_data, columns=[text_column])

#     if target is not None:
#         text_data[target] = data[target]
    
#     return text_data


# def text_preprocessing(data, text_column, target=None, stp_removal=False, emojis_removal=True):
#     text_data = data[text_column].copy()
    
#     functions = [lower_case, 
#                  expand_contractions, 
#                  sub_remove, 
#                  sub_spaces]

#     if target is not None:
#         if stp_removal==False:
#             additional_functions = [tokenization,
#                                     lemmatization,
#                                     join_tokens]

#         else:
#             additional_functions = [tokenization if emojis_removal else tokenization_emojis,
#                                     lemmatization,
#                                     stopwords_removal,
#                                     join_tokens]

#         functions.extend(additional_functions)

#     for function in functions:
#         text_data = text_data.apply(function)

#     text_data = pd.DataFrame(text_data, columns=[text_column])

#     if target is not None:
#         text_data[target] = data[target]
    
#     return text_data


def text_preprocessing(data, text_column, target=None, apply_stopwords_removal=False, emojis_removal=True):
    text_data = data[text_column].copy()
    
    functions = [lower_case, 
                 expand_contractions, 
                 sub_remove, 
                 sub_spaces, 
                 tokenization if emojis_removal else tokenization_emojis,
                 lemmatization,
                 stopwords_removal if apply_stopwords_removal else lambda x: x,
                 join_tokens]

    for function in functions:
        text_data = text_data.apply(function)

    text_data = pd.DataFrame(text_data, columns=[text_column])

    if target is not None:
        text_data[target] = data[target]
    
    return text_data


##### LOG RATIO

def genre_percentages(df):

    genre_percentages = (df['tag'].value_counts() / len(df))
    genre_percentages_dict = genre_percentages.to_dict()

    return genre_percentages, genre_percentages_dict

def genre_frequencies(df, genre_percentages, genre_percentages_dict):

    # Create a dictionary to store data for each genre
    genre_freqs = {}

    # Create a list to store all tokens from all genres
    all_tokens = []

    # Iterate through unique genres
    for genre in genre_percentages_dict.keys():
        # Create a subset of the DataFrame for the current genre
        genre_df = df.loc[df['tag'] == genre].drop(columns=['tag'])
        
        # Join the 'lyrics_string_fdist' column and tokenize
        genre_lyrics = ' '.join(list(genre_df['lyrics']))
        genre_tokens = word_tokenize(genre_lyrics)
        
        # Append tokens to the list for overall frequency distribution
        all_tokens.extend(genre_tokens)

        # Calculate frequency distribution for the current genre
        genre_freq = FreqDist(genre_tokens)
        
        # Store the data in the dictionary
        genre_freqs[genre] = genre_freq

    # Calculate overall frequency distribution
    overall_freq = FreqDist(all_tokens)

    # Add the overall frequency distribution to the dictionary
    genre_freqs['all'] = overall_freq
    
    # iverted order
    genre_freqs_inverted = {key: genre_freqs[key] for key in genre_percentages.keys()[::-1]}
    genre_freqs_inverted['all'] = genre_freqs['all']

    return genre_freqs, genre_freqs_inverted, overall_freq


def log_ratio(genre_freqs, genre_percentages, total_words=100, separate=False):
    # Get the overall frequency distribution for all genres
    all_freq = genre_freqs['all']

    # Create a dictionary to store the log ratios for each genre
    genre_log_ratios = {}

    # Create a set to keep track of selected words across genres
    selected_words_set = set()

    # Calculate total number of words in all genres
    total_all_words = all_freq.N()

    if not separate:

        for genre, genre_freq in genre_freqs.items():
            # Skip the overall frequency distribution
            if genre == 'all':
                continue

            # Get the percentage of representativeness for the current genre
            genre_percentage = genre_percentages.get(genre)

            # Calculate the number of words to select for the genre based on its representativeness
            words_to_select = int(round(genre_percentage * total_words))

            # Calculate the genre's top words
            genre_freq_top = genre_freq.most_common(words_to_select * 5)

            # Calculate log ratios for the top words
            log_ratios = {
                word: math.log(((freq + 1) / (genre_freq.N() + 1)) / ((all_freq[word] + 1) / (total_all_words + 1)))
                for word, freq in genre_freq_top
            }

            # Sort log ratios
            sorted_log_ratios = sorted(log_ratios.items(), key=itemgetter(1), reverse=True)

            # Select the top words that are not already selected
            selected_words = []
            for word, ratio in sorted_log_ratios:
                if word not in selected_words_set:
                    selected_words.append((word, ratio))
                    selected_words_set.add(word)

                    # Break once the required number of unique words are selected for the genre
                    if len(selected_words) == words_to_select:
                        break

            # Store the top log ratios for the genre
            genre_log_ratios[genre] = selected_words

        return genre_log_ratios
            
    else:

        for genre, genre_freq in genre_freqs.items():
            # Skip the overall frequency distribution
            if genre == 'all':
                continue

            # Equal percentages if separate is True
            genre_percentage = 1/6

            # Calculate the number of words to select for the genre based on its representativeness
            words_to_select = int(round(genre_percentage * total_words))

            # Calculate the genre's top words
            genre_freq_top = genre_freq.most_common(words_to_select * 5)

            # Calculate log ratios for the top words
            log_ratios = {
                word: math.log(((freq + 1) / (genre_freq.N() + 1)) / ((all_freq[word] + 1) / (total_all_words + 1)))
                for word, freq in genre_freq_top
            }

            # Sort log ratios
            sorted_log_ratios = sorted(log_ratios.items(), key=itemgetter(1), reverse=True)

            # Select the top words that are not already selected
            selected_words = []
            for word, ratio in sorted_log_ratios:
                selected_words.append((word, ratio))

                # Break once the required number of unique words are selected for the genre
                if len(selected_words) == words_to_select:
                    break

            # Store the top log ratios for the genre
            genre_log_ratios[genre] = selected_words

        return genre_log_ratios




def filter_lyrics(lyrics, word_set):
    return ' '.join(word for word in lyrics.split() if word in word_set)


##### VECTORIZATION
def count_vectorizer_to_df(train, words):
    # Create an instance of the CountVectorizer class - Default vectorizer does not remove stop words
    vectorizer = CountVectorizer(vocabulary=words, stop_words=None, token_pattern=r'(?u)\b\w+\b')

    # Fit the vectorizer to the text data and transform the text data into a frequency matrix
    train_frequency_matrix = vectorizer.fit_transform(train)

    # Convert the frequency matrix to a Pandas DataFrame
    train_vectorized = pd.DataFrame(train_frequency_matrix.toarray(), columns=vectorizer.get_feature_names_out(), index=train.index) #get_feature_names()

    return train_vectorized


def oh_count_vectorizer_to_df(train, words):
    # Create an instance of the CountVectorizer class - Default vectorizer does not remove stop words
    vectorizer = CountVectorizer(vocabulary=words, binary = True, stop_words=None, token_pattern=r'(?u)\b\w+\b')

    # Fit the vectorizer to the text data and transform the text data into a onehot encoded matrix
    ohe_matrix = vectorizer.fit_transform(train)

    # Convert the frequency matrix to a Pandas DataFrame
    df = pd.DataFrame(ohe_matrix.toarray(), columns=vectorizer.get_feature_names_out(), index=train.index) #get_feature_names()

    return df


def tf_idf_to_df(train):
    # Create an instance of the TdidfVectorized class - Default vectorizer does not remove stop words
    vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')

    # Fit the vectorizer to the text data and transform the text data into a frequency matrix
    frequency_matrix = vectorizer.fit_transform(train)

    # Convert the frequency matrix to a Pandas DataFrame
    df = pd.DataFrame(frequency_matrix.toarray(), columns=vectorizer.get_feature_names_out(), index=train.index) #get_feature_names()

    return df


##### MODEL EVALUATION
def model_evaluation(model, X_train, X_val, y_train, y_val):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    print('Training set:')
    print(classification_report(y_train, y_train_pred))
    print('Validation set:')
    print(classification_report(y_val, y_val_pred))


def avg_score(model, X, y, scaler=None): 
    '''
    Calculate the average F1 score for a given model using cross-validation.

    Parameters:
    -----------
    model : sklearn model object
        The model to evaluate.

    X : pandas.DataFrame
        Feature matrix.

    y : pandas.Series
        Target variable.

    scaler : Scaler object, optional (default=None)
        Scaler for feature scaling.

    Returns:
    --------
    str
        A string containing the average F1 score +/- its standard deviation for train and test sets.

    Notes:
    ------
    - Utilizes Stratified K-Fold cross-validation with 10 splits.
    - Computes F1 score for train and test sets and calculates their average and standard deviation.
    '''
    # Apply k-fold cross-validation
    skf = StratifiedKFold(n_splits=5)

    # Create lists to store the results from different folds
    score_train = []
    score_test = []

    for train_index, val_index in skf.split(X, y):
        # Get the indexes of the observations assigned for each partition
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        if scaler is not None:
            # Fit and transform scaler on training data
            scaling = scaler.fit(X_train)
            X_train = scaling.transform(X_train)
            # Transform validation data using the scaler fitted on training data
            X_val = scaling.transform(X_val)
        
        # Fit the model to the training data
        model.fit(X_train, y_train)
        
        # Calculate F1 score for train and test sets
        value_train = f1_score(y_train, model.predict(X_train), average='weighted')
        value_test = f1_score(y_val, model.predict(X_val), average='weighted')
        
        # Append the F1 scores
        score_train.append(value_train)
        score_test.append(value_test)
 
    # Calculate the average and the standard deviation for train and test F1 scores
    avg_train = round(np.mean(score_train), 3)
    avg_test = round(np.mean(score_test), 3)
    std_train = round(np.std(score_train), 3)
    std_test = round(np.std(score_test), 3)
    
    # Format and return the results as a string
    return (
        str(avg_train) + '+/-' + str(std_train),
        str(avg_test) + '+/-' + str(std_test)
    )


def show_results(model, dict, cv=False):
    for technique, data in dict.items():
        if cv:
            print(technique, 
                  avg_score(model, data[0], data[1]))
        else:
            print(technique)
            model_evaluation(model, data[0], data[1], data[2], data[3])


def grid_search(model, parameters, X, y, log=False):
    grid_search = GridSearchCV(estimator=model,
                               param_grid=parameters,
                               scoring='f1_weighted',
                               cv=2,
                               verbose=2
                            #    n_jobs=-1,
                               )

    grid_search.fit(X, y)
    best_parameters = grid_search.best_params_
    best_accuracy = grid_search.best_score_

    print('Best parameters: {}'.format(best_parameters))
    print('Best accuracy: {}'.format(best_accuracy))
    
    if log:
        with open('record.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([X.columns, model, parameters, best_accuracy])


# def random_search(model, parameters, X, y, log=False):
#     grid_search = RandomizedSearchCV(estimator=model,
#                                      param_grid=parameters,
#                                      scoring='f1_weighted',
#                                      cv=2,
#                                      verbose=2,
#                                     #  n_jobs=-1
#                                      )

#     grid_search.fit(X, y)
#     best_parameters = grid_search.best_params_
#     best_accuracy = grid_search.best_score_

#     print('Best parameters: {}'.format(best_parameters))
#     print('Best accuracy: {}'.format(best_accuracy))
    
#     if log:
#         with open('record.csv', 'a', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow([X.columns, model, parameters, best_accuracy])
            



##### SENTIMENT ANALYSIS

##### VADER
def vader_wrapper(filtered_lyrics):

    vader = SentimentIntensityAnalyzer()

    if type(filtered_lyrics) == list:

        sent_compound_list = []

        for sentence in filtered_lyrics:
            # calculates the compound sentiment score for each sentence using VADER 
            sent_compound_list.append(vader.polarity_scores(sentence)["compound"])

        polarity = np.array(sent_compound_list).mean()
        
    else:
        polarity = vader.polarity_scores(filtered_lyrics)["compound"]

    return polarity

def vader_analysis(word_set, df):

    vader = SentimentIntensityAnalyzer()

    # filter lyrics
    df['filtered_lyrics'] = df['lyrics'].apply(lambda x: filter_lyrics(x, word_set))

    # polarity lyric by lyric
    df['polarity'] = df['filtered_lyrics'].apply(lambda x: vader.polarity_scores(x)["compound"])

    # # extract the compound
    # df["polarity"] =  df["filtered_lyrics"].apply(lambda x: vader_wrapper(str(x)))

    # extract the label
    df['sentiment_label'] = np.select(
    [df['polarity'] < -0.2, df['polarity'] > 0.2],
    ['negative', 'positive'],
    default='neutral')
    
    df = df.drop("lyrics", axis=1)

    return df

def polarity_by_gender(df):

    ### mean polarity by gender
    polarity_by_tag = df.groupby('tag')['polarity'].mean().reset_index()

    ### label
    polarity_by_tag['sentiment_label'] = np.select(
        [polarity_by_tag['polarity'] < -0.2, 
        polarity_by_tag['polarity'] > 0.2],
        ['negative', 'positive'],
        default='neutral')

    return polarity_by_tag




