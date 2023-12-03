import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
import math
from operator import itemgetter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


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


def expand_contractions(text):
    contractions_dict = { 
        "ain't": "is not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
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
        "I'd": "I would",
        "I'd've": "I would have",
        "I'll": "I will",
        "I'll've": "I will have",
        "I'm": "I am",
        "I've": "I have",
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


def sub_remove(text):
    # Remove noise
    x = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t\n])|(\w+:\/\/\S+)|^rt|http.+?", "", text, flags=re.MULTILINE)
    
    # Replace newline and tab characters with spaces
    x = re.sub(r'[\t\n]', ' ', x)
    
    return x


def sub_spaces(text):
    x = re.sub(r' +', ' ', text)
    return(x)


# def text_preprocessing(data, text_column, target=None):
#     text_data = data[text_column].copy()
    
#     functions = [lambda x: x.lower(), 
#                  expand_contractions, 
#                  sub_remove, 
#                  sub_spaces]

#     if target is not None:
#         regexp = RegexpTokenizer('\w+')
#         wordnet_lem = WordNetLemmatizer()
#         stopwords = nltk.corpus.stopwords.words('english')
        
#         functions.extend([regexp.tokenize,
#                         #   lambda x: pos_tag(x),
#                           lambda x: [wordnet_lem.lemmatize(y) for y in x],
#                           lambda x: [item for item in x if item not in stopwords],
#                           lambda x: ' '.join(x)
#                         ])
    
#     for function in functions:
#         text_data = text_data.apply(function)

#     if target is not None:
#         text_data = pd.DataFrame(text_data, columns=[text_column])
#         text_data[target] = data[target]
    
#     return text_data


# def simplify_pos_tag(pos_tag):
#     if pos_tag.startswith('V'):
#         return 'v'  # Verbs
    
#     elif pos_tag.endswith('RB'):
#         return 'r'  # Adverbs
    
#     elif pos_tag == 'JJ':
#         return 'a'  # Adjectives
    
#     else:
#         return 'n'  # Nouns


def get_wordnet_pos(pos_tag):
    """
    Map Treebank part-of-speech tags to WordNet part-of-speech tags.
    """
    if pos_tag.startswith('V'):
        return wordnet.VERB
    
    elif 'RB' in pos_tag:
        return wordnet.ADV
    
    elif pos_tag.startswith('JJ'):
        return wordnet.ADJ
    
    else:
        return wordnet.NOUN


def lemmatize_with_mapping(words, words_map):
    lemmatizer = WordNetLemmatizer()

    lemmatized_words = []
    for word in words:
        pos = words_map.get(word)
        lemmatized_words.append(lemmatizer.lemmatize(word, pos=pos))

    return lemmatized_words


def text_preprocessing(data, text_column, target=None):
    text_data = data[text_column].copy()
    
    functions = [lambda x: x.lower(), 
                 expand_contractions, 
                 sub_remove, 
                 sub_spaces]

    for function in functions:
        text_data = text_data.apply(function)

    if target is not None:
        regexp = RegexpTokenizer('\w+')
        text_data = text_data.apply(regexp.tokenize)

        words = [word for tokens in text_data for word in tokens]
        words_unique = list(set(words))
        words_tagged = pos_tag(words_unique)
        words_pos_map = {word: get_wordnet_pos(pos_tag) for word, pos_tag in words_tagged}

        # lemmatizer = WordNetLemmatizer()
        # text_data = [
        #     [lemmatizer.lemmatize(word, pos=words_pos_map.get(word)) for word in sentence]
        #     for sentence in text_data
        #     ]
        
        # stopwords = nltk.corpus.stopwords.words('english')

        additional_functions = [lemmatize_with_mapping,
                    #   lambda x: [item for item in x if item not in stopwords],
                      lambda x: ' '.join(x)
                      ]
    
        for additional_function in additional_functions:
            text_data = text_data.apply(additional_function, args=(words_pos_map,) 
                                        if additional_function == lemmatize_with_mapping else ())

    # if target is not None:
        text_data = pd.DataFrame(text_data, columns=[text_column])
        text_data[target] = data[target]
    
    return text_data


def log_ratio(genre_freqs, genre_percentages, total_words=100):
    # Get the overall frequency distribution for all genres
    all_freq = genre_freqs['freq_all']

    # Create a dictionary to store the log ratios for each genre
    genre_log_ratios = {}

    # Create a set to keep track of selected words across genres
    selected_words_set = set()

    # Calculate total number of words in all genres
    total_all_words = all_freq.N()

    for genre, genre_freq in genre_freqs.items():
        # Skip the overall frequency distribution
        if genre == 'freq_all':
            continue

        # Get the percentage of representativeness for the current genre
        genre_percentage = genre_percentages.get(genre.split('_')[1])

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


def count_vectorizer_to_df(train, words):
    # Create an instance of the CountVectorizer class - Default vectorizer does not remove stop words
    vectorizer = CountVectorizer(vocabulary=words, stop_words=None)

    # Fit the vectorizer to the text data and transform the text data into a frequency matrix
    train_frequency_matrix = vectorizer.fit_transform(train)

    # Convert the frequency matrix to a Pandas DataFrame
    train_vectorized = pd.DataFrame(train_frequency_matrix.toarray(), columns=vectorizer.get_feature_names_out(), index=train.index)

    return train_vectorized


def oh_count_vectorizer_to_df(list_of_strings, words):
    
    # Create an instance of the CountVectorizer class - Default vectorizer does not remove stop words
    vectorizer = CountVectorizer(vocabulary=words, binary = True, stop_words=None)

    # Fit the vectorizer to the text data and transform the text data into a onehot encoded matrix
    ohe_matrix = vectorizer.fit_transform(list_of_strings)

    # Convert the frequency matrix to a Pandas DataFrame
    df = pd.DataFrame(ohe_matrix.toarray(), columns=vectorizer.get_feature_names()) #get_feature_names_out()

    return df


def tf_idf_to_df(list_of_strings):
    
    # Create an instance of the TdidfVectorized class - Default vectorizer does not remove stop words
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer to the text data and transform the text data into a frequency matrix
    frequency_matrix = vectorizer.fit_transform(list_of_strings)

    # Convert the frequency matrix to a Pandas DataFrame
    df = pd.DataFrame(frequency_matrix.toarray(), columns=vectorizer.get_feature_names()) #get_feature_names_out()

    return df
