import matplotlib.pyplot as plt
import re
import nltk
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

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


def sub_remove(text):
    # Remove noise
    x = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t\n])|(\w+:\/\/\S+)|^rt|http.+?", "", text, flags=re.MULTILINE)
    
    # Replace newline and tab characters with spaces
    x = x.replace(r'[\t\n]', ' ')
    
    return x


def sub_spaces(text):
    x = re.sub(r' +', ' ', text)
    return(x)
