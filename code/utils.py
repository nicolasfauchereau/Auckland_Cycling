import os
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error as MAE

def median_filter(df, varname = None, window=24, std=3): 
    """
    A simple median filter, removes (i.e. replace by np.nan) observations that exceed N (default = 3) 
    tandard deviation from the median over window of length P (default = 24) centered around 
    each observation.

    Parameters
    ----------
    df : pandas.DataFrame
        The pandas.DataFrame containing the column to filter.
    varname : string
        Column to filter in the pandas.DataFrame. No default. 
    window : integer 
        Size of the window around each observation for the calculation 
        of the median and std. Default is 24 (time-steps).
    std : integer 
        Threshold for the number of std around the median to replace 
        by `np.nan`. Default is 3 (greater / less or equal).

    Returns
    -------
    dfc : pandas.Dataframe
        A copy of the pandas.DataFrame `df` with the new, filtered column `varname`
    """
    
    dfc = df.loc[:,[varname]]
    
    dfc['median']= dfc[varname].rolling(window, center=True).median()
    
    dfc['std'] = dfc[varname].rolling(window, center=True).std()
    
    dfc.loc[dfc.loc[:,varname] >= dfc['median']+std*dfc['std'], varname] = np.nan
    
    dfc.loc[dfc.loc[:,varname] <= dfc['median']-std*dfc['std'], varname] = np.nan
    
    return dfc.loc[:, varname]

def prepare_data(data, year=2017): 
    """
    prepare the data for ingestion by fbprophet: 

    see: https://facebook.github.io/prophet/docs/quick_start.html
    
    1) divide in training and test set, using the `year` parameter (int)
    
    2) reset the index and rename the `datetime` column to `ds`
    
    returns the training and test dataframes

    Parameters
    ----------
    data : pandas.DataFrame 
        The dataframe to prepare, needs to have a datetime index
    year: integer 
        The year separating the training set and the test set (includes the year)

    Returns
    -------
    data_train : pandas.DataFrame
        The training set, formatted for fbprophet.
    data_test :  pandas.Dataframe
        The test set, formatted for fbprophet.
    """
    
    
    data_train = data.loc[:str(year - 1),:]
    
    data_test = data.loc[str(year):,:]
    
    data_train.reset_index(inplace=True)
    
    data_test.reset_index(inplace=True)
    
    data_train = data_train.rename({'datetime':'ds'}, axis=1)
    
    data_test = data_test.rename({'datetime':'ds'}, axis=1)
    
    return data_train, data_test


def add_regressor(data, regressor, varname=None): 
    
    """
    adds a regressor to a `pandas.DataFrame` of target (predictand) values 
    for use in fbprophet 

    Parameters
    ----------
    data : pandas.DataFrame 
        The pandas.DataFrame in the fbprophet format (see function `prepare_data` in this package)
    regressor : pandas.DataFrame 
        A pandas.DataFrame containing the extra-regressor
    varname : string 
        The name of the column in the `regressor` DataFrame to add to the `data` DataFrame

    Returns
    -------
    verif : pandas.DataFrame
        The original `data` DataFrame with the column containing the 
        extra regressor `varname`

    """

    data_with_regressors = data.copy()
    
    data_with_regressors.loc[:,varname] = regressor.loc[:,varname]
    
    return data_with_regressors

def add_regressor_to_future(future, regressors_df): 
    """
    adds extra regressors to a `future` DataFrame dataframe created by fbprophet

    Parameters
    ----------
    data : pandas.DataFrame
        A `future` DataFrame created by the fbprophet `make_future` method  
        
    regressors_df: pandas.DataFrame 
        The pandas.DataFrame containing the regressors (with a datetime index)

    Returns
    -------
    futures : pandas.DataFrame
        The `future` DataFrame with the regressors added
    """
    
    futures = future.copy() 
    
    futures.index = pd.to_datetime(futures.ds)
    
    regressors = pd.concat(regressors_df, axis=1)
    
    futures = futures.merge(regressors, left_index=True, right_index=True)
    
    futures = futures.reset_index(drop = True)
    
    return futures


def make_verif(forecast, data_train, data_test): 
    """
    Put together the forecast (coming from fbprophet) 
    and the overved data, and set the index to be a proper datetime index, 
    for plotting

    Parameters
    ----------
    forecast : pandas.DataFrame 
        The pandas.DataFrame coming from the `forecast` method of a fbprophet 
        model. 
    
    data_train : pandas.DataFrame
        The training set, pandas.DataFrame

    data_test : pandas.DataFrame
        The training set, pandas.DataFrame
    
    Returns
    -------
    forecast : 
        The forecast DataFrane including the original observed data.

    """
    
    forecast.index = pd.to_datetime(forecast.ds)
    
    data_train.index = pd.to_datetime(data_train.ds)
    
    data_test.index = pd.to_datetime(data_test.ds)
    
    data = pd.concat([data_train, data_test], axis=0)
    
    forecast.loc[:,'y'] = data.loc[:,'y']
    
    return forecast

def plot_verif(verif, year=2017):
    """
    plots the forecasts and observed data, the `year` argument is used to visualise 
    the division between the training and test sets. 

    Parameters
    ----------
    verif : pandas.DataFrame
        The `verif` DataFrame coming from the `make_verif` function in this package

    year : integer
        The year used to separate the training and test set. Default 2017

    Returns
    -------
    f : matplotlib Figure object

    """
    
    f, ax = plt.subplots(figsize=(14, 8))
    
    train = verif.loc[:str(year - 1),:]
    
    ax.plot(train.index, train.y, 'ko', markersize=3)
    
    ax.plot(train.index, train.yhat, color='steelblue', lw=0.5)
    
    ax.fill_between(train.index, train.yhat_lower, train.yhat_upper, color='steelblue', alpha=0.3)
    
    test = verif.loc[str(year):,:]
    
    ax.plot(test.index, test.y, 'ro', markersize=3)
    
    ax.plot(test.index, test.yhat, color='coral', lw=0.5)
    
    ax.fill_between(test.index, test.yhat_lower, test.yhat_upper, color='coral', alpha=0.3)
    
    ax.axvline(str(year), color='0.8', alpha=0.7)
    
    ax.grid(ls=':', lw=0.5)
    
    return f

def plot_verif_component(verif, component='rain', year=2017): 
    """
    plots a specific component of the `verif` DataFrame

   Parameters
    ----------
    verif : pandas.DataFrame
        The `verif` DataFrame coming from the `make_verif` function in this package. 

    component : string 
        The name of the component (i.e. column name) to plot in the `verif` DataFrame. 

    year : integer
        The year used to separate the training and test set. Default 2017

    Returns
    -------
    f : matplotlib Figure object

    """
    
    f, ax = plt.subplots(figsize=(14, 7))
    
    train = verif.loc[:str(year - 1),:]
        
    ax.plot(train.index, train.loc[:,component] * 100, color='0.8', lw=1, ls='-')
    
    ax.fill_between(train.index, train.loc[:, component+'_lower'] * 100, train.loc[:, component+'_upper'] * 100, color='0.8', alpha=0.3)
    
    test = verif.loc[str(year):,:]
        
    ax.plot(test.index, test.loc[:,component] * 100, color='k', lw=1, ls='-')
    
    ax.fill_between(test.index, test.loc[:, component+'_lower'] * 100, test.loc[:, component+'_upper'] * 100, color='0.8', alpha=0.3)
    
    ax.axvline(str(year), color='k', alpha=0.7)
    
    ax.grid(ls=':', lw=0.5)
    
    return f


def plot_joint_plot(verif, x='yhat', y='y', title=None, fpath = '../figures/paper', fname = None): 
    """
    
    Parameters
    ---------- 
    verif : pandas.DataFrame 

    x : string 
        The variable on the x-axis
        Defaults to `yhat`, i.e. the forecast or estimated values.

    y : string 
        The variable on the y-axis
        Defaults to `y`, i.e. the observed values

    title : string 
        The title of the figure, default `None`. 
    
    fpath : string 
        The path to save the figures, default to `../figures/paper`

    fname : string
        The filename for the figure to be saved
        ommits the extension, the figure is saved in png, jpeg and pdf
 
    Returns
    -------
    f : matplotlib Figure object
    """

    g = sns.jointplot(x='yhat', y='y', data = verif, kind="reg", color="0.4")
    
    g.fig.set_figwidth(8)
    g.fig.set_figheight(8)

    ax = g.fig.axes[1]
    
    if title is not None: 
        ax.set_title(title, fontsize=16)

    ax = g.fig.axes[0]

    ax.set_xlim([-5, None])
    ax.set_ylim([-5, 3000])

    ax.text(100, 2500, "R = {:+4.2f}\nMAE = {:4.1f}".format(verif.loc[:,['y','yhat']].corr().iloc[0,1], MAE(verif.loc[:,'y'].values, verif.loc[:,'yhat'].values)), fontsize=16)

    ax.set_xlabel("model's estimates", fontsize=15)
    
    ax.set_ylabel("observations", fontsize=15)
    
    ax.grid(ls=':')

    [l.set_fontsize(13) for l in ax.xaxis.get_ticklabels()]
    [l.set_fontsize(13) for l in ax.yaxis.get_ticklabels()];

    ax.grid(ls=':')
    
    if fname is not None: 
        for ext in ['png','jpeg','pdf']: 
            g.fig.savefig(os.path.join(fpath, "{}.{}".format(fname, ext)), dpi=200)

