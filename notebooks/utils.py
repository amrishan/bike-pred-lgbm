import pandas as pd
import numpy as np
import configparser
import pymssql
import sys
import glob
import shutil
import os
import re

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from IPython.display import HTML, display


def eval_metrics(actual, pred):
    
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    y_true, y_pred = np.array(actual), np.array(pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true))
    print(mape)
    return rmse, mae, r2, mape


def display_all(df):
    with pd.option_context("display.max_rows", 1000, 
                           "display.max_columns", 1000):
        display(df)


def remove_files(n, path):
    """Remove old files/directories based on
       required number.
    Args:
       n: Number of files 
    path: The path where the file/dir resides
    """
    ls_of_files = glob.glob(path+'*')
    ls_of_files.sort(key=os.path.getmtime)
    ls_rmv_files = set(ls_of_files) - set(ls_of_files[-n:])
    print(ls_of_files)
    for i in ls_rmv_files:
        if os.path.isfile(i) == True:
            os.remove(i)
        else:
            shutil.rmtree(i)
          

def add_datepart(df, fldname, drop=True, time=False, ret=False):
    """add_datepart converts a column of df from a datetime64 to many columns containing
    the information from the date. This applies changes inplace.
    Parameters:
    -----------
    df: A pandas data frame. df gain several new columns.
    fldname: A string that is the name of the date column you wish to expand.
        If it is not a datetime64 series, it will be converted to one with pd.to_datetime.
    drop: If true then the original date column will be removed.
    time: If true time features: Hour, Minute, Second will be added.
     ret: if true returns the dataframe object
    Examples:
    ---------
    >>> df = pd.DataFrame({ 'A' : pd.to_datetime(['3/11/2000', '3/12/2000', '3/13/2000'], infer_datetime_format=False) })
    >>> df
        A
    0   2000-03-11
    1   2000-03-12
    2   2000-03-13
    >>> add_datepart(df, 'A')
    >>> df
        AYear AMonth AWeek ADay ADayofweek ADayofyear AIs_month_end AIs_month_start AIs_quarter_end AIs_quarter_start AIs_year_end AIs_year_start AElapsed
    0   2000  3      10    11   5          71         False         False           False           False             False        False          952732800
    1   2000  3      10    12   6          72         False         False           False           False             False        False          952819200
    2   2000  3      11    13   0          73         False         False           False           False             False        False          952905600
    """
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64
 
    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 
            'Is_year_end',            'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if ret:
        if drop:
            return df.drop(fldname, axis=1)
        else:
            return df
    else:
        if drop: df.drop(fldname, axis=1, inplace=True)