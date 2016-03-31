import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm


"""
# Read daily prices and convert to monthly price
csv_file = 'stock_adj_close_2010_2015.csv';
prices = pd.read_csv( csv_file );
prices['Date'] = prices['Date'].apply( lambda x: datetime.datetime.strptime( x, "%Y-%m-%d" ).date() );
#prices.set_index( 'Date', inplace=True );
#prices.index = pd.to_datetime( prices.index )		# convert index to DateTimeIndex for time series functionality

prices['Month'] = prices['Date'].map( lambda x: pd.Period( x.strftime("%Y-%m"), freq='M') );

monthly_prices = prices.groupby('Month').last();
monthly_prices.to_csv('stock_monthly_prices.csv' );
"""

# Read monthly prices
csv_file = 'stock_monthly_prices.csv';
prices = pd.read_csv( csv_file );

prices.index = prices["Month"].map( lambda x: pd.Period(x, freq='M') );
prices.drop( ['Month', "Date" ], axis = 1, inplace = True );

returns = prices.diff();
returns = returns.iloc[1:,:];
n_returns = len( returns );

# Iterate through all periods
param_arr = []
for i in range(1,n_returns):


