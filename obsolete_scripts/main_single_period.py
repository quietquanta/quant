import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm


# 1. Read daily prices and convert to monthly returns
csv_file = 'stock_adj_close_2000_2015.csv';
prices = pd.read_csv( csv_file );
#prices['Date'] = prices['Date'].apply( lambda x: datetime.datetime.strptime( x, "%Y-%m-%d" ).date() );
#prices['Month'] = prices['Date'].map( lambda x: pd.Period( x.strftime("%Y-%m"), freq='M') );
prices["Date"] = pd.to_datetime( prices["Date"] );			# convert date str to datetime
prices.set_index( 'Date', inplace = True );

monthly = prices.resample( "BM", how = "last" );
monthly_returns = monthly.pct_change().iloc[1:];


# 2. Run regression with past month's return
column_names = [ "coefficient", "lower", "upper" ];



coeff_index = [];
coeff_data = [];

for i in range( len(monthly_returns) - 1 ):
#for i in range( 5 ):
	period = monthly_returns.index[i];		# current month
	coeff_index.append( period );

	x = monthly_returns.iloc[ i, : ];		# return of current month (pandas.Series)
	X = sm.add_constant( x );				# add intercept variable (pandas.DataFrames)
	y = monthly_returns.iloc[ i+1, : ];		# return of next month (pandas.Series)

	model = sm.OLS( y, X );
	res = model.fit();						# pandas.Series

	coeff = res.params.loc[ period ];		# coefficient
	conf_int_lower = res.conf_int().loc[ period, 0 ];
	conf_int_upper = res.conf_int().loc[ period, 1 ];

	coeff_data.append( [ coeff, conf_int_lower, conf_int_upper ] );

coeff_df = pd.DataFrame( coeff_data, columns = column_names, index = coeff_index );



