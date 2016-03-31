import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm


# 1. Read daily prices and convert to monthly returns
csv_file = 'stock_adj_close_1991_2015.csv';
prices = pd.read_csv( csv_file );
#prices['Date'] = prices['Date'].apply( lambda x: datetime.datetime.strptime( x, "%Y-%m-%d" ).date() );
#prices['Month'] = prices['Date'].map( lambda x: pd.Period( x.strftime("%Y-%m"), freq='M') );
prices["Date"] = pd.to_datetime( prices["Date"] );			# convert date str to datetime
prices.set_index( 'Date', inplace = True );

monthly = prices.resample( "BM", how = "last" );
monthly_returns = monthly.pct_change().iloc[1:];


# 2. Run regression with historical monthly return
sample_lookback = 60;					# look back 60 periods for larger sample

reg_lookback_periods = [ 1, 12 ];		# For each t, choose the return from "t-delta_t" as independent variable, where delta_t is in
										# reg_lookback_periods
reg_lookback_periods = range(1, 13 );
reg_lookback = max( reg_lookback_periods );					# look-back period for regression
reg_coeff_names = [ "coeff_0" ] + [ "coeff_%d" % x for x in reg_lookback_periods ];

coeff_index = [];
coeff_data = [];

start = reg_lookback + sample_lookback;
end = len(monthly_returns) ;

for i in range( start, end ):
	period = monthly_returns.index[i];		# current month
	coeff_index.append( period );

	# 1. Form OLS model using the returns within [t-sample_lookback, t-1]
	# 1.1 collect data: y and X
	y_arr = list();		# collect y
	X_arr = list();		# collect X
	for j in range( i-sample_lookback, i ):			# j-th row is independent variable "y"
		y_j = np.array( monthly_returns.iloc[j,:] );
		X_j_indices = [ j-_x for _x in reg_lookback_periods ];
		X_j = np.array( monthly_returns.iloc[ X_j_indices, : ]).T;

		y_arr.append( y_j );
		X_arr.append( X_j );
		
	y = np.concatenate( y_arr );
	X = np.concatenate( X_arr );
	X = sm.add_constant( X );				# add intercept

	# 1.2 regression
	model = sm.OLS( y, X );
	res = model.fit();

	coeff_data.append( res.params );


coeff_df = pd.DataFrame( coeff_data, columns = reg_coeff_names, index=coeff_index );


# Plot results
from matplotlib import pyplot as plt;

coeff_df["coeff_1"].plot();
plt.show();

