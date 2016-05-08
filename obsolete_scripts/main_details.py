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
reg_lookback = 12;						# look-back period for regression
reg_lookback_periods = [ 1, 12 ];		# For each t, choose the return from "t-delta_t" as independent variable, where delta_t is in
										# reg_lookback_periods
reg_lookback_periods = range(1, 13 );	# alternative lookback_periods
reg_coeff_names = [ "coeff_0" ] + [ "coeff_%d" % x for x in reg_lookback_periods ];

coeff_index = [];
coeff_data = [];

start = reg_lookback + sample_lookback;
end = len(monthly_returns) ;

stock_pos_hist = dict();		# indices of stocks to keep for each period from start to end
return_predict_hist = dict();
portfolio_size = 10;			# number of stocks to keep for each period

# start regression from time "i"
# start prediction from time "i+1"

for i in range( start, end-1 ):
#for i in range( start, start+1):			# for testing only
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

	# 1.3 Use fitted model res at time "i" to predict the return on time "i+1"
	X_new_indices = [ i+1-x for x in reg_lookback_periods ];
	X_new = monthly_returns.iloc[X_new_indices, :];			# p x N matrix, where N is number of stocks in the universe
	X_new = X_new.transpose();		
	X_new = sm.add_constant( X_new );						# prepend constant before historical returns
	y_predict = res.predict( X_new );
	y_rank = y_predict.argsort()[::-1];						# rank of y_predict in descending order (i.e from Max to Min)
	y_indices_selected = y_rank[:portfolio_size];			# keep only the top-performing stocks based on the prediction

	stock_selected = monthly_returns.columns.values[ y_indices_selected ];	# ticker of selected stocks
	stock_pos_hist[period] = stock_selected;			# add current predicted positions to the history of backtesting
	return_predict_hist[period] = y_predict[ y_indices_selected ];
	

stock_pos_hist_df = pd.DataFrame( stock_pos_hist ).transpose();
return_predict_hist_df = pd.DataFrame( return_predict_hist ).transpose();

#3. Evaluate performance
start_value = 1;		# Assume starting with 1 unit of asset (e.g. $1 million)

return_seq = [];
period_seq = [];
for i in range( len(stock_pos_hist_df) - 1 ):		# for each selected position at step "i", get its return at step "i+1"
	period = stock_pos_hist_df.index[i+1];
	period_seq.append( period );					# Add period to sequence

	curr_stock_list = list( stock_pos_hist_df.iloc[i, :] );
	curr_stock_returns = monthly_returns.loc[ period, curr_stock_list ];
	port_return = curr_stock_returns.mean()
	return_seq.append( port_return );
	
return_series = pd.Series( return_seq, period_seq );
cum_return_series = ( return_series + 1 ).cumprod();




# Get S&P benchmark
snp_500 = pd.read_csv( "benchmark.csv" );
snp_500.set_index( "Date", inplace = True );
snp_500.index = pd.to_datetime( snp_500.index );


from matplotlib import pyplot as plt
is_to_plot = True;
if is_to_plot:
	cum_return_series.plot();
	snp_500.plot();
	plt.show();
