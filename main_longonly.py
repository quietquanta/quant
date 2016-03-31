import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt

from my_helper_fn import *

#------------------------------------------------------
# 1. Read data
#------------------------------------------------------
# (1) Read daily prices and convert to monthly returns
price_csv = 'stock_adj_close_1991_2015.csv';
stock_prices = read_from_csv( price_csv );
monthly_stock_prices = stock_prices.resample( "BM", how = "last" );
monthly_stock_returns = monthly_stock_prices.pct_change().iloc[1:];
# (2) Benchmark and risk-free rate
snp_500 = read_from_csv( "benchmark.csv" );
snp_500 = snp_500.resample( "BM", how = "last" );
snp_500_returns = snp_500.pct_change().iloc[1:];

rf_annualized_rate = read_from_csv( "riskfree.csv", rescale_factor = 0.01 );
def deannualization_func( annual_rate, freq="M" ):
	if freq is "M":
		return (1+annual_rate)**(1/12) - 1

rf_rate = rf_monthly_rate = rf_annualized_rate.apply( deannualization_func );


#------------------------------------------------------
# 2. Run regression with historical monthly return
#------------------------------------------------------
def regTopStocks(
	monthly_returns,
	sample_lookback = 60,			# number of periods of looking back for training data
	reg_lookback = 12,				# number of periods of lagged period for the multi-variate linear regression
	num_of_stocks = 10,
):
	"""
	Select top stocks based on the prediction of multivariate linear regression using historical data
	"""
	reg_lookback_periods = range(1, reg_lookback + 1 );	# lagged periods that are included in the multivariate linear model
	reg_coeff_names = [ "coeff_0" ] + [ "coeff_%d" % x for x in reg_lookback_periods ];	# column names for regression coefficients

	coeff_index = [];
	coeff_data = [];

	stock_pos_hist = dict();		# selected stocks (tickers) to keep for each period from start to end
	return_predict_hist = dict();	# predicted return for each portfolio of selected stocks

	start = reg_lookback + sample_lookback;		# starting row for regression
	end = len(monthly_returns) ;				# last row for regression

	for i in range( start, end-1 ):		# "end" is the last observed real return, therefore regression ends at one step before it
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
		y_indices_selected = y_rank[:num_of_stocks];			# keep only the top-performing stocks based on the prediction

		stock_selected = monthly_returns.columns.values[ y_indices_selected ];	# ticker of selected stocks
		stock_pos_hist[period] = stock_selected;			# add current predicted positions to the history of backtesting
		return_predict_hist[period] = y_predict[ y_indices_selected ];
	

	stock_pos_hist_df = pd.DataFrame( stock_pos_hist ).transpose();
	return_predict_hist_df = pd.DataFrame( return_predict_hist ).transpose();

	return stock_pos_hist_df, return_predict_hist_df;

stock_pos_hist_df, return_predict_hist_df = regTopStocks( monthly_stock_returns, reg_lookback = 12);



#------------------------------------------------------------------------
# 3. Backtesting
#------------------------------------------------------------------------
def calcStrategyReturn(
	stock_pos_hist_df,		# DataFrame of historically selected portfolio: each row is a list of stock tickers
	monthly_returns,		# monthly returns for the stock in the universe
):
	"""
	Calculate cumulative return for a given time series of stock positions
	"""
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

	return return_series, cum_return_series;

strategy_returns, cum_strategy_returns = calcStrategyReturn( stock_pos_hist_df, monthly_stock_returns );



#------------------------------------------------------------------------
# 4. Performance Analysis
#------------------------------------------------------------------------

# 4.0 Preparation
def myLinearRegression(
	y,
	X,
	include_intercept = False,
):
	"""
	Generic univariate linear regression given two return series
	"""
	if include_intercept:
		X = sm.add_constant( X );
	model = sm.OLS( y, X );
	res = model.fit();
	return res;

strategy_excess_returns = strategy_returns.sub( rf_rate.squeeze(), axis=0 ).dropna();
benchmark_returns = snp_500_returns.loc[ strategy_excess_returns.index, :];
benchmark_excess_returns = benchmark_returns.sub( rf_rate.squeeze(), axis=0 ).dropna();
#benchmark_excess_returns = benchmark_excess_returns.loc[ strategy_excess_returns.index, :];

# 4.1 CAMP
camp_result = myLinearRegression( strategy_excess_returns, benchmark_excess_returns, include_intercept = True );
alpha = camp_result.params.iloc[0];
beta = camp_result.params.iloc[1];

# 4.2 Sharpe Ratio
sharpe = strategy_excess_returns.mean() / strategy_excess_returns.std();
sharpe_annualized = sharpe * np.sqrt(12);			# annualized from monthly returns

# 4.3 Sortino Ratio
below_mean_positions = strategy_excess_returns < strategy_excess_returns.mean();
semi_std = strategy_excess_returns[ below_mean_positions ].std();
sortino = strategy_excess_returns.mean() / semi_std;

# 4.4 Information Ratio
strategy_minus_benchmark = strategy_returns.sub( benchmark_returns.squeeze(), axis=0 ).dropna();
info_ratio = strategy_minus_benchmark.mean() / strategy_minus_benchmark.std();



#------------------------------------------------------------------------
# Plot result
#------------------------------------------------------------------------
is_to_plot = True;
if is_to_plot:
	cum_strategy_returns.plot();
	snp_500.plot();
	rf_rate.plot();
	plt.show();
