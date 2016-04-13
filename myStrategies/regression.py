import numpy as np;
import pandas as pd;
import statsmodels.api as sm;

from performance_analysis import calcCAMP, calcRatioGeneric;
from strategies import RegressionStrategy;

class RegressionLongShort( RegressionStrategy ):
	"""	Cross-sectional equity long-short strategy based on autoregression using historical pricing data.
	"""
	def __init__( self, 
			prices, 					# Dataframe of stock price histories for all stocks in the universe
			riskfree_rate,				# Series of Riskfree Rate, e.g. Riskfree Rate
			benchmark_returns,			# Series of Benchmark returns, e.g. S&P 500
			resample_freq = "BM",
			sample_lookback = 60,			# number of periods of looking back for training data
			reg_lags_and_weights = { 1:1 },			# lags for autoregression, and the weight for each lag
			num_longs = 10,					# number of stocks to long for each period
			num_shorts = 10,				# number of stocks to short for each period
	):

		self.prices = prices.resample( "BM" ).last();
		self.returns = self.prices.pct_change().iloc[1:];

		self.riskfree_rate = riskfree_rate;
		self.benchmark_returns = benchmark_returns;

		# Regression parameters
		self.sample_lookback = sample_lookback;
		self.reg_lags_and_weights = reg_lags_and_weights;

		self.num_longs = num_longs;
		self.num_shorts = num_shorts;

		self.backtest_finished = False;

	def init_summary( self ):
		"""	Return a summary of the strategy
		"""
		print "Prices:\n", self.prices.iloc[:5,:5], "\n";
		print "Returns:\n", self.returns.iloc[:5,:5], "\n";

		print "Regression facts:\n";
		print "\tsample_lookback = %d" % self.sample_lookback;
		print "\tregression lags =", self.reg_lags_and_weights.keys();
		print "\tregression lag weights =", self.reg_lags_and_weights.values();

		print "Position Settings:\n"
		print "\tNumber of Longs \t= %d"%self.num_longs;
		print "\tNumber of Shorts \t= %d"%self.num_shorts;

	#------------------------------------------------------------
	# Backtest 
	#------------------------------------------------------------
	def BackTest( self ):
		"""	Go through history and calculate
		(1) Historical positions (Long/Short) as a result of the strategy
		(2) Predicted return for each stock in the universe
		"""
		if self.backtest_finished:					# if backtest has been done, return result directly
			return self.backtest_result;

		# Calculate historical positions
		self._CalcHistoricalPositions();

		# Simulation
		period_seq = [];
		long_return_seq = [];			# return sequence of longed positions
		short_return_seq = [];			# return sequence of shorted positions
		return_seq = [];				# return sequence of overall portfolio

		for i in range( len( self.long_pos_hist_df) - 1 ):		# for each selected position at step "i", get its return at step "i+1"
			period = self.long_pos_hist_df.index[i+1];
			period_seq.append( period );					# Add period to sequence

			long_stock_list = list( self.long_pos_hist_df.iloc[i,:] );
			short_stock_list = list( self.short_pos_hist_df.iloc[i,:] );

			long_returns = self.returns.loc[ period, long_stock_list ];
			short_returns = self.returns.loc[ period, short_stock_list ];		# series of NaN if short_returns is empty

			long_ave_return = long_returns.mean();
			if len(short_returns) == 0:
				short_weight = 0.
				short_ave_return = 0;
				port_return = long_ave_return;
			else:
				short_weight = 0.5;				# fraction of portfolio that's in short positions
				short_ave_return = short_returns.mean();			# return NaN if empty series
				port_return = (1 - short_weight ) * long_ave_return - short_weight * short_ave_return;

			long_return_seq.append( long_ave_return );
			short_return_seq.append( -short_ave_return );	# short position return is the negative of stock returns
			return_seq.append( port_return );

		overall_return_series = pd.Series( return_seq, period_seq );
		long_return_series = pd.Series( long_return_seq, period_seq );
		short_return_series = pd.Series( short_return_seq, period_seq );

		self.backtest_result = {	\
			"overall" : overall_return_series,\
			"long" : long_return_series,\
			"short" : short_return_series,\

			"cum_overall" : (1+overall_return_series).cumprod(),\
			"cum_long" : (1+long_return_series).cumprod(),\
			"cum_short" : (1+short_return_series).cumprod()
		};

		self.backtest_finished = True;
		return self.backtest_result;

	def _CalcHistoricalPositions( self ):
		returns = self.returns;
		sample_lookback = self.sample_lookback;

		regression_lags = self.reg_lags_and_weights.keys();
		max_regression_lag = max( regression_lags );
		reg_coeff_names = [ "coeff_0" ] + [ "coeff_%d" % x for x in regression_lags ];	# column names for regression coefficients

		coeff_index = [];
		coeff_data = [];

		long_pos_hist = dict();		# long positions from beginning to end
		short_pos_hist = dict();	# short positions from beginning to end
		predicted_returns_hist = dict();	# predicted return for each stock in the universe

		start = max_regression_lag + sample_lookback;		# starting row for regression
		end = len(returns) ;				# last row for regression

		for i in range( start, end-1 ):		# "end" is the last observed real return, therefore regression ends at one step before it
			period = returns.index[i];		# current month

			positions_i = self._predict( i );

			long_pos_hist[period] = positions_i["long_positions"];
			short_pos_hist[period] = positions_i["short_positions"];
			predicted_returns_hist[period] = positions_i["universe_prediction"];

	
		self.long_pos_hist_df = pd.DataFrame( long_pos_hist ).transpose();
		self.short_pos_hist_df = pd.DataFrame( short_pos_hist ).transpose();
		self.predicted_returns_hist_df = pd.DataFrame( predicted_returns_hist ).transpose();
		self.predicted_returns_hist_df.columns = returns.columns;			# rename columns using stock tickers

		return ( self.long_pos_hist_df, self.short_pos_hist_df, self.predicted_returns_hist_df );

	def _predict( self, current_i ):	# current_i is the numeric index of a certain period in the return Series
		"""	Function that ranks stocks in the universe for a given step "i"
		"""
		# Run linear regression on historical returns
		i_start = current_i - self.sample_lookback;
		i_end = current_i;
		res = self._regression_OLS( i_start, i_end );			# res must have a method called "predict"!!

		# Use the regression model to predict returns for "current_i + 1" and rank the universe
		reg_lags_and_weights = self.reg_lags_and_weights;
		returns = self.returns;
		num_longs = self.num_longs;
		num_shorts = self.num_shorts;

		X_new = list();
		for lag in reg_lags_and_weights:
			x_index = current_i + 1 - lag;
			weight = reg_lags_and_weights[lag];
			X_new.append( weight * np.array( returns.iloc[ x_index, : ] ) );	# multiply lagged returns with corresponding weight
		X_new = np.array( X_new ).T;					# After transpose, each row is the lag-weighted return of one stock
		X_new = sm.add_constant( X_new );						# prepend constant before historical returns
		y_predict = res.predict( X_new );						# predicted return for the stocks in the universe
		y_rank = y_predict.argsort()[::-1];						# rank of y_predict in descending order (i.e from Max to Min)

		if num_longs > 0:
			long_indices = y_rank[:num_longs];			# long top-performing stocks
		else:
			long_indices = list();
		if num_shorts > 0:
			short_indices = y_rank[(-num_shorts):]		# short bottom-performing stocks
		else:
			short_indices = list();

		long_stocks = returns.columns.values[ long_indices ];
		short_stocks = returns.columns.values[ short_indices ];

		periods = returns.index[current_i];

		# assemble returned values
		ret = { "long_positions" : long_stocks, "short_positions" : short_stocks, "universe_prediction" : y_predict };
		return ret;

	def _regression_OLS( self, i_start, i_end ):
		X, y = self._AssembleRegressionData_i( i_start, i_end );
		model = sm.OLS( y, X );
		res = model.fit();
		return res;

	def _AssembleRegressionData_i( self, i_start, i_end ):	# i_start
		"""	OLS regression on stock returns between i_start and i_end
		"""
		returns = self.returns;
		reg_lags_and_weights = self.reg_lags_and_weights;

		y_arr = list();		# collect y
		X_arr = list();		# collect X
		for j in range( i_start, i_end ):			# j-th row is independent variable "y"
			y_j = np.array( returns.iloc[j,:] );
			y_arr.append( y_j );


# to be added for adding weight to X
			X_j = list();
			for lag in reg_lags_and_weights:
				x_index = j - lag;
				weight = reg_lags_and_weights[lag];
				X_j.append( weight * np.array( returns.iloc[ x_index, : ] ) );

			X_j = np.array( X_j ).T;			# After transpose, each row is the return of the same stock but with different lags
			X_arr.append( X_j );
		
		y = np.concatenate( y_arr );
		X = np.concatenate( X_arr );
		X = sm.add_constant( X );				# add intercept

		return (X, y)




	#------------------------------------------------------------
	# Performance Analysis
	#------------------------------------------------------------
	def BackTestAnalysis( self ):
		backtest_res = self.backtest_result;

		strategy_returns = backtest_res[ "overall" ];
		riskfree_rate = self.riskfree_rate;
		benchmark_returns = self.benchmark_returns;

		# Average and standard deviation
		ave_return = strategy_returns.mean();
		volatility = strategy_returns.std();

		# CAMP
		alpha, beta = calcCAMP( strategy_returns, riskfree_rate, benchmark_returns );

		# Sharpe Ratio, Sortino Ratio, and Info Ratio
		sharpe = calcRatioGeneric( strategy_returns, riskfree_rate, annualization_factor = np.sqrt(12) );
		sortino = calcRatioGeneric( strategy_returns, riskfree_rate, use_semi_std = True, annualization_factor = np.sqrt(12) );
		info_ratio = calcRatioGeneric( strategy_returns, benchmark_returns, annualization_factor = np.sqrt(12) );

		self.backtest_analysis = {
			"Average Return" : ave_return,
			"Volatility" : volatility,
			"CAMP" : (alpha, beta),
			"Sharpe" : sharpe,
			"Sortino" : sortino,
			"Info_Ratio" : info_ratio
		};

		return self.backtest_analysis;
