import numpy as np;
import scipy as sp;
import pandas as pd;
import statsmodels.api as sm;
import statsmodels;

from performance_analysis import calcCAMP, calcRatioGeneric;
from strategies import RegressionStrategy;

class Regression_OLS( RegressionStrategy ):
	"""	Cross-sectional equity long-short strategy based on autoregression using historical pricing data.
	"""
	def __init__( self, 
			prices, 					# Dataframe of stock price histories for all stocks in the universe
			riskfree_rate,				# DataFrame of Riskfree Rate, e.g. Riskfree Rate
			benchmark_returns,			# DataFrame of Benchmark returns, e.g. S&P 500
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

		# Default values should not be overriden at initialization
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

		riskfree_rate = self.riskfree_rate;
		benchmark_returns = self.benchmark_returns;

		# Calculate historical positions
		self._CalcHistoricalPositions();
		predicted_returns_hist_df = self.predicted_returns_hist_df;
		real_returns_hist_df = self.returns;

		# Simulation
		period_seq = [];
		strategy_long_return_seq = [];			# return sequence of longed positions
		strategy_short_return_seq = [];			# return sequence of shorted positions
		strategy_return_seq = [];				# return sequence of overall portfolio
		port_return_seq = [];

		pred_vs_real_seq = [];					# store measures for the quality of prediction from PREVIOUS period

		asset_weights_seq = [];					# { 'riskfree' : w_1, 'benchmark' : w_2, 'strategy' : w_3 } where w_1 + w_2 + w_3 = 1

		# Given the positions throughout step i, calculate relevant returns
		for i in range( len( self.long_pos_hist_df) ):
			period = self.long_pos_hist_df.index[i];
			period_seq.append( period );					# Add period to sequence

			# Evaluate the prediction quality from previous period's results
			if i > 0:
				prev_period = self.long_pos_hist_df.index[i-1];					# previous period
				prev_reg_result = self.reg_result_hist_df[ prev_period ];		# regression result from previous period

				prev_pred_returns = predicted_returns_hist_df.loc[ prev_period ];	#predicted returns during previous period
				prev_real_returns = real_returns_hist_df.loc[ prev_period ];		#real returns during previous period

				pred_vs_real_diff_std = prev_pred_returns.sub( prev_real_returns ).std();	# standard deviation of "predicted-real"
				pred_vs_real_corrcoefs = np.corrcoef( np.array( prev_pred_returns ), np.array( prev_real_returns ) );	# correlation
				pred_vs_real_spearmanr, pred_vs_real_spearmanr_pv = \
							sp.stats.spearmanr( np.array( prev_pred_returns ),\
												np.array( prev_real_returns ) );	# Spearman corr and p-value

				pred_vs_real_seq.append(\
					{ 	"diff_std" : pred_vs_real_diff_std,\
						"correlation" : pred_vs_real_corrcoefs[0,1],\
						"spearman_ranking_corr" : pred_vs_real_spearmanr,\
						"spearman_ranking_pvalue" : pred_vs_real_spearmanr_pv,\
					}
				);

			# Calculate various returns
			long_stock_list = list( self.long_pos_hist_df.iloc[i,:] );
			short_stock_list = list( self.short_pos_hist_df.iloc[i,:] );

			long_returns = self.returns.loc[ period, long_stock_list ];
			short_returns = self.returns.loc[ period, short_stock_list ];		# series of NaN if short_returns is empty

			long_ave_return = long_returns.mean();
			if len(short_returns) == 0:
				relative_short_weight = 0.			# weights of short positions for the regression strategy
				short_ave_return = 0;
				strategy_return = long_ave_return;
			else:
				relative_short_weight = 0.5;					# weights of short positions for the regression strategy
				short_ave_return = short_returns.mean();			# return NaN if empty series
				strategy_return = (1 - relative_short_weight ) * long_ave_return - relative_short_weight * short_ave_return;

			strategy_long_return_seq.append( long_ave_return );
			strategy_short_return_seq.append( -short_ave_return );	# short position return is the negative of stock returns
			strategy_return_seq.append( strategy_return );

			# Determin if strategy should be "on" based on prediction quality and strategy prediction
			diff_std_compared_to = 0.1;								# Select a proper measure to compare diff_std to
#			diff_std_compared_to = prev_real_returns.std();
#			diff_std_compared_to = prev_reg_result.resid.mean();

			strategy_is_on = ( i == 0 ) or \
								( pred_vs_real_diff_std <= diff_std_compared_to and \
									pred_vs_real_spearmanr_pv < 0.05 and \
									strategy_return > 0 );

			# Asset allocation for current period. Rebalance if necessary
			if strategy_is_on:
				w_riskfree = 0.;
				w_benchmark = 0.0;
				w_strategy = 1.;
			else:
				w_riskfree = 1.;
				w_benchmark = 0.0;
				w_strategy = 0.;
			asset_weights_seq.append( { 'riskfree' : w_riskfree, 'benchmark' : w_benchmark, 'w_strategy' : w_strategy } );

			# Overall
			period_riskfree = riskfree_rate.loc[ period, u'^IRX' ];			# Or use .squeeze() to convert to Series
			period_benchmark = - benchmark_returns.loc[ period, u'^GSPC' ];	# negative for beta hedging
			port_return = w_riskfree * period_riskfree + w_benchmark * period_benchmark + w_strategy * strategy_return;
			port_return_seq.append( port_return );

		# Form Series of various returns
		strategy_return_series = pd.Series( strategy_return_seq, period_seq );
		strategy_long_return_series = pd.Series( strategy_long_return_seq, period_seq );
		strategy_short_return_series = pd.Series( strategy_short_return_seq, period_seq );
		port_return_series = pd.Series( port_return_seq, period_seq );

		asset_weights_df = pd.DataFrame( asset_weights_seq, index=period_seq );

		pred_vs_real_df = pd.DataFrame( pred_vs_real_seq, index = period_seq[:-1] );	# Last period should be excluded as prev_vs_real_df is constructed from "previous period"

		# Add backtest results to object
		self.backtest_result = {	\
			"portfolio" : port_return_series,\
			"strategy" : strategy_return_series,\
			"strategy_long" : strategy_long_return_series,\
			"strategy_short" : strategy_short_return_series,\

			"cum_portfolio" : (1 + port_return_series).cumprod(),\
			"cum_strategy" : (1 + strategy_return_series ).cumprod(),\
			"cum_strategy_long" : (1 + strategy_long_return_series).cumprod(),\
			"cum_strategy_short" : (1 + strategy_short_return_series).cumprod(),\

			"asset_weights" : asset_weights_df,\

			"pred_vs_real_df" : pred_vs_real_df,\
		};

		self.backtest_finished = True;
		return self.backtest_result;

	def _CalcHistoricalPositions( self ):
		returns = self.returns;
		sample_lookback = self.sample_lookback;

		regression_lags = self.reg_lags_and_weights.keys();
		max_regression_lag = max( regression_lags );
		reg_coeff_names = [ "coeff_0" ] + [ "coeff_%d" % x for x in regression_lags ];	# column names for regression coefficients

		# Additional info
		reg_info_index = [];
		reg_result_data = [];
		coeff_data = [];				# regression coefficients
		normality_pv_data = [];			# normality test on regression residual
		hetero_pv_data = [];			# Heteroskedasticity test

		long_pos_hist = dict();		# long positions from beginning to end
		short_pos_hist = dict();	# short positions from beginning to end
		predicted_returns_hist = dict();	# predicted return for each stock in the universe
		universe_ranking_hist = dict();

		start = max_regression_lag + sample_lookback;		# starting row for regression
		end = len(returns) ;				# last row for regression

		# Record long/short positions to be held throughout "target_period"
		for i in range( start, end-1 ):		# "end" is the last observed real return, therefore regression ends at one step before it
			target_period = returns.index[i + 1];	# the next period for which predictions are made, i.e. (i+1)st period

			prediction_i = self._predict( i );		# Prediction is made given info up to current month i

			long_pos_hist[ target_period] = prediction_i["long_positions"];
			short_pos_hist[ target_period] = prediction_i["short_positions"];
			predicted_returns_hist[ target_period ] = prediction_i["universe_prediction"];
			universe_ranking_hist[ target_period ] = prediction_i[ "universe_ranking" ];

			reg_info_index.append( target_period );
			reg_result_data.append( prediction_i[ "reg_result" ] );			# Regression Result for this period

			if prediction_i.has_key( "regression_coefficients" ):
				coeff_data.append( prediction_i[ "regression_coefficients" ] );
			if prediction_i.has_key( "normality_pvalue" ):
				normality_pv_data.append( prediction_i[ "normality_pvalue" ] );
			if prediction_i.has_key( "heteroskedasticity_pvalue" ):
				hetero_pv_data.append( prediction_i[ "heteroskedasticity_pvalue" ] );

		# Append prediction results to the object
		self.long_pos_hist_df = pd.DataFrame( long_pos_hist ).transpose();
		self.short_pos_hist_df = pd.DataFrame( short_pos_hist ).transpose();
		self.predicted_returns_hist_df = pd.DataFrame( predicted_returns_hist ).transpose();
		self.predicted_returns_hist_df.columns = returns.columns;			# rename columns using stock tickers
		self.universe_ranking_hist_df = pd.DataFrame( universe_ranking_hist ).transpose();

		# Append regression info to the object
		self.reg_result_hist_df = pd.Series( reg_result_data, reg_info_index );
		if len( coeff_data ) > 0 and len( reg_info_index ) == len( coeff_data ):
			self.reg_coefficients_df = pd.DataFrame( coeff_data, index = reg_info_index, columns = reg_coeff_names );

		if len( normality_pv_data ) > 0 and len( reg_info_index ) == len( normality_pv_data ):
			self.normality_pvalues_series = pd.Series( normality_pv_data, reg_info_index );

		if len( hetero_pv_data ) > 0 and len( reg_info_index ) == len( hetero_pv_data ):
			self.heteroskedasticity_pvalues_series = pd.Series( hetero_pv_data, reg_info_index );

	def _predict( self, current_i ):	# current_i is the numeric index of a certain period in the return Series
		"""	Function that ranks stocks in the universe for a given step "i"
		"""
		# Run linear regression on historical returns
		i_start = current_i - self.sample_lookback;
		i_end = current_i;
		regression = self._regression( i_start, i_end );			
		reg_result = regression[ "reg_result" ];		# reg_result must have a method called "predict"!!

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
		y_predict = reg_result.predict( X_new );						# predicted return for the stocks in the universe
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
		universe_ranking = returns.columns.values[ y_rank ];			# Stock tickers ordered based on prediction
		
		periods = returns.index[current_i];

		# assemble returned values
		ret = { "long_positions" : long_stocks, \
				"short_positions" : short_stocks, \
				"universe_prediction" : y_predict, \
				"universe_ranking" : universe_ranking, \
				"reg_result" : reg_result,\
		};

		if regression.has_key( "reg_coef" ):
			ret["regression_coefficients"] = regression[ "reg_coef" ];
		if regression.has_key( "normality_pvalue" ):
			ret["normality_pvalue"] = regression[ "normality_pvalue" ];
		if regression.has_key( "heteroskedasticity_pvalue" ):
			ret["heteroskedasticity_pvalue" ] = regression[ "heteroskedasticity_pvalue" ];

		return ret;

	def _regression( self, i_start, i_end ):
		X, y = self._AssembleRegressionData_i( i_start, i_end );
		model = sm.OLS( y, X );
		fitting_result = model.fit();

		# Normality test
		_, normality_pvalue, _, _ = statsmodels.stats.stattools.jarque_bera( fitting_result.resid );

		# Heteroskedasticity test
		_, hetero_pvalue, _, _ = statsmodels.stats.diagnostic.het_breushpagan( fitting_result.resid, model.exog );

		res = { "reg_result" : fitting_result,\
				"reg_coef" : fitting_result.params,\
				"normality_pvalue" : normality_pvalue,\
				"heteroskedasticity_pvalue" : hetero_pvalue,\
		};
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

		port_returns = backtest_res[ "portfolio" ];
		riskfree_rate = self.riskfree_rate;
		benchmark_returns = self.benchmark_returns;

		# Average and standard deviation
		ave_return = port_returns.mean();
		volatility = port_returns.std();

		# CAMP
		alpha, beta = calcCAMP( port_returns, riskfree_rate, benchmark_returns );

		# Sharpe Ratio, Sortino Ratio, and Info Ratio
		sharpe = calcRatioGeneric( port_returns, riskfree_rate, annualization_factor = np.sqrt(12) );
		sortino = calcRatioGeneric( port_returns, riskfree_rate, use_semi_std = True, annualization_factor = np.sqrt(12) );
		info_ratio = calcRatioGeneric( port_returns, benchmark_returns, annualization_factor = np.sqrt(12) );

		self.backtest_analysis = {
			"Average Return" : ave_return,
			"Volatility" : volatility,
			"CAMP" : (alpha, beta),
			"Sharpe" : sharpe,
			"Sortino" : sortino,
			"Info_Ratio" : info_ratio
		};

		return self.backtest_analysis;
