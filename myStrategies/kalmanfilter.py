import numpy as np
import scipy as sp
import pandas as pd
from pykalman import KalmanFilter

from strategies import RegressionStrategy
from performance_analysis import calcCAMP, calcRatioGeneric;

class RegressionKalmanFilter( RegressionStrategy ):
	"""
	Use Kalman Filter to obtain auto-regressive coefficient
	"""
	def __init__( self, 
			prices, 					# Dataframe of stock price histories for all stocks in the universe
			riskfree_rate,				# Series of Riskfree Rate, e.g. Riskfree Rate
			benchmark_returns,			# Series of Benchmark returns, e.g. S&P 500
			resample_freq = "BM",
			sample_lookback = 60,			# number of periods of looking back for training data
			num_longs = 10,					# number of stocks to long for each period
			num_shorts = 10,				# number of stocks to short for each period
	):

		self.prices = prices.resample( resample_freq ).last();
		self.returns = self.prices.pct_change().iloc[1:];

		self.riskfree_rate = riskfree_rate;
		self.benchmark_returns = benchmark_returns;

		# Regression parameters
		self.num_longs = num_longs;
		self.num_shorts = num_shorts;

		self.backtest_finished = False;

	def init_summary( self ):
		"""	Return a summary of the strategy
		"""
		pass;

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


		for i in range( len( self.long_pos_hist_df) ):		# for each selected position at step "i", get its return at step "i+1"
			period = self.long_pos_hist_df.index[i];
			period_seq.append( period );					# Add period to sequence

			if i > 0:
				prev_period = self.long_pos_hist_df.index[i-1];					# previous period

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

			long_stock_list = list( self.long_pos_hist_df.iloc[i,:] );
			short_stock_list = list( self.short_pos_hist_df.iloc[i,:] );

			long_returns = self.returns.loc[ period, long_stock_list ];
			short_returns = self.returns.loc[ period, short_stock_list ];		# series of NaN if short_returns is empty

			long_ave_return = long_returns.mean();
			if len(short_returns) == 0:
				relative_short_weight = 0.
				short_ave_return = 0;
				strategy_return = long_ave_return;
			else:
				relative_short_weight = 0.5;				# fraction of portfolio that's in short positions
				short_ave_return = short_returns.mean();			# return NaN if empty series
				strategy_return = (1 - relative_short_weight ) * long_ave_return - relative_short_weight * short_ave_return;

			strategy_long_return_seq.append( long_ave_return );
			strategy_short_return_seq.append( -short_ave_return );	# short position return is the negative of stock returns
			strategy_return_seq.append( strategy_return );

			# Determin if strategy should be "on" based on prediction quality and strategy prediction
			diff_std_compared_to = 0.1;								# Select a proper measure to compare diff_std to
#			diff_std_compared_to = prev_real_returns.std();

			strategy_is_on = ( i > 0 ) and \
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


		strategy_return_series = pd.Series( strategy_return_seq, period_seq );
		strategy_long_return_series = pd.Series( strategy_long_return_seq, period_seq );
		strategy_short_return_series = pd.Series( strategy_short_return_seq, period_seq );
		port_return_series = pd.Series( port_return_seq, period_seq );

		asset_weights_df = pd.DataFrame( asset_weights_seq, index=period_seq );

		pred_vs_real_df = pd.DataFrame( pred_vs_real_seq, index = period_seq[:-1] );	# Last period should be excluded as prev_vs_real_df is constructed from "previous period"

		self.backtest_result = {	\
			"portfolio" : port_return_series,\
			"strategy" : strategy_return_series,\
			"strategy_long" : strategy_long_return_series,\
			"strategy_short" : strategy_short_return_series,\

			"cum_portfolio" : (1 + port_return_series).cumprod(),\
			"cum_strategy" : (1 + strategy_return_series).cumprod(),\
			"cum_strategy_long" : (1 + strategy_long_return_series).cumprod(),\
			"cum_strategy_short" : (1 + strategy_short_return_series).cumprod(),\

			"asset_weights" : asset_weights_df,\
			"pred_vs_real_df" : pred_vs_real_df,\
		};

		self.backtest_finished = True;
		return self.backtest_result;


	def _CalcHistoricalPositions( self ):
		"""
		Calculate historical positions based on prediction made with Kalman Filter
		"""
		
		predicted_returns_hist_df = self._PredictionforAllPeriods();
		returns = self.returns;
		num_longs = self.num_longs;
		num_shorts = self.num_shorts;

		start = 10;						# omit the first 10 days when Kalman Filter is still adjusting.
		end = len( returns );

		long_pos_hist = dict();		# long positions from beginning to end
		short_pos_hist = dict();	# short positions from beginning to end

		for i in range( start, end-1 ):
			period = returns.index[i];			# index of date at position i
			predicted_returns_hist_df_i = np.array( predicted_returns_hist_df.loc[ period, : ] );
			rank = predicted_returns_hist_df_i.argsort()[::-1];		# rank of y_predict in descending order (i.e from Max to Min)

			# Long positions and short positions to be held during period "period". Note that the decision has been made by the
			# previous period
			long_pos_hist[period] = returns.columns.values[ rank[:num_longs] ];
			if num_shorts > 0:
				short_pos_hist[period] = returns.columns.values[ rank[(-num_shorts):] ]
			else:
				short_pos_hist[period] = list();

		self.long_pos_hist_df = pd.DataFrame( long_pos_hist ).transpose();
		self.short_pos_hist_df = pd.DataFrame( short_pos_hist ).transpose();
			
		return (self.long_pos_hist_df, self.short_pos_hist_df );


	def _PredictionforAllPeriods( self ):
		"""
		Predict return for all stocks in the universe
		"""
		returns = self.returns;

		intercept, slope = self._KalmanFilterRegression();
		input_returns = returns.shift(1);			# shift t-1 return to t as the inputs: predicted(t) = beta_0 + beta_1 * r(t-1)
		predicted_returns = input_returns.multiply( slope, axis=0 ).add( intercept, axis=0 );

		self.predicted_returns_hist_df = predicted_returns;

		return self.predicted_returns_hist_df;





	def _KalmanFilterRegression( self ):
		""" Use Kalman Filter to obtain first-order auto-regression parameters
			r_t = beta_0 + beta_1 * r_(t-1)
		"""
		returns = self.returns;
		_trans_cov_delta = 1e-3;

		# Transition matrix and covariance
		trans_mat = np.eye(2);								# Assume beta is not to change from t-1 to t
		_delta = _trans_cov_delta;										
		trans_cov = _delta / (1 - _delta) * np.eye(2);		# This _delta and trans_cov seem to have great impact on the result

		# form Observation Matrix
		data = returns.values[:-1,:];
		_, num_stocks = data.shape;

		data = np.expand_dims( data, axis = 2 );			# T-by-2-by-1 array
		obs_mat = np.insert( data, 1, 1, axis = 2 );		# Insert column of ones T-2-2 array
		obs_cov = np.eye( num_stocks );						# assume zero correlation among noises in observed stock returns

		#print "Shape of observation matrix is ", obs_mat.shape;
		#print "Example of obs_mat is ", obs_mat[:2,:,:];

		# Observed stock returns: r_t
		index = returns.index[1:];							# index for beta_1(t)
		observations = returns.values[1:,:]					# matrix of r_t

		# Form Kalman Filter and then filter!
		kf = KalmanFilter( n_dim_obs = num_stocks,
							n_dim_state = 2,				# 2 regression parameters
							initial_state_mean = np.zeros(2),
							initial_state_covariance = np.ones((2,2)),
							transition_matrices = trans_mat,
							transition_covariance = trans_cov,
							observation_matrices = obs_mat,
							observation_covariance = obs_cov,
		);

		state_means, state_covs = kf.filter( observations );

		slope = pd.Series( state_means[:,0], index );
		intercept = pd.Series( state_means[:,1], index );

		kf_coefficients_df = pd.DataFrame( [ intercept, slope ], index = [ "coeff_0", "coeff_1" ] );
		self.kf_coefficients_df = kf_coefficients_df.transpose();

		return (intercept, slope);




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
