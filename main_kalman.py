import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from pykalman import KalmanFilter

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
# 2. Use Kalman Filter to run first-order augoregression
#------------------------------------------------------
def myKalmanFilter(
	returns,
):
	""" Use Kalman Filter to obtain first-order auto-regression parameters
		r_t = beta_0 + beta_1 * r_(t-1)
	"""
	# Transition matrix and covariance
	trans_mat = np.eye(2);								# Assume beta is not to change over time
	_delta = 1e-1;										
	trans_cov = _delta / (1 - _delta) * np.eye(2);		# This _delta and trans_cov seem to have great impact on the result

	# form Observation Matrix
	data = returns.values[:-1,:];
	_, num_stocks = data.shape;
	print "Number of stocks is ", num_stocks;
	data = np.expand_dims( data, axis = 2 );			# T-by-2-by-1 array
	obs_mat = np.insert( data, 1, 1, axis = 2 );		# Insert column of ones T-2-2 array
	obs_cov = np.eye( num_stocks );						# assume zero correlation among noises in observed stock returns

	print "Shape of observation matrix is ", obs_mat.shape;
	print "Example of obs_mat is ", obs_mat[:2,:,:];

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
#	return state_means;

	slope = pd.Series( state_means[:,0], index );
	intercept = pd.Series( state_means[:,1], index );
	return (intercept, slope);

intercept, slope = myKalmanFilter( monthly_stock_returns );

slope.plot();
plt.show();
