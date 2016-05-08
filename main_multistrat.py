import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from pykalman import KalmanFilter

from my_helper_fn import *
from myStrategies.regression import Regression_OLS
from myStrategies.kalmanfilter import RegressionKalmanFilter

#------------------------------------------------------
# 1. Read data
#------------------------------------------------------
# (1) Read daily prices and convert to monthly returns
price_csv = 'stock_adj_close_2000_2015.csv';
stock_prices = read_from_csv( price_csv );
monthly_stock_prices = stock_prices.resample( "BM", how="last" );
monthly_stock_returns = monthly_stock_prices.pct_change().iloc[1:];
# (2) Benchmark and risk-free rate
snp_500 = read_from_csv( "benchmark.csv" );
snp_500 = snp_500.resample( "BM", how="last" );
snp_500_returns = snp_500.pct_change().iloc[1:];
benchmark_returns = snp_500_returns;

rf_annualized_rate = read_from_csv( "riskfree.csv", rescale_factor = 0.01 );
rf_annualized_rate = rf_annualized_rate.resample( "BM", how="last" );
def deannualization_func( annual_rate, freq="M" ):
	if freq is "M":
		return (1+annual_rate)**(1./12) - 1

rf_rate = rf_monthly_rate = rf_annualized_rate.apply( deannualization_func );
riskfree_rate = rf_rate;


#------------------------------------------------------------
# 2. Create strategies
#------------------------------------------------------------

# 2.1 Regression_OLS
reg_lags_and_weights = { 1:1 };
#reg_lags_and_weights = { 1:1, 12:1 };
strategy_ols = Regression_OLS( 
						stock_prices,
						riskfree_rate,
						benchmark_returns,
						resample_freq = "BM",
						sample_lookback = 60,			# number of periods of looking back for training data
						reg_lags_and_weights = reg_lags_and_weights,
						num_longs = 10,					# number of stocks to long for each period
						num_shorts = 0,				# number of stocks to short for each period
);
ols_backtest = strategy_ols.BackTest();
ols_backtest_analysis = strategy_ols.BackTestAnalysis();
print ols_backtest_analysis;

# 2.2 RegressionKalmanFilter
strategy_kalmanfilter = RegressionKalmanFilter(
				stock_prices,
				riskfree_rate,
				benchmark_returns,
				resample_freq = "BM",
				num_longs = 10,					# number of stocks to long for each period
				num_shorts = 0,				# number of stocks to short for each period
);

kf_backtest = strategy_kalmanfilter.BackTest();
kf_backtest_analysis = strategy_kalmanfilter.BackTestAnalysis();
print kf_backtest_analysis

#---------------------------------------------------------------
# 3. Compare backtest results
#---------------------------------------------------------------

common_start = max( ols_backtest["portfolio"].index[0], kf_backtest["portfolio"].index[0] );
ols_port_monthly = ols_backtest["portfolio"].loc[ common_start:];
ols_strategy_monthly = ols_backtest['strategy'].loc[ common_start:];

kf_monthly = kf_backtest[ "portfolio" ].loc[common_start:];
ols_port_cum_returns = (1 + ols_port_monthly ).cumprod();
ols_strategy_cum_returns = (1 + ols_strategy_monthly ).cumprod();
kf_cum_returns = (1 + kf_monthly ).cumprod();

#---------------------------------------------------------------
# 4. Plots
#---------------------------------------------------------------
if True:
	fig1 = plt.figure(1);
	plt.plot( ols_port_monthly.index, ols_port_monthly.values, label = "Autoregression with\nfixed-size moving window" );
	plt.plot( kf_monthly.index, kf_monthly.values, label = "Kalman Filter" );
	plt.xlabel( "Date" );
	plt.ylabel( "Monthly Returns" );
	plt.title( "Monthly Returns of Strategies" );
	plt.legend();

	fig2 = plt.figure(2);
	plt.plot( ols_port_cum_returns.index, ols_port_cum_returns.values, label = "OLS-Portfolio with\nfixed-size moving window" );
	plt.plot( ols_strategy_cum_returns.index, ols_strategy_cum_returns.values, \
				label = "OLS-Strategy with\nfixed-sized moving window" );
	plt.plot( kf_cum_returns.index, kf_cum_returns.values, label = "Kalman Filter" );
	plt.xlabel( "Date" );
	plt.ylabel( "Cumulative Returns" );
	plt.title( "Cumulative Returns of Strategies" );
	plt.legend( 
#		bbox_to_anchor = (0,1),
		loc = 0
	);

	plt.show();
