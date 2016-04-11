import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from pykalman import KalmanFilter

from my_helper_fn import *
from myStrategies.regression import RegressionLongShort
from myStrategies.kalmanfilter import RegressionKalmanFilter

#------------------------------------------------------
# 1. Read data
#------------------------------------------------------
# (1) Read daily prices and convert to monthly returns
price_csv = 'stock_adj_close_2000_2015.csv';
stock_prices = read_from_csv( price_csv );
monthly_stock_prices = stock_prices.resample( "BM" ).last();
monthly_stock_returns = monthly_stock_prices.pct_change().iloc[1:];
# (2) Benchmark and risk-free rate
snp_500 = read_from_csv( "benchmark.csv" );
snp_500 = snp_500.resample( "BM" ).last();
snp_500_returns = snp_500.pct_change().iloc[1:];
benchmark_returns = snp_500_returns;

rf_annualized_rate = read_from_csv( "riskfree.csv", rescale_factor = 0.01 );
def deannualization_func( annual_rate, freq="M" ):
	if freq is "M":
		return (1+annual_rate)**(1/12) - 1

rf_rate = rf_monthly_rate = rf_annualized_rate.apply( deannualization_func );
riskfree_rate = rf_rate;


#------------------------------------------------------------
# 2. Create strategies
#------------------------------------------------------------

# 2.1 RegressionLongShort
reg_lags_and_weights = { 1:1 };
reg_lags_and_weights = { 1:1, 12:1 };
strategy_longshort = RegressionLongShort( 
						stock_prices,
						riskfree_rate,
						benchmark_returns,
						resample_freq = "BM",
						sample_lookback = 60,			# number of periods of looking back for training data
						reg_lags_and_weights = reg_lags_and_weights,
						num_longs = 10,					# number of stocks to long for each period
						num_shorts = 0,				# number of stocks to short for each period
);
longshort_backtest = strategy_longshort.BackTest();
longshort_backtest_analysis = strategy_longshort.BackTestAnalysis();
print longshort_backtest_analysis;

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

common_start = max( longshort_backtest["overall"].index[0], kf_backtest["overall"].index[0] );
ls_monthly = longshort_backtest["overall"].loc[ common_start:];
kf_monthly = kf_backtest[ "overall" ].loc[common_start:];
ls_cum_returns = (1 + ls_monthly ).cumprod();
kf_cum_returns = (1 + kf_monthly ).cumprod();

#---------------------------------------------------------------
# 4. Plots
#---------------------------------------------------------------
is_to_plot = True;
if is_to_plot:
	fig1 = plt.figure(1);
	plt.plot( ls_monthly.index, ls_monthly.values, label = "Autoregression with\nfixed-size moving window" );
	plt.plot( kf_monthly.index, kf_monthly.values, label = "Kalman Filter" );
	plt.xlabel( "Date" );
	plt.ylabel( "Monthly Returns" );
	plt.title( "Monthly Returns of Strategies" );
	plt.legend();

	fig2 = plt.figure(2);
	plt.plot( ls_cum_returns.index, ls_cum_returns.values, label = "Autoregression with\nfixed-size moving window" );
	plt.plot( kf_cum_returns.index, kf_cum_returns.values, label = "Kalman Filter" );
	plt.xlabel( "Date" );
	plt.ylabel( "Cumulative Returns" );
	plt.title( "Cumulative Returns of Strategies" );
	plt.legend();

	plt.show();
