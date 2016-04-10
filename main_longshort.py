import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt

from my_helper_fn import *

from myStrategies.regression import RegressionLongShort

#------------------------------------------------------
# 1. Read data
#------------------------------------------------------
# (1) Read daily prices and convert to monthly returns
price_csv = 'stock_adj_close_2000_2015.csv';
stock_prices = read_from_csv( price_csv );
# (2) Benchmark and risk-free rate
snp_500 = read_from_csv( "benchmark.csv" );
snp_500 = snp_500.resample( "BM" ).last();
benchmark_returns = snp_500_returns = snp_500.pct_change().iloc[1:];

rf_annualized_rate = read_from_csv( "riskfree.csv", rescale_factor = 0.01 );
def deannualization_func( annual_rate, freq="M" ):
	if freq is "M":
		return (1+annual_rate)**(1/12) - 1

riskfree_rate = rf_monthly_rate = rf_annualized_rate.apply( deannualization_func );

#-------------------------------------------------------------------------
# 2. Backtest strategy
#-------------------------------------------------------------------------
"""
longonly_strat = RegressionLongShort( stock_prices,
			resample_freq = "BM",
			sample_lookback = 60,			# number of periods of looking back for training data
			regression_lags = [1],			# lags for autoregression
			num_longs = 10,					# number of stocks to long for each period
			num_shorts = 0,					# number of stocks to short for each period
);
longonly_res = longonly_strat.BackAnalysis();

"""
reg_lags_and_weights = { 1:1, 2:0.9, 3:0.9**2, 12:0.9**11 };
#reg_lags_and_weights = { 1:1 };
strat = RegressionLongShort( 
			stock_prices,
			riskfree_rate,
			benchmark_returns,
			resample_freq = "BM",
			sample_lookback = 60,			# number of periods of looking back for training data
			reg_lags_and_weights = reg_lags_and_weights,
			num_longs = 10,					# number of stocks to long for each period
			num_shorts = 0,				# number of stocks to short for each period
);
strat_is_long_only = strat.num_shorts <= 0;
backtest_res = strat.BackTest();
perf_analysis = strat.BackTestAnalysis();
print perf_analysis;


#-------------------------------------------------------------------------
# Plot
#-------------------------------------------------------------------------
from matplotlib import pyplot as plt
if True:
	plt.figure();
	backtest_res[ "cum_overall" ].plot();

	plt.figure();
	backtest_res[ "cum_long" ].plot();

	if not strat_is_long_only:
		plt.figure();
		backtest_res[ "cum_short" ].plot()

	plt.show();
