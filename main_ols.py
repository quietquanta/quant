import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt

from my_helper_fn import *

from myStrategies.regression import Regression_OLS

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
rf_annualized_rate = rf_annualized_rate.resample( "BM" ).last();
def deannualization_func( annual_rate, freq="M" ):
	if freq is "M":
		return (1+annual_rate)**(1./12) - 1

riskfree_rate = rf_monthly_rate = rf_annualized_rate.apply( deannualization_func );

#-------------------------------------------------------------------------
# 2. Backtest strategy
#-------------------------------------------------------------------------
"""
longonly_strat = Regression_OLS( stock_prices,
			resample_freq = "BM",
			sample_lookback = 60,			# number of periods of looking back for training data
			regression_lags = [1],			# lags for autoregression
			num_longs = 10,					# number of stocks to long for each period
			num_shorts = 0,					# number of stocks to short for each period
);
longonly_res = longonly_strat.BackAnalysis();

"""
lags = range(1,13);
reg_lags_and_weights = {};
for lag in lags:
	reg_lags_and_weights[ lag ] = 1;

strat = Regression_OLS( 
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
portfolio_performance, strategy_performance = strat.BackTestAnalysis();
print portfolio_performance;
print strategy_performance;

#-------------------------------------------------------------------------
# Plot
#-------------------------------------------------------------------------
from matplotlib import pyplot as plt
if True:
	plt.figure();
	backtest_res[ "cum_portfolio" ].plot( legend = True );

	plt.figure();
	backtest_res[ "cum_strategy" ].plot( legend = True);

#	plt.figure();
#	backtest_res[ 'pred_vs_real_df' ].loc[:,"diff_std"].plot( legend=True );

#	plt.figure();
#	backtest_res[ 'pred_vs_real_df' ].loc[:,"correlation"].plot( legend=True );

#	plt.figure();
#	backtest_res[ 'pred_vs_real_df' ].loc[:,"spearman_ranking_corr"].plot( legend=True );

#	plt.figure();
#	backtest_res[ 'pred_vs_real_df' ].loc[:,"spearman_ranking_pvalue"].plot( legend=True );

#	plt.figure();
#	backtest_res[ "cum_long" ].plot();

#	if not strat_is_long_only:
#		plt.figure();
#		backtest_res[ "cum_short" ].plot()

	plt.show();
