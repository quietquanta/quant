import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt

from my_helper_fn import *
from myStrategies.regression_lasso import Regression_Lasso

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


#------------------------------------
# Form class of strategy based on Lasso
#------------------------------------
lags = range(1,2);
reg_lags_and_weights = {};
for lag in lags:
	reg_lags_and_weights[ lag ] = 1;

strategy = Regression_Lasso(
			stock_prices,
			riskfree_rate,
			benchmark_returns,
			resample_freq = "BM",
			reg_lags_and_weights = reg_lags_and_weights,
			num_longs = 10,					# number of stocks to long for each period
			num_shorts = 0,				# number of stocks to short for each period
);

backtest_res = strategy.BackTest();
print strategy.BackTestAnalysis();

#---------------------------------
# Plot
#---------------------------------
if True:
	plt.figure();
	backtest_res['cum_portfolio'].plot();

	plt.figure();
	backtest_res['cum_strategy'].plot();

	plt.figure();
	backtest_res[ 'pred_vs_real_df' ].loc[:,"diff_std"].plot( legend=True );

#	plt.figure();
#	backtest_res[ 'pred_vs_real_df' ].loc[:,"correlation"].plot( legend=True );

#	plt.figure();
#	backtest_res[ 'pred_vs_real_df' ].loc[:,"spearman_ranking_corr"].plot( legend=True );

	plt.figure();
	backtest_res[ 'pred_vs_real_df' ].loc[:,"spearman_ranking_pvalue"].plot( legend=True );

	plt.show();
