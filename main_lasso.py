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
snp_500 = snp_500.resample( "BM" ).last();
snp_500_returns = snp_500.pct_change().iloc[1:];
benchmark_returns = snp_500_returns;

rf_annualized_rate = read_from_csv( "riskfree.csv", rescale_factor = 0.01 );
rf_annualized_rate = rf_annualized_rate.resample( "BM" ).last();
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

#long_pos, short_pos = strategy._CalcHistoricalPositions();
strategy.BackTest();
print strategy.BackTestAnalysis();



#---------------------------------
# Plot
#---------------------------------
if True:
	plt.figure();
	cum_overall_series = strategy.backtest_result['cum_portfolio'];
	cum_overall_series.plot();
	plt.show();

	plt.figure();
	strategy.backtest_result['cum_strategy'].plot();
	plt.show()
