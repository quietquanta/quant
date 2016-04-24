""" Functions for analyzing the performance of a given strategy
"""
import numpy as np;
import statsmodels.api as sm;

def _myLinearRegression(
	y,
	X,
	include_intercept = False,
):
	"""	Generic univariate linear regression given two return series
	"""
	if include_intercept:
		X = sm.add_constant( X );
	model = sm.OLS( y, X );
	res = model.fit();
	return res;


def calcCAMP(
	strategy_returns,
	riskfree_rate,
	benchmark_returns,
	annualization_factor = 1,				# If returns and rates are monthly, annualization_factor should be reset to sqrt(12)
):
	""" CAMP model. Return alpha and beta
	"""
	strategy_excess_returns = strategy_returns.sub( riskfree_rate.squeeze(), axis=0 ).dropna() * annualization_factor;
	benchmark_returns = benchmark_returns.loc[ strategy_excess_returns.index, :];		# align benchmark to strategy
	benchmark_excess_returns = benchmark_returns.sub( riskfree_rate.squeeze(), axis=0 ).dropna() * annualization_factor;

	camp_result = _myLinearRegression( strategy_excess_returns, benchmark_excess_returns, include_intercept = True );
	alpha = camp_result.params.iloc[0];
	beta = camp_result.params.iloc[1];

	return (alpha, beta);

def calcRatioGeneric(
	strategy_returns,
	benchmark_returns,
	use_semi_std = False,
	annualization_factor = 1,		# assume returns are annual. For monthly returns, it should be reset to sqrt(12)
):
	""" Generic function that calculate Sharpe Ratio, Sortino Ratio and Information Ratio
	"""
	strategy_excess_returns = strategy_returns.sub( benchmark_returns.squeeze(), axis=0 ).dropna();
	mean = strategy_excess_returns.mean();

	if use_semi_std:	# Use semi standard deviation if True
		below_mean = ( strategy_excess_returns < mean );
		returns_below_mean = strategy_excess_returns[below_mean];
		std = np.sqrt( (returns_below_mean**2 ).sum() );
	else:
		std = strategy_excess_returns.std();

	ratio = mean/std * annualization_factor;
	return ratio;


