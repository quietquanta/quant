import datetime
import pandas as pd

import ystockquote			# Data API for Yahoo Finance

def download_stock_price_hist(
	tickers = [ 'AAPL' ],
	price_column = 'Adj Close',								# assume it's the Adjusted Close price that are interested
	start = datetime.date( 2009, 12, 31 ),				# assume start is guaranteed to be a weekday
	end = datetime.date( 2015, 12, 31 ),
	csv_file = "stock_price_test.csv",
):
	"""
	The function collect the adjusted close prices of all given stocks in "tickers" and save it in a csv.
	It's useful for collecting time-series
	"""
	# Check validity of inputs
	if len( tickers ) <= 0:
		print "Tickers must not be empty";
		return False;
	if start > end:
		print "Start date " + start.isoformat() + " can't be later than End date " + end.isoformat();

	df = pd.DataFrame();			# data frame to return
	for _i in range( len(tickers) ):
		ticker = tickers[_i];
		print "Index" + str(_i) + "\t" + "Ticker: " + ticker;

		start_str = start.isoformat();
		end_str = end.isoformat();
		hist = ystockquote.get_historical_prices( ticker, start_str, end_str );	# dictionary with date string as the key

		# Get time series of stock prices (Don't sort before forming the Series!!!)
		date_index = [];
		price_data = [];
		for key, val in hist.iteritems():
			date_index.append( datetime.datetime.strptime( key, "%Y-%m-%d" ).date() );
			price_data.append( float( val[ price_column ] ) )

		if min( date_index ) > start:								# Pass if the no stock price is available on Start
			continue;
		stock_ts = pd.Series( price_data, date_index );
		stock_ts = stock_ts.sort_index();

		# Add current stock TS to the DataFrame
		df[ticker] = stock_ts;
	
	df.to_csv( csv_file, index_label='Date' );
	return True;

if __name__ == '__main__':
	download_stock_price_hist( ['A', 'AAPL'] );
