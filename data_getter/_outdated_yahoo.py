"""
(2016-03-01 )
Oudated!!! yahoo_finance seems to be broken
"""

import datetime
import pandas as pd

from yahoo_finance import Share			# Data API for Yahoo Finance

def download_stock_prices(
	tickers = [ 'AAPL' ],
	price_column = 'Adj_Close',								# assume it's the Adjusted Close price that are interested
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

		stock = Share( ticker );
		start_str = start.isoformat();
		end_str = end.isoformat();
		hist = stock.get_historical( start_str, end_str );

		# Get time series of stock prices (Don't sort before forming the Series!!!)
		price_data = [ float(x[price_column]) for x in hist ];
		date_index = [ datetime.datetime.strptime(x['Date'], "%Y-%m-%d" ).date() for x in hist];
		if min( date_index ) > start:								# Pass if the no stock price is available on Start
			continue;
		stock_ts = pd.Series( price_data, date_index );

		# Add current stock TS to the DataFrame
		df[ticker] = stock_ts;
	
	df.to_csv( csv_file );
	return True;

########################################
def download_stock_data(
	tickers = [ 'AAPL' ],
	start = datetime.date( 2009, 12, 31 ),				# assume start is guaranteed to be a weekday
	end = datetime.date( 2015, 12, 31 ),
	csv_file = "stock_price_test.csv",
):
	"""
	The function saves all the columns of the data to csv: Adj_Close, Open, High, Low, Close, Symbol.
	It would be useful for forming a panel of all raw data of stocks.	
	"""
	# Check validity of inputs
	if len( tickers ) <= 0:
		print "Tickers must not be empty";
		return False;
	if start > end:
		print "Start date " + start.isoformat() + " can't be later than End date " + end.isoformat();

	df_arr = [];	# list of DataFrame: each DataFrame is for a stock
	min_date = start;

	for _i in range( len(tickers) ):
		ticker = tickers[_i];
		print "Index" + str(_i) + "\t" + "Ticker: " + ticker;

		stock = Share( ticker );

		start_str = start.isoformat();
		end_str = end.isoformat();
		hist = stock.get_historical( start_str, end_str );

		# Create DataFrame, and set index to Date	
		df = pd.DataFrame( hist );

		# Data Cleaning
		df['Date'] = df['Date'].apply( \
					lambda x: datetime.datetime.strptime(x, "%Y-%m-%d" ).date() );	# Change date in string to datetime object

		# Set Index and then sort by index
		df.set_index( "Date", inplace=True );
		df.sort_index( ascending=True, inplace=True);

		# skip this stock if it's first date is after the given start date
		if min_date < df.index.min():
			continue;
		else:
			df_arr.append( df );

	print "First date for stock price is " + min_date.isoformat();
	for df in df_arr:
		df = df.loc[ df.index >= min_date ];	# select the common time horizon

	data = pd.concat( df_arr );
	data.to_csv( csv_file );

	return True;


if __name__ == "__main__":
	foo = download_stock_prices();
	if foo:
		print "Stock prices are downloaded successfully"
	else:
		print "Failed to download stock prices!"
