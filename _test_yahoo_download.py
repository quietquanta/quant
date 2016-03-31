import datetime
from data_getter.yahoo_stock_quote import download_stock_price_hist
from data_getter.stock_symbol_universe import get_sp500_symbols

download_stocks = False;
if download_stocks:
	tickers = get_sp500_symbols();
	print "Selected stocks are:\n\t" + "\n\t".join(tickers);

	download_stock_price_hist( tickers, start = datetime.date( 1999, 12, 31 ), end = datetime.date( 2015, 12, 31), csv_file = "stock_adj_close.csv" );


download_benchmark = True;
if download_benchmark:
	tickers = [ "^GSPC", "SPY" ];		# S&P 500, SPDR S&P 500 ETF
	download_stock_price_hist( tickers, start = datetime.date( 1990, 12, 31 ), end = datetime.date( 2015, 12, 31), csv_file = "benchmark.csv" );


download_riskfree = True;
if download_riskfree:
	rf_tickers = [ "^IRX" ];
	start = datetime.date( 1990, 12, 31 );
	end = datetime.date( 2015, 12, 31 );
	csv_file_name = "riskfree.csv";
	download_stock_price_hist( rf_tickers, start = start, end = end, csv_file = csv_file_name );
