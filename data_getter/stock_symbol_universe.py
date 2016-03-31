import os;

def get_sp500_symbols():
	""" Return a list of stock symbols of S&P 500
	"""
	file_name = '/home/xiaoxian/Quant_Projects/sp_500.txt';
	f = open( file_name, 'r' );
	if not f:
		raise file_name + ' does not exist!!\nCheck if the file name is correctly set up.';
	ret = f.read().splitlines();
	f.close();

	ret.sort();			# Alphabetical order

	return ret;


if __name__ == "__main__":
	print "First 10 stock symbols in S&P 500:\n";
	ret = get_sp500_symbols();
	for x in ret[:10]:
		print x

