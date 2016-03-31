import pandas as pd;

def read_from_csv(
	filename,
	rescale_factor = 1
):
	"""
	Helper function to read data from CSV files. By default, set Date as index.
	"""
	data = pd.read_csv( filename );
	data.set_index( "Date", inplace = True );
	data.index = pd.to_datetime( data.index );
	data = data * rescale_factor;

	return data;
