
class RegressionStrategy:
	def __init__( self ):
		pass;

	def init_summary( self ):
		"""	Return a summary of the strategy
		"""
		pass;

	def BackTest( self ):
		"""	Go through history and calculate
		(1) Historical positions (Long/Short) as a result of the strategy
		(2) Predicted return for each stock in the universe
		"""
		pass;

	#------------------------------------------------------------
	# Performance Analysis
	#------------------------------------------------------------
	def BackTestAnalysis( self ):
		pass;
