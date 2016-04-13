from sklearn.linear_model import LassoCV			# Alternative is LassoLarsCV (least-angle regression)

from regression import RegressionLongShort

class RegressionLasso( RegressionLongShort ):
	def _regression( self, i_start, i_end ):
		"""
		Model of Lasso
		"""
		X, y = self._AssembleRegressionData_i( i_start, i_end );

		lasso = LassoCV( cv = 10 );
		lasso.fit( X, y );
		return lasso;

