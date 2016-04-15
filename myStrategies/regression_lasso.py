import numpy as np
from sklearn.linear_model import LassoCV			# Alternative is LassoLarsCV (least-angle regression)

from regression import RegressionLongShort

class RegressionLasso( RegressionLongShort ):
	def _regression( self, i_start, i_end ):
		"""
		Model of Lasso
		"""
		X, y = self._AssembleRegressionData_i( i_start, i_end );

		lasso = LassoCV( cv = 10 );
		lasso.fit_intercept = True;
		lasso.fit( X, y );

# Extract Coefficients from LassoCV doesn't quite work. Need to continue
#		reg_coefficients = [ lasso.intercept_ ] + list( lasso.coef_ );
#		print reg_coefficients

		reg_coefficients = None;			# Needs to be updated to show coefficients for predict!!

		return lasso, reg_coefficients;

