import OptimizationTask
from scipy.optimize import minimize
import numpy as np
from sklearn.naive_bayes import GaussianNB
Nfev = 0

def callBackF(xf):
    global Nfev
    Nfev += 1
    print "callback " + str(Nfev) + " X " + str(xf)

class Optimizator:
    def optimize (self, gradient, objective, x0, strategy):
        print "start optimizator work"
	res = minimize(objective, x0, method = "BFGS", jac = gradient, options = {'maxiter': 30}, callback = callBackF)
	print res
        return res

