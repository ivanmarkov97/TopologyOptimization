import numpy as np
from sklearn.naive_bayes import GaussianNB

class OptimizationTask:
    objective_func = None
    objective_func_prev = None

    clf = GaussianNB()

    coef_1 = 0
    coef_2 = 0

   # X = n.array['v': 10, 'p': 20]
    #grad = np.array[0, 0]

    def __init__(self, c1, c2):
        self.coef_1 = c1
	self.coef_2 = c2

    def objective (self, x):
        print "objective calculation X"
	self.objective_func = self.coef_1*(1 - x[0])**2 + self.coef_2*(x[1] - x[0]**2)**2 
	print self.objective_func, x[0], x[1]
	if self.objective_func_prev == None:
		self.objective_func_prev = self.objective_func
	if self.objective_func < self.objective_func_prev:
		print "new func better!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
		self.objective_func_prev = self.objective_func
	else:
		print "new func worse!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
	return self.objective_func

    def constrants (self,x):
        print "calculation constrains values"

    def criteria (self, x):
	helper = np.array([0.0, 0.0, 0.0, 0.0])
	cr = np.zeros_like(helper)
	cr[0] = (1 - x[0]**2)
	cr[1] = (x[1] - x[0]**2)
	cr[2] = (x[0])
	cr[3] = (x[1])
	return cr

    def criteria_1(self, x):
	helper = np.array([0.0, 0.0, 0.0])
	cr = np.zeros_like(helper)
	cr[0] = x[0]
	cr[1] = x[1]
	cr[2] = self.coef_1*(1 - x[0])**2 + self.coef_2*(x[1] - x[0]**2)**2
	return cr

    def criteria_2(self, x):
	helper = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
	cr = np.zeros_like(helper)
	cr[0] = (1 - x[0]**2)
        cr[1] = (x[1] - x[0]**2)
        cr[2] = (x[0])
        cr[3] = (x[1])
	cr[4] = self.coef_1*(1 - x[0])**2 + self.coef_2*(x[1] - x[0]**2)**2
	cr[5] = self.coef_1*(1 - x[0])**2 + self.coef_2*(x[1] - x[0]**2)**2
	return cr

    def gradient (self,x, strategy):
        print "calculation OPTask gradient"
        #return strategy.gradient(self.cells, x)
