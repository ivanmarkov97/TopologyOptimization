import OptimizationTask
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import matplotlib.pyplot as plt
import cPickle as pickle
#np.seterr(divide='ignore', invalid='ignore')

class BaseStrategy:
    num_call = 0
    grad = np.array([[None, None]])
    coef_1 = None
    coef_2 = None
    _X1 = []
    _Y1 = []
    _X2 = []
    _Y2 = []
    _X3 = []
    _Y3 = []
    _G1 = []
    _G2 = []
    _G3 = []
    n_it_1 = 0
    n_it_2 = 0
    n_it_3 = 0
    #model_X = GaussianNB()
    #model_Y = GaussianNB()

    ot = OptimizationTask.OptimizationTask(1, 100)

    def method(self):
	self.num_call = self.num_call + 1
	print "num call = " + str(self.num_call)

    def initCoef(self, c1, c2):
        self.ot = OptimizationTask.OptimizationTask(c1, c2)
        self.coef_1 = c1
        self.coef_2 = c2
   
#    def learnStrat(self, cr, m_X, m_Y):
#        self.model_X.fit(cr, m_X)
#        self.model_Y.fit(cr, m_Y)
#        self.fit_count = self.fit_count + 1
#        print "<<<<<<<<<FIT_COUNT  = " + str(self.fit_count) + ">>>>>>>>>>>>>>>>>>"
#        return self.model_X, self.model_Y

    def showFunctionMinimize(self):
        fig = plt.figure()
        plt.plot(self._X1, self._Y1, self._X2, self._Y2, self._X3, self._Y3)
        plt.grid(True)
        print fig.axes
        plt.xlabel("iterations")
        plt.ylabel("Function Value")
        plt.title("Function(iteration)")
        plt.show()

    def showFunctionMinimizeGradient(self):
        fig = plt.figure()
        plt.plot(self._X1, self._G1, self._X2, self._G2, self._X3, self._G3)
        plt.grid(True)
        plt.xlabel("iterations")
        plt.ylabel("Gradient Value")
        plt.title("Gradient(iteration)")
        print fig.axes
        plt.show()

    def clearGraphics(self):
        self._X = []
        self._grad = []
        self._Y = []
        self.n_it = 0
	self.grad = np.array([0.0, 0.0])

class Strategy1(BaseStrategy):
    _X = []
    _Y = []
    _grad = []   
    n_it = 0

    fit_count = 0

    crit = np.array([[None, None, None, None]])
    model_X = GaussianNB()
    model_Y = GaussianNB()

    mas_X = [1]
    mas_Y = [1]

    flag_crit = 0
    flag_grad = 0
   

    def gradientByCrit(self, x):
	print "calculation gradientByCrit"

	mX = GaussianNB()
	mY = GaussianNB()

	cr = self.ot.criteria(x)

	print "cr " + str(cr)
	ret = np.array([0.0, 0.0])
	ret = np.zeros_like(x)

	if len(self.crit) == 1 and self.flag_crit == 0:
		self.flag_crit = 1
		self.crit[0] = cr
	else:
		add_crit = np.zeros_like(cr)
        	self.crit = np.vstack((self.crit, add_crit))
		self.crit[len(self.crit) - 1] = cr
	
	der = np.array([0.0, 0.0])
	der = np.zeros_like(x)
	der[0] = -4*x[0]*cr[1] - 2*cr[0]
	der[1] = 2*cr[1]
	
	if len(self.grad) == 1 and self.flag_grad == 0:
		self.flag_grad = 1
		self.grad[0] = der
		self.mas_X[0] = der[0]
		self.mas_Y[0] = der[1]
	else:
		add_grad = np.zeros_like(x)
       		self.grad = np.vstack((self.grad, add_grad))
		self.mas_X.append(0)
		self.mas_Y.append(0)
		self.grad[len(self.grad) - 1] = der

	if self.grad[len(self.grad) - 1][0] > 1e-3:
		ret[0] = 1.0
	elif self.grad[len(self.grad) - 1][0] < -1e-3:
		ret[0] = -1.0
	else:
		ret[0] = 0.0
	
	self.mas_X[len(self.mas_X) - 1] = ret[0]

	if self.grad[len(self.grad) - 1][1] > 1e-3:
		ret[1] = 1.0
	elif self.grad[len(self.grad) - 1][1] < -1e-3:
		ret[1] = -1.0
	else:
		ret[1] = 0.0	
	
	self.mas_Y[len(self.mas_Y) - 1] = ret[1]
	
	print "###########################################################"
	self.grad[len(self.grad) - 1][0] = ret[0]
        self.grad[len(self.grad) - 1][1] = ret[1]

	mX, mY = self.learnStrat(self.crit, self.mas_X, self.mas_Y)
        gr_x = mX.predict(cr)
        gr_y = mX.predict(cr)
	print "///////////////////////////////////////////////////////////////////////////////////////////////"
	print "|"
	print "|"
	print ret, gr_x, gr_y
	print "|"
	print "|"
	print "///////////////////////////////////////////////////////////////////////////////////////////////"
	#ret[0] = gr_x
	#ret[1] = gr_y
	return ret

    def gradient (self, x):
        print "calculation strategy gradient"
        cr = self.ot.criteria(x)
        xm = x[1:-1]
        ret = np.zeros_like(x)
        xm_m1 = x[:-2]
        xm_p1 = x[2:]
        der = np.zeros_like(x)
        der[1:-1] = 2*(xm-xm_m1**2) - 4*(xm_p1 - xm**2)*xm - 2*(1-xm)
        der[0] = -4*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
        der[1] = 2*(x[-1]-x[-2]**2)
        der[1] = 2*(x[1]-x[0]**2)

	if self.n_it_1 < 25:
        	self.n_it_1 = self.n_it_1 + 1
        	self._X.append(self.n_it)
		self._X1.append(self.n_it_1)
        	self._Y.append(self.ot.objective(x))
		self._Y1.append(self.ot.objective(x))
        	self._grad.append(der[0]*der[0] + der[1]*der[1])
		self._G1.append(der[0]*der[0] + der[1]*der[1])

        if der[0] > 1e-3:
                ret[0] = 1.0
        elif der[0] < -1e-3:
                ret[0] = -1.0
        else:
                ret[0] = 0.0

        if der[1] > 1e-3:
                ret[1] = 1.0
        elif der[1] < -1e-3:
                ret[1] = -1.0
        else:
                ret[1] = 0.0

	with open("Str1_model_X.pickle", "rb") as f:
		new_model_X = pickle.load(f)
	with open("Str1_model_Y.pickle", "rb") as f1:
		new_model_Y = pickle.load(f1)
	
	gr_x = new_model_X.predict(cr)
	gr_y = new_model_Y.predict(cr)
        #gr_x = self.model_X.predict(cr)
        #gr_y = self.model_Y.predict(cr)
        print cr
        if gr_x == 0:
                ret[0] = gr_x
        if gr_y == 0:
                ret[1] = gr_y
	#ret[0] = gr_x
	#ret[1] = gr_y
        return ret

    def learnStrat(self, cr, m_X, m_Y):
        self.model_X.fit(cr, m_X)
        self.model_Y.fit(cr, m_Y)
        self.fit_count = self.fit_count + 1
        print "<<<<<<<<<FIT_COUNT  = " + str(self.fit_count) + ">>>>>>>>>>>>>>>>>>"
	with open("Str1_model_X.pickle", "wb") as f:
		pickle.dump(self.model_X, f)
	with open("Str1_model_Y.pickle", "wb") as f1:
		pickle.dump(self.model_Y, f1)
        return self.model_X, self.model_Y



class Strategy2(BaseStrategy):
    _X = []
    _Y = []
    _grad = []   
    n_it = 0

    fit_count = 0

    crit = np.array([[None, None, None, None, None, None]])
    model_X = GaussianNB()
    model_Y = GaussianNB()

    mas_X = [1]
    mas_Y = [1]

    flag_crit = 0
    flag_grad = 0
   

    def gradientByCrit(self, x):
	print "calculation gradientByCrit"

	mX = GaussianNB()
	mY = GaussianNB()

	cr = self.ot.criteria_2(x)

	print "cr " + str(cr)
	ret = np.array([0.0, 0.0])
	ret = np.zeros_like(x)

	if len(self.crit) == 1 and self.flag_crit == 0:
		self.flag_crit = 1
		self.crit[0] = cr
	else:
		add_crit = np.zeros_like(cr)
        	self.crit = np.vstack((self.crit, add_crit))
		self.crit[len(self.crit) - 1] = cr
	
	der = np.array([0.0, 0.0])
	der = np.zeros_like(x)
	der[0] = -4*x[0]*cr[1] - 2*cr[0]
	der[1] = 2*cr[1]
	
	if len(self.grad) == 1 and self.flag_grad == 0:
		self.flag_grad = 1
		self.grad[0] = der
		self.mas_X[0] = der[0]
		self.mas_Y[0] = der[1]
	else:
		add_grad = np.zeros_like(x)
       		self.grad = np.vstack((self.grad, add_grad))
		self.mas_X.append(0)
		self.mas_Y.append(0)
		self.grad[len(self.grad) - 1] = der

	if self.grad[len(self.grad) - 1][0] > 5e-3:
		ret[0] = 1.0
	elif self.grad[len(self.grad) - 1][0] < -5e-3:
		ret[0] = -1.0
	else:
		ret[0] = 0.0
	
	self.mas_X[len(self.mas_X) - 1] = ret[0]

	if self.grad[len(self.grad) - 1][1] > 5e-3:
		ret[1] = 1.0
	elif self.grad[len(self.grad) - 1][1] < -5e-3:
		ret[1] = -1.0
	else:
		ret[1] = 0.0	
	
	self.mas_Y[len(self.mas_Y) - 1] = ret[1]
	
	print "###########################################################"
	self.grad[len(self.grad) - 1][0] = ret[0]
        self.grad[len(self.grad) - 1][1] = ret[1]

	mX, mY = self.learnStrat(self.crit, self.mas_X, self.mas_Y)
        gr_x = mX.predict(cr)
        gr_y = mX.predict(cr)
	print "///////////////////////////////////////////////////////////////////////////////////////////////"
	print "|"
	print "|"
	print ret, gr_x, gr_y
	print "|"
	print "|"
	print "///////////////////////////////////////////////////////////////////////////////////////////////"
	return ret

    def gradient (self, x):
        print "calculation strategy gradient"
        cr = self.ot.criteria_2(x)
        xm = x[1:-1]
        ret = np.zeros_like(x)
        xm_m1 = x[:-2]
        xm_p1 = x[2:]
        der = np.zeros_like(x)
        der[1:-1] = 2*(xm-xm_m1**2) - 4*(xm_p1 - xm**2)*xm - 2*(1-xm)
        der[0] = -4*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
        der[1] = 2*(x[-1]-x[-2]**2)
        der[1] = 2*(x[1]-x[0]**2)

	if self.n_it_2 < 25:
        	self.n_it_2 = self.n_it_2 + 1
        	self._X.append(self.n_it)
		self._X2.append(self.n_it_2)
        	self._Y.append(self.ot.objective(x))
		self._Y2.append(self.ot.objective(x))
        	self._grad.append(der[0]*der[0] + der[1]*der[1])
		self._G2.append(der[0]*der[0] + der[1]*der[1])

        if der[0] > 1e-3:
                ret[0] = 1.0
        elif der[0] < -1e-3:
                ret[0] = -1.0
        else:
                ret[0] = 0.0

        if der[1] > 1e-3:
                ret[1] = 1.0
        elif der[1] < -1e-3:
                ret[1] = -1.0
        else:
                ret[1] = 0.0

	with open("Str2_model_X.pickle", "rb") as f:
                new_model_X = pickle.load(f)
        with open("Str2_model_Y.pickle", "rb") as f1:
                new_model_Y = pickle.load(f1)

        gr_x = new_model_X.predict(cr)
        gr_y = new_model_Y.predict(cr)
	
        #gr_x = self.model_X.predict(cr)
        #gr_y = self.model_Y.predict(cr)
        print cr
        if gr_x == 0:
                ret[0] = gr_x
        if gr_y == 0:
                ret[1] = gr_y
	#ret[0] = gr_x
	#ret[1] = gr_y	
        return ret

    def learnStrat(self, cr, m_X, m_Y):
        self.model_X.fit(cr, m_X)
        self.model_Y.fit(cr, m_Y)
        self.fit_count = self.fit_count + 1
        print "<<<<<<<<<FIT_COUNT  = " + str(self.fit_count) + ">>>>>>>>>>>>>>>>>>"
	with open("Str2_model_X.pickle", "wb") as f:
                pickle.dump(self.model_X, f)
        with open("Str2_model_Y.pickle", "wb") as f1:
                pickle.dump(self.model_Y, f1)
        return self.model_X, self.model_Y


class Strategy3(BaseStrategy):
    _X = []
    _Y = []
    _grad = []   
    n_it = 0

    fit_count = 0

    crit = np.array([[None, None, None]])
    model_X = GaussianNB()
    model_Y = GaussianNB()

    mas_X = [1]
    mas_Y = [1]

    flag_crit = 0
    flag_grad = 0
   

    def gradientByCrit(self, x):
	print "calculation gradientByCrit"

	mX = GaussianNB()
	mY = GaussianNB()

	cr = self.ot.criteria_1(x)

	print "cr " + str(cr)
	ret = np.array([0.0, 0.0])
	ret = np.zeros_like(x)

	if len(self.crit) == 1 and self.flag_crit == 0:
		self.flag_crit = 1
		self.crit[0] = cr
	else:
		add_crit = np.zeros_like(cr)
        	self.crit = np.vstack((self.crit, add_crit))
		self.crit[len(self.crit) - 1] = cr
	
	der = np.array([0.0, 0.0])
	der = np.zeros_like(x)
	der[0] = -4*x[0]*cr[1] - 2*cr[0]
	der[1] = 2*cr[1]
	
	if len(self.grad) == 1 and self.flag_grad == 0:
		self.flag_grad = 1
		self.grad[0] = der
		self.mas_X[0] = der[0]
		self.mas_Y[0] = der[1]
	else:
		add_grad = np.zeros_like(x)
       		self.grad = np.vstack((self.grad, add_grad))
		self.mas_X.append(0)
		self.mas_Y.append(0)
		self.grad[len(self.grad) - 1] = der

	if self.grad[len(self.grad) - 1][0] > 1e-2:
		ret[0] = 1.0
	elif self.grad[len(self.grad) - 1][0] < -1e-2:
		ret[0] = -1.0
	else:
		ret[0] = 0.0
	
	self.mas_X[len(self.mas_X) - 1] = ret[0]

	if self.grad[len(self.grad) - 1][1] > 1e-2:
		ret[1] = 1.0
	elif self.grad[len(self.grad) - 1][1] < -1e-2:
		ret[1] = -1.0
	else:
		ret[1] = 0.0	
	
	self.mas_Y[len(self.mas_Y) - 1] = ret[1]
	
	print "###########################################################"
	self.grad[len(self.grad) - 1][0] = ret[0]
        self.grad[len(self.grad) - 1][1] = ret[1]

	mX, mY = self.learnStrat(self.crit, self.mas_X, self.mas_Y)
        gr_x = mX.predict(cr)
        gr_y = mX.predict(cr)
	print "///////////////////////////////////////////////////////////////////////////////////////////////"
	print "|"
	print "|"
	print ret, gr_x, gr_y
	print "|"
	print "|"
	print "///////////////////////////////////////////////////////////////////////////////////////////////"
	#ret[0] = gr_x
	#ret[1] = gr_y
	return ret

    def gradient (self, x):
        print "calculation strategy gradient"
        cr = self.ot.criteria_1(x)
        xm = x[1:-1]
        ret = np.zeros_like(x)
        xm_m1 = x[:-2]
        xm_p1 = x[2:]
        der = np.zeros_like(x)
        der[1:-1] = 2*(xm-xm_m1**2) - 4*(xm_p1 - xm**2)*xm - 2*(1-xm)
        der[0] = -4*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
        der[1] = 2*(x[-1]-x[-2]**2)
        der[1] = 2*(x[1]-x[0]**2)

	if self.n_it_3 < 25:
        	self.n_it_3 = self.n_it_3 + 1
        	self._X.append(self.n_it)
		self._X3.append(self.n_it_3)
        	self._Y.append(self.ot.objective(x))
		self._Y3.append(self.ot.objective(x))
        	self._grad.append(der[0]*der[0] + der[1]*der[1])
		self._G3.append(der[0]*der[0] + der[1]*der[1])

        if der[0] > 1e-3:
                ret[0] = 1.0
        elif der[0] < -1e-3:
                ret[0] = -1.0
        else:
                ret[0] = 0.0

        if der[1] > 1e-3:
                ret[1] = 1.0
        elif der[1] < -1e-3:
                ret[1] = -1.0
        else:
                ret[1] = 0.0

	with open("Str3_model_X.pickle", "rb") as f:
                new_model_X = pickle.load(f)
        with open("Str3_model_Y.pickle", "rb") as f1:
                new_model_Y = pickle.load(f1)

        gr_x = new_model_X.predict(cr)
        gr_y = new_model_Y.predict(cr)

        #gr_x = self.model_X.predict(cr)
        #gr_y = self.model_Y.predict(cr)
        print cr
        if gr_x == 0:
                ret[0] = gr_x
        if gr_y == 0:
                ret[1] = gr_y
	#ret[0] = gr_x
	#ret[1] = gr_y
        return ret

    def learnStrat(self, cr, m_X, m_Y):
        self.model_X.fit(cr, m_X)
        self.model_Y.fit(cr, m_Y)
        self.fit_count = self.fit_count + 1
        print "<<<<<<<<<FIT_COUNT  = " + str(self.fit_count) + ">>>>>>>>>>>>>>>>>>"
	with open("Str3_model_X.pickle", "wb") as f:
                pickle.dump(self.model_X, f)
        with open("Str3_model_Y.pickle", "wb") as f1:
                pickle.dump(self.model_Y, f1)
        return self.model_X, self.model_Y


def run():
	c = [[1,1], [2,2], [-3,-3], [4,4]]
	i = 0
	s = Strategy()
	while i < len(c):
		print c[i]
		s.gradientByCrit([2,3], c[i])
		i = i + 1
#run()





