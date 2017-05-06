import Cell
import OptimizationTask
import Task
import numpy as np
import scipy
import random

class StrategySearch:

    cells = []
    strategy = None
    x0 = np.array([1.3, 0.9])
    x1 = np.array([1., 0.7])
    x2 = np.array([1.1, 1.1])

    def learn (self, strategy, task, optimizator):
        print "learning from tasks"
        task.solve()
	ot = OptimizationTask.OptimizationTask(1, 100)
	ot = task.getOptTask()
        cells = task.cells
	res = optimizator.optimize(strategy.gradientByCrit, ot.objective, self.x0, strategy)
        return strategy

    def search (self, strategy, task, optimizator, n):
        print "begin search"
	print type(strategy)
        cells = task.cells
	self.initCells(task.size)
	ot = OptimizationTask.OptimizationTask(1, 100)
	if n == 0:
		res = optimizator.optimize(strategy.gradient, ot.objective, self.x0, strategy)
	if n == 1:
		res = optimizator.optimize(strategy.gradient, ot.objective, self.x1, strategy)
	if n == 2:
		res = optimizator.optimize(strategy.gradient, ot.objective, self.x2, strategy)

	print res
	#strategy.showFunctionMinimize()
        #strategy.showFunctionMinimizeGradient()
        #strategy.clearGraphics()
        return strategy

    def initCells(self, size):
	i = 0
	cells = []
	while i < size:
		c = Cell.Cell()
		random.seed()
		c.var['xyz']['x'] = random.uniform(0, 20)
		c.var['xyz']['y'] = random.uniform(0, 20)
		c.var['density'] = 1
		cells.append(c)
		i = i + 1
	print len(cells)
	i = 0
	while i < size:
		#print cells[i].getCell()
		i = i + 1


