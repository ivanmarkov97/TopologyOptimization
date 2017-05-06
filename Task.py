import OptimizationTask
import numpy as np

class Task:

    #concrete task

    input_file = ""
    task_type = "" # task type
    solver= ""
    result_file = ""
    cells = []
    optTask = None
    size = 20*20
    x0 = np.array([1.2, 0.8])
  # def solve (x):
   #    solver.setDensity(x)#pyfoam
    #   code = solver.solve()
     #  cells = solver.read_results()
       #return (code, cells)

    def getOptTask(self): 
	ot = OptimizationTask.OptimizationTask(1, 100)      
	self.optTask = ot
	return self.optTask

    def getGenOptTask(self, a, b):
	ot = OptimizationTask.OptimizationTask(a, b)
	self.optTask = ot
	return self.optTask

   
    def solve(self):
	print "solving"
    #def learnOnSet(set_name):
    #   sets = db.readSet(set_name)
     #  strategy = Strategy.Strategy()
      # for i in sets:
       #    var = i
        #   self.learnVariant(strategy)
       #db.save(strategy, "result.txt")

  # def learnVariant(self, strategy):
        #db = DB.DB()
        #task = db.readTask("hello.txt")
	#for i in range(self.size):
        #	cells.append(Cell.Cell())
        #self.optTask = task.optTask() #OptimizationTask.OptimizationTask()
	#self.optTask = task.getOptTask()

#==================================================

        #optTask.init()
        #solver = task.solver() # Solver.Solver()
        #solver.run(optTask.task)
        #solver.check()
        #Vanya_krasavchik

#================================================

        #strategy = Strategy.Strategy()
#==============================================

        #optimizator = Optimizator.Optimizator()
        #strategySearch = StrategySearch.StrategySearch()
        #strategy = strategySearch.search(strategy, optimizator, task)

#==============================================

        #optTask.gradient(0, strategy)

        #self.donext()

        #db.save(strategy, "result.txt");

        #return strategy

