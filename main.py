import DB
import Cell
import Task
import Optimizator
import StrategySearch
import Strategy
import random
import cPickle as pickle
from Tkinter import *

db = DB.DB()

def learningMode():
	learnOnSet("")

def learnVariant(strategy, n):
	db.connect()
        task = db.readTask("Task")
        for i in range(task.size):
                task.cells.append(Cell.Cell())

        optimizator = Optimizator.Optimizator()
        strategySearch = StrategySearch.StrategySearch()
	#strategy.initCoef(1, 100)
	strategy = strategySearch.learn(strategy, task, optimizator)
	return strategy


def learnOnSet(set_name=""):
	set_num = 1
	i = 0
	db.connect()
        strategy = Strategy.Strategy1()
	while i < set_num:
		strategy = learnVariant(strategy, 1)
		i = i + 1
	with open("Strategy1.pickle", "wb") as f:
                        pickle.dump(strategy, f)
	i = 0
	strategy = Strategy.Strategy2()
	while i < set_num:
		strategy = learnVariant(strategy, 2)
		i = i + 1
	with open("Strategy2.pickle", "wb") as f:
                        pickle.dump(strategy, f)
	i = 0
        strategy = Strategy.Strategy3()
        while i < set_num:
                strategy = learnVariant(strategy, 3)
                i = i + 1
	with open("Strategy3.pickle", "wb") as f:
                        pickle.dump(strategy, f)
        db.save("read.txt")


def testingMode():
	print "testing mode"
	db.connect()
	random.seed()
	#strategy = Strategy.Strategy1()
	for i in range(1):
		test_task = db.readTask("Task")
		for j in range(test_task.size):
	                test_task.cells.append(Cell.Cell())

		c1 = random.randint(0, 200)
		c2 = random.randint(0, 200)

		optimizator = Optimizator.Optimizator()
		strategySearch = StrategySearch.StrategySearch()

		#strategy.initCoef(c1, c2)

		with open("Strategy1.pickle", "rb") as f:
			strategy = pickle.load(f)

		strategy = strategySearch.search(strategy, test_task, optimizator, 0)
	#strategy = Strategy.Strategy2()
        for i in range(1):
                test_task = db.readTask("Task")
                for j in range(test_task.size):
                        test_task.cells.append(Cell.Cell())

                c1 = random.randint(0, 200)
                c2 = random.randint(0, 200)

                optimizator = Optimizator.Optimizator()
                strategySearch = StrategySearch.StrategySearch()

                #strategy.initCoef(c1, c2)

		with open("Strategy2.pickle", "rb") as f:
                        strategy = pickle.load(f)

                strategy = strategySearch.search(strategy, test_task, optimizator, 1)
	#strategy = Strategy.Strategy3()
        for i in range(1):
                test_task = db.readTask("Task")
                for j in range(test_task.size):
                        test_task.cells.append(Cell.Cell())

                c1 = random.randint(0, 200)
                c2 = random.randint(0, 200)

                optimizator = Optimizator.Optimizator()
                strategySearch = StrategySearch.StrategySearch()

                #strategy.initCoef(c1, c2)

		with open("Strategy3.pickle", "rb") as f:
                        strategy = pickle.load(f)

                strategy = strategySearch.search(strategy, test_task, optimizator, 2)
	strategy.showFunctionMinimize()
        strategy.showFunctionMinimizeGradient()
        strategy.clearGraphics()

	db.save("read.txt")


def solvingOptTaskMode():
	print "solving mode"

def generateTaskMode():
	print "generate mode"
	db.connect()
	random.seed()
	gen_task = Task.Task()
	a = random.randint(0, 200)
	b = random.randint(0, 200)
	ot = gen_task.getGenOptTask(a, b)
	
	db.save("read.txt")

def Ext(event):
	print "exit"

def run():
	choose = 0
	root = Tk()
	but = Button(root)
	but1 = Button(root)
	but2 = Button(root)
	but3 = Button(root)
	but4 = Button(root)
	but5 = Button(root)

	but["text"] = "Learning"
	but["width"] = 30
	but["height"] = 5
	but["bg"] = "blue"
	but["fg"] = "white"
	
	but1["text"] = "Testing"
	but1["width"] = 30
        but1["height"] = 5
        but1["bg"] = "blue"
        but1["fg"] = "white"
#
	but2["text"] = "Solve"
	but2["width"] = 30
        but2["height"] = 5
        but2["bg"] = "blue"
        but2["fg"] = "white"
#
	but3["text"] = "Generate"
	but3["width"] = 30
        but3["height"] = 5
        but3["bg"] = "blue"
        but3["fg"] = "white"

	but4["text"] = "ShowFunctionMinimize"
        but4["width"] = 30
        but4["height"] = 5
        but4["bg"] = "blue"
        but4["fg"] = "white"

	but5["text"] = "ShowGradientFunc"
        but5["width"] = 30
        but5["height"] = 5
        but5["bg"] = "blue"
        but5["fg"] = "white"
#
	but.bind("<Button-1>", learningMode)
	but1.bind("<Button-1>", testingMode)
	but2.bind("<Button-1>", solvingOptTaskMode)
	but3.bind("<Button-1>", generateTaskMode)
#	
	but.pack()
	but1.pack()
	but2.pack()
	but3.pack()
	but4.pack()
	but5.pack()
#
	#root.mainloop()

	while choose != 5:
		choose = input("Choose application mode ")
		if choose == 1:
			print "1 mode"
			learningMode()
		if choose == 2:
			print "2 mode"
			testingMode()
		if choose == 3:
			print "3 mode"
			solvingOptTaskMode()
		if choose == 4:
			print "4 mode"
			generateTaskMode()
		if choose == 5:
			print "exit"
run()
