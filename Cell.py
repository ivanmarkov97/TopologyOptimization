class Cell:
    var = {'density': 0, 'variable': {'v':10, 'p':1000}, 'xyz': {'x':0.0, 'y':0.0, 'z':0.0}}
    objective_func = (1 - var['xyz']['x'])**2 + 100*(var['xyz']['y'] - var['xyz']['x']**2)**2
    #density = 1.0
    #variables = {'v':10 , 'p': 1000}
    #xyz= (0.0,0.0,0)

    def __init__(self):
	#print "init"
	self.var['density'] = 0
	self.var['variable']['v'] = 0
	self.var['variable']['p'] = 0
	self.var['xyz']['x'] = 0.0
	self.var['xyz']['y'] = 0.0
	self.var['xyz']['z'] = 0.0

    def getCell(self):
	print self.var
	return self.var


    #def initCells(self, n, m):
	#cell_list = []
	#for i in range(n*m):
		#cell = Cell()
		#cell_list.append(cell)
	#return cell_list
	

#c = Cell()
#l = c.initCells(5, 5)
#for i in l:
#	print i.getCell()
