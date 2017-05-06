import MySQLdb
import Task

class DB:
	db = None
	def connect(self):
		self.db = MySQLdb.connect(host="localhost", user="root", passwd="i4611366968", db="project", charset='utf8')

	def create (path):
        	pass

	def save (self, result):
        	print "write into file " +  result

	def readSet(self, sets=""):
                lt = []
                self.cursor = self.db.cursor()
                sql = """select * from """ + "Task"
                self.cursor.execute(sql)
                data = self.cursor.fetchall()
                for res in data:
                        lt.append(res)
                return lt



	def readTask(self, path):
		self.cursor = self.db.cursor()
		sql = """select * from """ + path
		self.cursor.execute(sql)
		data = self.cursor.fetchall()
		task = Task.Task()
		for res in data:
			task.input_file = res[1].encode("utf8")
			task.task_type = res[2].encode("utf8")
			task.solver = res[3].encode("utf8")
			task.result_file = res[4].encode("utf8")

			print "task input file: " + task.input_file
			print "task type: " + task.task_type
			print "task solver: " + task.solver
			print "task result file: " + task.result_file


        		f = open(task.input_file, 'r')
        		result = f.read()
        		print "read from file " +  result
		return task

