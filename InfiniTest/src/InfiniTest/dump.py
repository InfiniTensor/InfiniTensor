import logging
import numpy

class Dump(object):
    def __init__(self, name:str = "default"):
        self.name = name

    def dumpData(self, input_data:numpy.ndarray, case:str = "default", file_path:str="", precision:int=3):
        if file_path == "":
            logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
            numpy.set_printoptions(threshold=numpy.inf, precision=precision)
            logging.info("\033[32m" + "[INFO] " + "\033[0m" + ": " + case + " [Datatype: " + str(input_data.dtype) + ", Shpae: " + str(input_data.shape) + "]")
            logging.info(input_data)
        else:
            logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO, filename=file_path)
            numpy.set_printoptions(threshold=numpy.inf, precision=precision)
            logging.info("[INFO] " + ": " + case + " [Datatype: " + str(input_data.dtype) + ", Shpae: " + str(input_data.shape) + "]")
            logging.info(input_data)

    def unitTest(self):
        a = numpy.random.random(size=(20,10))
        b = a
        self.dumpData(case="case", input_data=a, precision=2)

if __name__ == "__main__":
    dump = Dump()
    dump.unitTest()


