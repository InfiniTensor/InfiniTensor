import logging
import os
import inspect
import numpy
import utils

class Accuracy(object):
    def __init__(self, name:str = "default", diff1_threshold:float = 0.003, diff2_threshold:float = 0.003, diff3_threshold:float = 0.003, diff4_threshold:list = [0.400, 0.600]):
        self.name = name
        self.diff1_threshold = numpy.float64(diff1_threshold)
        self.diff2_threshold = numpy.float64(diff2_threshold)
        self.diff3_threshold = numpy.float64(diff3_threshold)
        self.diff4_threshold = [numpy.float64(diff4_threshold[0]), numpy.float64(diff4_threshold[1])]

    def basicCheck(self, case:str, base:numpy.ndarray, test:numpy.ndarray):
        logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
        baseLength = base.size
        testLength = test.size
        if baseLength != testLength:
            logging.info("\033[31m" + "[UNPASSED] " + "\033[0m" + ": " + case + " The length of two inputs are not equal.")
            return 1;
        if baseLength == 0 or testLength == 0:
            logging.info("\033[33m" + "[WARNNING] " + "\033[0m" + ": " + case + " The length of two inputs are zero.")
            return 1;
        if base.dtype != test.dtype:
            logging.info("\033[31m" + "[UNPASSED] " + "\033[0m" + ": " + case + " The data type of two inputs are different.")
            return 1;
        return 0;

    def computeDifference0(self, case:str, base:numpy.ndarray, test:numpy.ndarray):
        """
        逐个检查 base 中的元素与 test 中的元素是否完全一致。
        """
        state = self.basicCheck(case, base, test)
        if state == 1:
            return;
        logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
        baseCopy = base.ravel()
        testCopy = test.ravel()
        if (baseCopy != testCopy).any():
            logging.info("\033[31m" + "[UNPASSED] " + "\033[0m" + ": " + case + " The value of two inputs are not equal.")
            return;
        logging.info("\033[32m" + "[PASSED] " + "\033[0m" + ": " + case)

    def computeDifference1(self, case:str, base:numpy.ndarray, test:numpy.ndarray):
        """
        逐个检查 base 中的元素与 test 中的元素，计算获得最大单点绝对误差。
        error_value = max( |base[i] - test[i]| )
        """
        state = self.basicCheck(case, base, test)
        if state == 1:
            return;
        logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
        baseCopy = base.astype(numpy.float64).ravel()
        testCopy = test.astype(numpy.float64).ravel()
        result = numpy.abs(baseCopy - testCopy)
        maxError = numpy.max(result)
        if maxError > self.diff1_threshold:
            logging.info("\033[31m" + "[UNPASSED] " + "\033[0m" + ": " + case + " The error " + str(maxError) + " is out of the threshold " + str(self.diff1_threshold) + ".")
        else:
            logging.info("\033[32m" + "[PASSED] " + "\033[0m" + ": " + case + " The error is " + str(maxError))

    def computeDifference2(self, case:str, base:numpy.ndarray, test:numpy.ndarray):
        """
        逐个检查 base 中的元素与 test 中的元素，计算获得最大单点相对误差。
        error_value = max( |base[i] - test[i]| / |base[i]| + EPSILON )
        """
        state = self.basicCheck(case, base, test)
        if state == 1:
            return;
        logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
        baseCopy = base.astype(numpy.float64).ravel()
        testCopy = test.astype(numpy.float64).ravel()
        upValue = numpy.abs(baseCopy - testCopy)
        downValue = numpy.abs(baseCopy) + utils.FLOAT64_EPSILON
        result = upValue / downValue
        maxError = numpy.max(result)
        if maxError > self.diff2_threshold:
            logging.info("\033[31m" + "[UNPASSED] " + "\033[0m" + ": " + case + " The error " + str(maxError) + " is out of the threshold " + str(self.diff2_threshold) + ".")
        else:
            logging.info("\033[32m" + "[PASSED] " + "\033[0m" + ": " + case + " The error is " + str(maxError))

    def computeDifference3(self, case:str, base:numpy.ndarray, test:numpy.ndarray):
        """
        计算获得平均相对误差。
        error_value = ( ∑|base[i] - test[i]| ) / ( ∑|base[i]| + EPSILON )
        """
        state = self.basicCheck(case, base, test)
        if state == 1:
            return;
        logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
        baseCopy = base.astype(numpy.float64).ravel()
        testCopy = test.astype(numpy.float64).ravel()
        upValue = numpy.sum(numpy.abs(baseCopy - testCopy))
        downValue = numpy.sum(numpy.abs(baseCopy)) + utils.FLOAT64_EPSILON
        maxError = upValue / downValue
        if maxError > self.diff3_threshold:
            logging.info("\033[31m" + "[UNPASSED] " + "\033[0m" + ": " + case + " The error " + str(maxError) + " is out of the threshold " + str(self.diff3_threshold) + ".")
        else:
            logging.info("\033[32m" + "[PASSED] " + "\033[0m" + ": " + case + " The error is " + str(maxError))

    def computeDifference4(self, case:str, base:numpy.ndarray, test:numpy.ndarray):
        """
        误差有偏性度量
        not_equal_num = count( base[i]! = test[i] )
        less_num = count( base[i] < test[i] )
        error_value = less_num / not_equal_num
        """
        state = self.basicCheck(case, base, test)
        if state == 1:
            return;
        logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
        baseCopy = base.astype(numpy.float64).ravel()
        testCopy = test.astype(numpy.float64).ravel()
        down = numpy.count_nonzero(numpy.not_equal(baseCopy, testCopy))
        small = numpy.count_nonzero(numpy.less(baseCopy, testCopy))
        error = small / down
        if error < self.diff4_threshold[0] or error > self.diff4_threshold[1]:
            logging.info("\033[31m" + "[UNPASSED] " + "\033[0m" + ": " + case + " The error " + str(error) + " is out of the threshold [" + str(self.diff4_threshold[0]) + ", " + str(self.diff4_threshold[1]) +"].")
        else:
            logging.info("\033[32m" + "[PASSED] " + "\033[0m" + ": " + case + " The error is " + str(maxError))

    def unitTest(self):
        a = numpy.array([1,2,3])
        b = a
        self.computeDifference0("passed test case", a, b)
        b = numpy.array([1,2,3])
        self.computeDifference0("passed test case", a, b)
        b = numpy.array([1])
        self.computeDifference0("unequal length case", a, b)
        b = numpy.array([])
        c = numpy.array([]) 
        self.computeDifference0("zero length case", b, c)
        b = numpy.array([1.0,2.0,3.0])
        self.computeDifference0("different data type case", a, b)
        b = numpy.array([2,3,4])
        self.computeDifference0("unpassed test case", a, b)
        b = numpy.array([1.0,2.0,3.0])
        c = numpy.array([2.0,3.0,4.0])
        self.computeDifference1("unpassed test case", b, c)
        self.computeDifference1("passed test case", b, b)
        self.computeDifference2("unpassed test case", b, c)
        self.computeDifference3("unpassed test case", b, c)
        b = numpy.array([1.1,2.0,3.1])
        c = numpy.array([1.0,3.0,3.0])
        self.computeDifference4("unpassed test case", b, c)
        

if __name__ == "__main__":
    check = Accuracy()
    check.unitTest()


