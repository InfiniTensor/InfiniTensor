import time
import logging
import device

class Profiling():
    def __init__(self):
        self.host_start = None
        self.host_end = None
        self.cuda_queue = None
        self.bang_queue = None

    def setBangQueue(self, queue):
        self.bang_queue = queue

    def setCudaQueue(self, queue):
        self.cuda_queue = queue

    def hostProfilingWrapper(self, times:int = 1):
        def hostProfiling(func):
            def wrapper(*args, **kwargs):
                logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
                logging.info("\033[32m" + "[INFO] " + "\033[0m" + ": " + "Start Host profiling " + str(func.__name__) + ".")
                startTime = time.time()
                for i in range(times):
                    func(*args, **kwargs)
                endTime = time.time()
                totalTime = (endTime - startTime) * 1000
                averageTime = totalTime / times
                logging.info("\033[32m" + "[INFO] " + "\033[0m" + ": " + "End profiling, Run " + str(times) + " times, Total " + str(totalTime) + " ms, Average " + str(averageTime) + "ms.")
            return wrapper
        return hostProfiling

    def bangProfilingWrapper(self, times:int = 1):
        def bangProfiling(func):
            def wrapper(*args, **kwargs):
                logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
                logging.info("\033[32m" + "[INFO] " + "\033[0m" + ": " + "Start Bang device profiling " + str(func.__name__) + ".")
                device.bangPlaceStartNotifier(self.bang_queue)
                for i in range(times):
                    func(*args, **kwargs)
                device.bangPlaceEndNotifier(self.bang_queue)
                totalTime = device.bangGetNotifierDuration(self.bang_queue) / 1000
                averageTime = totalTime / times
                logging.info("\033[32m" + "[INFO] " + "\033[0m" + ": " + "End profiling, Run " + str(times) + " times, Total " + str(totalTime) + " ms, Average " + str(averageTime) + "ms.")
            return wrapper
        return bangProfiling
    
    def cudaProfilingWrapper(self, times:int = 1):
        def cudaProfiling(func):
            def wrapper(*args, **kwargs):
                logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
                logging.info("\033[32m" + "[INFO] " + "\033[0m" + ": " + "Start Cuda device profiling " + str(func.__name__) + ".")
                device.cudaPlaceStartNotifier(self.cuda_queue)
                for i in range(times):
                    func(*args, **kwargs)
                device.cudaPlaceEndNotifier(self.cuda_queue)
                totalTime = device.cudaGetNotifierDuration(self.cuda_queue)
                averageTime = totalTime / times
                logging.info("\033[32m" + "[INFO] " + "\033[0m" + ": " + "End profiling, Run " + str(times) + " times, Total " + str(totalTime) + " ms, Average " + str(averageTime) + "ms.")
            return wrapper
        return cudaProfiling


    def hostProfilingStart(self):
        logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
        logging.info("\033[32m" + "[INFO] " + "\033[0m" + ": " + "Start Host profiling. ")
        self.host_start = time.time()

    def hostProfilingEnd(self):
        logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
        self.host_end = time.time()
        totalTime = (self.host_end - self.host_start) * 1000
        logging.info("\033[32m" + "[INFO] " + "\033[0m" + ": " + "End profiling, Run 1 times, Total " + str(totalTime) + " ms.")

    def bangProfilingStart(self):
        logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
        logging.info("\033[32m" + "[INFO] " + "\033[0m" + ": " + "Start Bang device profiling. ")
        device.bangPlaceStartNotifier(self.bang_queue)

    def bangProfilingEnd(self):
        logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
        device.bangPlaceEndNotifier(self.bang_queue)
        totalTime = device.bangGetNotifierDuration(self.bang_queue) / 1000
        logging.info("\033[32m" + "[INFO] " + "\033[0m" + ": " + "End profiling, Run 1 times, Total " + str(totalTime) + " ms.")

    def cudaProfilingStart(self):
        logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
        logging.info("\033[32m" + "[INFO] " + "\033[0m" + ": " + "Start Cuda device profiling. ")
        device.cudaPlaceStartNotifier(self.cuda_queue)

    def cudaProfilingEnd(self):
        logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
        device.cudaPlaceEndNotifier(self.cuda_queue)
        totalTime = device.cudaGetNotifierDuration(self.cuda_queue)
        logging.info("\033[32m" + "[INFO] " + "\033[0m" + ": " + "End profiling, Run 1 times, Total " + str(totalTime) + " ms.")


def unitTest():
    pro = Profiling()

    @pro.hostProfilingWrapper(times=10)
    def helloHost():
        time.sleep(0.1)

    @pro.hostProfilingWrapper(times=10)
    def byebye(a:int,b:int):
        device.add(i=a,j=b)

    @pro.bangProfilingWrapper(times=10)
    def helloBang():
        time.sleep(0.1)

    @pro.cudaProfilingWrapper(times=1)
    def helloCuda():
        time.sleep(0.1)

    device.cudaCreateQueue()
    queue = device.cudaGetQueue()
    pro.setCudaQueue(queue)

    helloHost()
    byebye(2,3)

    pro.hostProfilingStart()
    print("These are soem code.")
    pro.hostProfilingEnd()

    #helloBang()
    #helloCuda()

    #pro.bangProfilingStart()
    #time.sleep(0.1)
    #pro.bangProfilingEnd()

    #pro.cudaProfilingStart()
    #time.sleep(0.1)
    #pro.cudaProfilingEnd()


if __name__ == "__main__":
    unitTest()
 
