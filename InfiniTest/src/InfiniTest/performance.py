import time
import logging
import device

def hostProfilingWrapper(times:int = 1):
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

def bangProfilingWrapper(times:int = 1):
    def bangProfiling(func):
        def wrapper(*args, **kwargs):
            logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
            logging.info("\033[32m" + "[INFO] " + "\033[0m" + ": " + "Start Bang device profiling " + str(func.__name__) + ".")
            device.bangCreateQueue()
            queue = device.bangGetQueue()
            device.bangPlaceStartNotifier(queue)
            for i in range(times):
                func(*args, **kwargs)
            device.bangPlaceEndNotifier(queue)
            totalTime = device.bangGetNotifierDuration(queue) / 1000
            averageTime = totalTime / times
            logging.info("\033[32m" + "[INFO] " + "\033[0m" + ": " + "End profiling, Run " + str(times) + " times, Total " + str(totalTime) + " ms, Average " + str(averageTime) + "ms.")
        return wrapper
    return bangProfiling

def cudaProfilingWrapper(times:int = 1):
    def cudaProfiling(func):
        def wrapper(*args, **kwargs):
            logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
            logging.info("\033[32m" + "[INFO] " + "\033[0m" + ": " + "Start Cuda device profiling " + str(func.__name__) + ".")
            device.cudaCreateQueue()
            queue = device.cudaGetQueue()
            device.cudaPlaceStartNotifier(queue)
            for i in range(times):
                func(*args, **kwargs)
            device.cudaPlaceEndNotifier(queue)
            totalTime = device.cudaGetNotifierDuration(queue)
            averageTime = totalTime / times
            logging.info("\033[32m" + "[INFO] " + "\033[0m" + ": " + "End profiling, Run " + str(times) + " times, Total " + str(totalTime) + " ms, Average " + str(averageTime) + "ms.")
        return wrapper
    return cudaProfiling

def unitTest():

    @hostProfilingWrapper(times=10)
    def helloHost():
        time.sleep(0.1)

    @hostProfilingWrapper(times=10)
    def byebye(a:int,b:int):
        device.add(i=a,j=b)

    @bangProfilingWrapper(times=10)
    def helloBang():
        time.sleep(0.1)

    @cudaProfilingWrapper(times=1)
    def helloCuda():
        time.sleep(0.1)

    helloHost()
    byebye(2,3)
    helloBang()
    #helloCuda()


if __name__ == "__main__":
    unitTest()
 
