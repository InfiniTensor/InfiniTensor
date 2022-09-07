#include <algorithm>
#include <math.h>
#include <utils/validation.h>

namespace infini {

const double EPSILON = 1e-9;
const double EPSILON_FLOAT = 1e-6;
const double EPSILON_HALF = 1e-3;

template <typename T> int computeDifference1(T *baseline, T *test, int num) {
    int error = 0;
    for (int i = 0; i < num; ++i) {
        error += (baseline[i] != test[i] ? 1 : 0);
    }
    return error;
}

template <typename T> double computeDifference2(T *baseline, T *test, int num) {
    double max_error = 0;
    for (int i = 0; i < num; ++i) {
        double temp_error = fabs((double)baseline[i] - (double)test[i]);
        max_error = std::max(max_error, temp_error);
    }
    return max_error;
}

template <typename T> double computeDifference3(T *baseline, T *test, int num) {
    double max_error = 0;
    for (int i = 0; i < num; ++i) {
        double temp_error = fabs((double)baseline[i] - (double)test[i]) /
                            (fabs((double)baseline[i]) + EPSILON);
        max_error = std::max(max_error, temp_error);
    }
    return max_error;
}

template <typename T> double computeDifference4(T *baseline, T *test, int num) {
    double up_sum = 0.0;
    double down_sum = 0.0;
    for (int i = 0; i < num; ++i) {
        up_sum += fabs((double)baseline[i] - (double)test[i]);
        down_sum += fabs((double)baseline[i]);
    }
    return up_sum / (down_sum + EPSILON);
}

template <typename T> double computeDifference5(T *baseline, T *test, int num) {
    int small = 0;
    int down = 0;
    for (int i = 0; i < num; ++i) {
        if (baseline[i] != test[i]) {
            down += 1;
            small += (test[i] < baseline[i] ? 1 : 0);
        }
    }
    return (double)small / (double)down;
}

template int computeDifference1<int>(int *baseline, int *test, int num);
template double computeDifference2<float>(float *baseline, float *test,
                                          int num);
template double computeDifference2<double>(double *baseline, double *test,
                                           int num);
template double computeDifference3<float>(float *baseline, float *test,
                                          int num);
template double computeDifference3<double>(double *baseline, double *test,
                                           int num);
template double computeDifference4<float>(float *baseline, float *test,
                                          int num);
template double computeDifference4<double>(double *baseline, double *test,
                                           int num);
template double computeDifference5<float>(float *baseline, float *test,
                                          int num);
template double computeDifference5<double>(double *baseline, double *test,
                                           int num);

} // namespace infini
