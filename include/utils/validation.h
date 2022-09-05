#pragma once

namespace infini {

/*
 * Range: For int datatype tensor, Such as int,int64,int8 and so on.
 * Effect: Count the number of different data in the tensor.
 * Warning: Please use it only for int data.
 */
template <typename T> int computeDifference1(T *baseline, T *test, int num);

/*
 * Range: For float datatype tensor, Such as half,float,double.
 * Effect: Find the maximum absolute error.
 * Warning: Please use it only for float data.
 */
template <typename T> double computeDifference2(T *baseline, T *test, int num);

/*
 * Range: For float datatype tensor, Such as half,float,double.
 * Effect: Find the maximum relative error.
 * Warning: Please use it only for float data.
 */
template <typename T> double computeDifference3(T *baseline, T *test, int num);

/*
 * Range: For float datatype tensor, Such as half,float,double.
 * Effect: Compute the relative error for a tensor.
 * Warning: Please use it only for float data.
 */
template <typename T> double computeDifference4(T *baseline, T *test, int num);

/*
 * Range: For float datatype tensor, Such as half,float,double.
 * Effect: Check for deviations in data errors.
 * Warning: Please use it only for float data.
 */
template <typename T> double computeDifference5(T *baseline, T *test, int num);

} // namespace infini
