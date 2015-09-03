#ifndef IMAGEUTIL_H
#define IMAGEUTIL_H

#include <opencv2/opencv.hpp>
#include <armadillo>

// convert from armadillo matrix to opencv
template <typename T>
cv::Mat_<T> to_cvmat(const arma::Mat<T> &src)
{
  return cv::Mat_<T>{int(src.n_cols), int(src.n_rows), const_cast<T*>(src.memptr())}.t();
}

// convert from opencv matrix to armadillo TODO: check if moving can be done
template <typename T>
arma::Mat<T> to_arma(const cv::Mat_<T> &src)
{
  cv::Mat src_t = src.t();
  return arma::Mat<T>{src_t.template ptr<T>(), arma::uword(src.rows), arma::uword(src.cols)};
}

inline std::string string_format(const std::string fmt, ...)
{
  int size = 512;
  std::string str;
  va_list ap;
  while (1)
  {
    str.resize(size);
    va_start(ap, fmt);
    int n = vsnprintf((char *) str.c_str(), size, fmt.c_str(), ap);
    va_end(ap);
    if (n > -1 && n < size)
    {
      str.resize(n);
      return str;
    }
    if (n > -1)
      size = n + 1;
    else
      size *= 2;
  }
  return str;
}

#endif
