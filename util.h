#ifndef IMAGEUTIL_H
#define IMAGEUTIL_H

#include <cstdarg>
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

inline arma::mat integral_image(const arma::mat &a)
{
  using namespace arma;
  mat a_ii = zeros(a.n_rows+1, a.n_cols+1);
  for (size_t j = 0; j < a.n_rows; ++j)
  {
    double sum = 0.0;
    for (size_t i = 0; i < a.n_cols; ++i)
    {
      sum += a(j,i);
      a_ii(j+1,i+1) = sum + a_ii(j,i+1);
    }
  }
  return a_ii;
}

inline void integral_image(const arma::mat &a, arma::mat &a_ii)
{
  using namespace arma;
  for (size_t j = 0; j < a.n_rows; ++j)
  {
    double sum = 0.0;
    for (size_t i = 0; i < a.n_cols; ++i)
    {
      sum += a(j,i);
      a_ii(j+1,i+1) = sum + a_ii(j,i+1);
    }
  }
}

inline double integral_accu(const arma::mat &ii, size_t ymin, size_t xmin, size_t ymax, size_t xmax)
{
  return ii(ymax+1,xmax+1) - ii(ymin,xmax+1) - ii(ymax+1,xmin) + ii(ymin,xmin);
}

inline double integral_accu(const arma::mat &ii)
{
  return ii(ii.n_rows-1, ii.n_cols-1);
}



#endif
