#ifndef POM_H
#define POM_H

#include <armadillo>
#include <string>
#include <sstream>
#include <map>

#include "util.h"

/**
 * an instance 
 */
class Pom
{
public:
  // NOTE: we don't use integral images but this will allows using arbitrary silhouettes
  class Rectangle
  {
  public:
    int y_min, x_min, y_max, x_max;
    
    inline double area() const
    {
      return (x_max - x_min + 1) * (y_max - y_min + 1);
    }
  };

  Pom() = delete;

  Pom(const Pom &) = delete;

  // TODO: add constructor from camera parameters etc
  Pom(const std::string &config_path)
  {
    using namespace std;
    ifstream ifs(config_path);
    if (!ifs.is_open())
      throw runtime_error("Pom: can't open the file:" + config_path);

    string str;
    while (getline(ifs, str))
    {
      if (str.size() == 0 || str[0] == '#')
	continue;
      istringstream iss(str);
      string tag;      
      iss >> tag;

      if (tag == "ROOM")
      {
	iss >> W >> H >> n_views >> n_locs;
	rects.resize(n_views);
      }
      else if (!rects.empty() && tag == "RECTANGLE")
      {
	if (str.substr(str.rfind(' ')+1) == "notvisible")
	  continue;
	Rectangle r;
	size_t vid = 0, lid = 0;
	iss >> vid >> lid >> r.x_min >> r.y_min >> r.x_max >> r.y_max;
	r.y_min *= scale;
	r.x_min *= scale;
	r.y_max *= scale;
	r.x_max *= scale;

	if (lid >= n_locs)
	  throw runtime_error("Pom: location id out of bounds");
	
	if (r.y_min < 0 || r.y_max >= H || r.x_min < 0 || r.x_max >= W || r.area() <= 0)
	  continue;
	rects[vid].insert({lid, r});
      }
    }
  }
  
  // run detector on a single (multi-view) frame. TODO: see if initial values for q-s are needed
  arma::vec detect(const std::vector<arma::Mat<uint8_t>> &views)
  {
    using namespace arma;
    // average image for each view
    std::vector<mat> average(n_views, mat(H,W,fill::ones));

    // Q(X_{1:K} = 1)
    vec q = ones(n_locs) * p_prior;
    // natural parameter of Q(X_{1:K})
    vec nat_q = ones(n_locs) * log(p_prior / (1.0 - p_prior));
    // natural parameter update for a single iteration
    vec nat_q_curr(n_locs);

    vec nat_prior = ones(n_locs) * log(p_prior / (1.0 - p_prior));

    auto update_average_image = [&] ()
    {
      for (size_t vid = 0; vid < n_views; ++vid)
      {
	average[vid].fill(1.0);
	for (auto &p : rects[vid])
	{
	  auto lid = p.first;
	  const auto &r = p.second;
	  average[vid].submat(r.y_min, r.x_min, r.y_max, r.x_max) *= (1.0 - q(lid));
	}
      }
    };

    auto update_posterior = [&] ()
    {
      nat_q_curr.fill(0.0);
      
      for (size_t vid = 0; vid < n_views; ++vid)
      {
	for (auto &p : rects[vid])
	{
	  auto lid = p.first;
	  const auto &r = p.second;

	  auto B = views[vid].submat(r.y_min, r.x_min, r.y_max, r.x_max);
	  auto nA = average[vid].submat(r.y_min, r.x_min, r.y_max, r.x_max);
	  auto A = 1.0 - nA / (1.0 - q(lid) + p_eps);

	  double nat_1 = -accu(1 - B);
	  double nat_0 = -accu((B % (1 - A) + (1 - B) % A));

	  nat_q_curr(lid) += (nat_1 - nat_0) / accu(A) / sigma;
	}
      }
      nat_q_curr = nat_q_curr + nat_prior;

      // natural gradient update
      nat_q = (1.0 - step) * nat_q + step * nat_q_curr;
      //q = (1.0 - step) * q + step * 1.0 / (1 + exp(-nat_q_curr));
      q = 1.0 / (1.0 + exp(-nat_q));

      // classical pom update
      //q = (1.0 - step) * q + step * 1.0 / (1 + exp(-nat_q_curr));
    };

    // running the inference
    for (size_t it = 0; it < n_max_iters; ++it)
    {
      cout << string_format("current iteration: %d\n", it);
      update_average_image();
      update_posterior();
      for (size_t v = 0; v < views.size(); ++v)
	cv::imwrite(string_format("/tmp/average_%d_%d.png", v, it), 255 * to_cvmat(average[v]));
    }
    return q;
  }

  // total number of locations 
  size_t n_locs;
  size_t n_views;
  size_t H, W;

protected:  
  const double sigma = 0.5;
  const double p_prior = 1e-4;
  const double p_eps = 1e-5;
  const size_t n_max_iters = 100;
  const double max_error = 1e-1;
  const double step = 0.1;
  const double scale = 1;
  
  // a number of silhouettes
  std::vector<std::map<int, Rectangle>> rects;
};


  



#endif // POM_H
