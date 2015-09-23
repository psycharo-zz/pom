#ifndef POM_H
#define POM_H

#include <string>
#include <sstream>
#include <map>
#include <limits>
#include <armadillo>

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
      else if (tag == "STEP")
	iss >> step;
      else if (tag == "OPTIMIZER")
      {
	iss >> tag;
	if (tag == "NATURAL")
	  m_optimizer = Optimizer::NATURAL;
	else if (tag == "MEAN")
	  m_optimizer = Optimizer::MEAN;
	else
	  throw runtime_error(string_format("Pom: unknown optimizer: %s", tag.c_str()));
      }
      else if (!rects.empty() && tag == "RECTANGLE")
      {
	if (str.substr(str.rfind(' ')+1) == "notvisible")
	  continue;
	Rectangle r;
	size_t vid = 0, lid = 0;
	iss >> vid >> lid >> r.x_min >> r.y_min >> r.x_max >> r.y_max;

	if (lid >= n_locs)
	  throw runtime_error("Pom: location id out of bounds");
	
	if (r.y_min < 0 || r.y_max >= H || r.x_min < 0 || r.x_max >= W || r.area() <= 0)
	  continue;
	rects[vid].push_back({lid, r});
      }
    }
  }

  /** 
   * run detector on a single (multi-view) frame. 
   * TODO: returns a tuple (probabilites, number of iterations, distance)
   * TODO: see if initial values for q-s are needed
   */
  std::pair<arma::vec, arma::vec> detect(const std::vector<arma::mat> &views)
  {
    using namespace arma;
    using namespace std;

    // average image for each view 
    vector<mat> average(n_views, mat(H,W,fill::ones));
    // integral images
    vector<mat> A_ii(n_views, mat(H+1,W+1));
    vector<mat> AB_ii(n_views, mat(H+1,W+1));
    
    // Q(X_{1:K} = 1)
    vec q = ones(n_locs) * p_prior;
    // natural parameter of Q(X_{1:K})
    vec nat_q = ones(n_locs) * log(p_prior / (1.0 - p_prior));
    // natural parameter update for a single iteration
    vec nat_q_curr(n_locs);

    vec nat_prior = ones(n_locs) * log(p_prior / (1.0 - p_prior));

    // distances over iterations
    vector<double> distances;

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

	mat AB = views[vid] % average[vid];	
	integral_image(average[vid], A_ii[vid]);
	integral_image(AB, AB_ii[vid]);
      }
    };

    auto update_nat_q = [&] ()
    {
      auto prev_nat_q  = nat_q_curr;
      nat_q_curr.fill(0.0);

      for (size_t vid = 0; vid < n_views; ++vid)
      {
	for (size_t i = 0; i < rects[vid].size(); ++i)
	{
	  auto lid = rects[vid][i].first;
	  const auto &r = rects[vid][i].second;

	  double p_abs = 1.0 - q(lid) + p_eps;

	  double A_sum = integral_accu(A_ii[vid], r.y_min, r.x_min, r.y_max, r.x_max);
	  double AB_sum = integral_accu(AB_ii[vid], r.y_min, r.x_min, r.y_max, r.x_max);
	  
	  double dnat = (2 * AB_sum - A_sum) / p_abs;
	  nat_q_curr(lid) += dnat / (r.area() - A_sum / p_abs) / sigma;
	}
      }
      nat_q_curr = nat_q_curr + nat_prior;
    };

    auto update_posterior_natural = [&] ()
    {
      // natural gradient update      
      nat_q = (1.0 - step) * nat_q + step * nat_q_curr;
      q = 1.0 / (1.0 + exp(-nat_q));
    };

    auto update_posterior_mean = [&] () 
    {
      // standard pom update
      q = (1.0 - step) * q + step * 1.0 / (1 + exp(-nat_q_curr));
    };

    auto compute_distance = [&] ()
    {
      double distance = 0.0;
      for (size_t vid = 0; vid < n_views; ++vid)
      {
	auto &B = views[vid];
	auto &A = average[vid];
	distance += accu(B % A + (1 - B) % (1 - A));
      }
      return distance;
    };

    // TODO: try better error
    double error = std::numeric_limits<double>::max();

    // running the inference until convergence / max # of iterations
    for (size_t it = 0; it < n_max_iters; ++it)
    {
      update_average_image();
      update_nat_q();

      auto prev_q = q;

      if (m_optimizer == NATURAL)
	update_posterior_natural();
      else if (m_optimizer == MEAN)
	update_posterior_mean();

      error = accu(abs(prev_q - q));
      distances.push_back(compute_distance());
      if (error < max_error)
	break;
    }
    return {q, distances};
  }

  // total number of locations 
  size_t n_locs;
  size_t n_views;
  size_t H, W;
  double step = 0.1;  

protected:
  // TODO: this is better to be set OUTSIDE of the 
  const double sigma = 0.5;
  const double p_prior = 1e-4;
  const double p_eps = 1e-5;
  const size_t n_max_iters = 200;
  const double max_error = 1e-1;

  enum Optimizer
  {
    NATURAL,
    MEAN
  } m_optimizer = NATURAL;

  // a number of silhouettes
  std::vector<std::vector<std::pair<int, Rectangle>>> rects;
};

#endif // POM_H
