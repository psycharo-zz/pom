#include "pom.h"

#include <string>
#include <map>
#include <set>

#include <opencv2/core.hpp>

#include "util.h"


void usage()
{
  std::cout << "pom <data_path> <config_path> <begin frame id> <end frame id>" << std::endl;
}

int main(int argc, char *argv[])
{
  using namespace std;

  if (argc < 5)
  {
    usage();
    return -1;
  }
  
  string data_path = argv[1];
  string config_path = data_path + argv[2];
  int begin_fid = stoi(argv[3]);
  int end_fid = stoi(argv[4]);

  auto read_tags = [] (const string &path, const set<string> &tags) 
  {
    ifstream ifs(path);
    map<string, string> result;
    string str;
    while (getline(ifs, str))
    {
      if (str.size() == 0 || str[0] == '#')
	continue;
      istringstream iss(str);
      string key, value;
      iss >> key >> value;
      if (tags.count(key))
	result.emplace(key, value);
    }
    return result;
  };

  auto tags = read_tags(config_path,
			{"INPUT_VIEW_FORMAT", "RESULT_FORMAT"});  

  auto read_frame = [&] (size_t fid, size_t n_views, size_t H, size_t W)
  {
    vector<arma::mat> views;
    for (size_t vid = 0; vid < n_views; ++vid)
    {
      auto path = string_format(data_path + tags["INPUT_VIEW_FORMAT"], vid, fid);
      cv::Mat_<uint8_t> view = cv::imread(path, cv::IMREAD_GRAYSCALE) / 255;
      cv::resize(view, view, cv::Size(W, H));
      views.push_back(arma::conv_to<arma::mat>::from(to_arma(view)));
//      cv::imwrite(string_format(data_path + "output/%04d.png", vid), 255 * to_cvmat(views.back())); 
    }
    return views;
  };

  auto write_probs = [] (const string &path, const arma::vec &q)
  {
    ofstream ofs(path);
    for (size_t i = 0; i < q.size(); ++i)
      ofs << i << "\t" << q(i) << endl;
  };

  // reading rectangles
  Pom pom(config_path);

//  #pragma omp parallel for 
  for (int fid = begin_fid; fid <= end_fid; ++fid)
  {
    auto views = read_frame(fid, pom.n_views, pom.H, pom.W);

    arma::vec probs, distances;
    std::tie(probs, distances) = pom.detect(views);

    cout << string_format("frame %d converged in %d iterations with distance: %f",
			  fid, distances.size(), distances.min()) << endl;
    
    write_probs(string_format(data_path + tags["RESULT_FORMAT"], fid), probs);
    write_probs(string_format(data_path + tags["RESULT_FORMAT"] + ".dist", fid), distances);    

    
  }
 
  return 0;
}
