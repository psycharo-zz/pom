#include "pom.h"

#include <string>
#include <map>
#include <set>

#include <opencv2/core.hpp>

#include "util.h"

int main(int argc, char *argv[])
{
  using namespace std;

  if (argc < 5)
    throw runtime_error(string_format("%s: not enough arguments", argv[0]));
  
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

  auto tags = read_tags(config_path, {"INPUT_VIEW_FORMAT", "RESULT_VIEW_FORMAT", "RESULT_FORMAT"});  

  auto read_frame = [&] (size_t fid, size_t n_views, size_t H, size_t W)
  {
    vector<arma::Mat<uint8_t>> views;
    for (size_t vid = 0; vid < n_views; ++vid)
    {
      auto path = string_format(data_path + tags["INPUT_VIEW_FORMAT"], vid, fid);
      cv::Mat_<uint8_t> view = cv::imread(path, cv::IMREAD_GRAYSCALE) / 255;
      cv::resize(view, view, cv::Size(W, H));
      views.push_back(to_arma(view));
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

  for (auto fid = begin_fid; fid != end_fid; ++fid)
  {
    auto views = read_frame(fid, pom.n_views, pom.H, pom.W);
    cout << string(80, '-') << endl;
    cout << "processing frame:" << fid << endl;    
    auto q = pom.detect(views);
    cout << string(80, '-') << endl;    
    write_probs(string_format(data_path + tags["RESULT_FORMAT"], fid), q);
  }
 
  return 0;
}
