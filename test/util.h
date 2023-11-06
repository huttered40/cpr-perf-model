#include <cstdio>
#include <fstream>

/*
#include <format>
#include <string_view>

template<typename... Args>
void print(const std::string_view fmt_str, Args&&... args){
  //auto fmt_args{std::make_format_args(args...)};// Interesting that we do not forward
  auto fmt_args{std::make_format_args(std::forward<Args>(args)...)};
  std::string outstr{ vformat(fmt_str,fmt_args) };
  fputs(outstr.c_str(),stdout);
}
*/

template<typename T>
void print(const char* msg, T val){
  std::cout << msg << val << "\n";
}

template<typename T, typename U>
void print(const char* msg, T val1, U val2){
  std::cout << msg << val1 << ": " << val2 << "\n";
}

void print_model_info(int order, double* info){
/*
  print("Number of distinct configurations: {}\n",info[2]);
  print("Number of tensor elements: {}\n",info[1]);
  for (int i=0; i<tensor_order; i++){
    print("Tensor mode length {}: {}\n",i,info[3+i]);
  }
  print("Tensor density: {}\n",info[0]);
  print("Loss for first model: {}\n",info[3+order]);
  print("Loss for second model (trained using MLogQ2): {}\n",info[4+nparam]);
  print("Quadrature error: {}\n",info[6+nparam]);
  print("Low-rank approximation error on observed tensor elements: {}\n",info[7+nparam]);
  print("Training error: {}\n",info[5+nparam]);
*/
  print("Number of distinct configurations: ",info[2]);
  print("Number of tensor elements: ",info[1]);
  for (int i=0; i<order; i++){
    print("Tensor mode length ",i,info[3+i]);
  }
  print("Tensor density: ",info[0]);
  print("Loss for first model: ",info[3+order]);
  print("Loss for second model (trained using MLogQ2): ",info[4+order]);
  print("Quadrature error: ",info[6+order]);
  print("Low-rank approximation error on observed tensor elements: ",info[7+order]);
  print("Training error: ",info[5+order]);
}

void custom_assert(bool alert, const char* msg){
  if (!alert) fputs(msg,stdout);
}

double get_wall_time(){
  struct timeval time;
  if (gettimeofday(&time,NULL)){
    // Error
    return 0;
  }
  return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

void get_dataset(const char* dataset_file_path, int order, std::vector<double>& configurations, std::vector<double>& runtimes){
  std::ifstream my_file;
  my_file.open(dataset_file_path);

  std::string temp_num;
  // Read in column header
  for (int i=0; i<order+1; i++){
    getline(my_file,temp_num,',');
  }
  getline(my_file,temp_num,'\n');
  while (getline(my_file,temp_num,',')){
    getline(my_file,temp_num,',');
    configurations.push_back(atof(temp_num.c_str()));
    for (int i=1; i<order; i++){
      getline(my_file,temp_num,',');
      configurations.push_back(atof(temp_num.c_str()));
    }
    getline(my_file,temp_num,',');
    runtimes.push_back(atof(temp_num.c_str()));
    getline(my_file,temp_num,'\n');// read in standard deviation
  }
}
