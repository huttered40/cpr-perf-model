static double get_wall_time(){
  struct timeval time;
  if (gettimeofday(&time,NULL)){
    // Error
    return 0;
  }
  return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
          
