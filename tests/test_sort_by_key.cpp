
// Test of wrapper around thrust::sort_by_key

#include "ompx/sort.h"

#include <stdlib.h>
#include <iostream>

#define DTYPE int

#define N 13

#pragma omp begin declare target
class Cmp {
public:
  bool operator()(DTYPE a, DTYPE b) const {
    // reverse it for fun
    return a > b;
  }
};

bool cmp(void *a, void *b) {
  Cmp c;
  return c(*(DTYPE *)a, *(DTYPE *)b);
}
#pragma omp end declare target

void init(DTYPE* keys, double* values)
{
  for (int i = 0; i < N; ++i) {
    keys[i] = rand();
    values[i] = (double)keys[i];
  }

}

int main() {

  DTYPE keys[N];
  DTYPE *keys_begin = &keys[0];
  
  double values[N];
  int NumKeys = N; //sizeof(keys) / sizeof(keys[0]);


  //std::cout << "Sorting " << typeid(keys).name() << " via thrust in default device " << omp_get_default_device() << " from " << omp_get_num_devices() << " initial device " << omp_get_initial_device() << std::endl;
  int errors = 0;

  init(keys,values);
  
  #pragma omp target enter data map(to : keys_begin[ : NumKeys]) map(to: values[: NumKeys])
  {
    decltype(cmp) *dev_fptr = nullptr;

    #pragma omp target map(from : dev_fptr)
    dev_fptr = &cmp;
    #ifdef SORTDEV 
      ompx::sort_by_key(ompx::device,keys_begin, keys_begin + NumKeys, values, dev_fptr);
    #else
      ompx::sort_by_key(keys_begin, keys_begin + NumKeys, values, dev_fptr);
    #endif
    
  }
  #pragma omp target exit data map(from : keys_begin[ : NumKeys]) map(from: values[: NumKeys])

  //std::cout << "Results (reverse sort):" << std::endl;
    
  for (int i = 1; i < NumKeys; i++) {
    if (keys[i] > keys[i-1] || values[i-1] > values[i-1]) errors++;
      //std::cout << i << " " << keys[i] << "  " << values[i] << std::endl;
  }

  if (errors)
    std::cout << "Test FAIL" << std::endl;
  else
    std::cout << "Test PASS" << std::endl;
  return errors;
}
