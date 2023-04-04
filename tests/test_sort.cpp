
// Test of wrapper around thrust::sort

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

void init(DTYPE* keys)
{
  for (int i = 0; i < N; ++i) {
    keys[i] = rand();
  }
}

int main() {

  DTYPE keys[N];
  DTYPE *keys_begin = &keys[0];  

  int NumKeys = N; //sizeof(keys) / sizeof(keys[0]);
  int errors = 0;

  init(keys);

  #pragma omp target enter data map(to : keys_begin[ : NumKeys]) 
  {
    // We can't map function pointers automatically right now. We should though.
    // Workaround: We should instead use the use_device_ptr trick later.
    decltype(cmp) *dev_fptr = nullptr;

    #pragma omp target map(from : dev_fptr)
    dev_fptr = &cmp;
    #ifdef SORTDEV
      #ifdef SORTFUN
      ompx::sort(ompx::device, keys_begin, keys_begin + NumKeys, dev_fptr);
      #else
      ompx::sort(ompx::device, keys_begin, keys_begin + NumKeys);
      #endif
    #else
      #ifdef SORTFUN
      ompx::sort(keys_begin, keys_begin + NumKeys, dev_fptr);
      #else
      ompx::sort(keys_begin, keys_begin + NumKeys);
      #endif
    #endif
  }
  #pragma omp target exit data map(from : keys_begin[ : NumKeys]) 

  for (int i = 1; i < NumKeys; i++) {
    #ifdef SORTFUN
    if (keys[i] > keys[i-1]) errors++;
    #else
    if (keys[i] < keys[i-1]) errors++;
    #endif
    //std::cout << i << " " << keys[i] << std::endl;
  }
  if (errors) std::cout << "Test FAIL" << std::endl;
  else std::cout << "Test PASS" << std::endl;
  return errors;
}

