#ifndef OMPX_SORT_H
#define OMPX_SORT_H

#include <iostream>
#include <omp.h>

namespace ompx {

enum ompx_device_t {
  host   = 0x1,
  device = 0x10
};

typedef bool (*ompx_sort_cmp_ty)(void *, void *);
void ompx_sort_impl_dev(void *B, void *E, uint32_t size, ompx_sort_cmp_ty F);
void ompx_sort_impl_host(void *B, void *E, uint32_t size, ompx_sort_cmp_ty F);

void ompx_sort_by_key_impl_dev(void *B, void *E, uint32_t sizeK, void *V, uint32_t sizeV, ompx_sort_cmp_ty F);
void ompx_sort_by_key_impl_host(void *B, void *E, uint32_t sizeK, void *V, uint32_t sizeV, ompx_sort_cmp_ty F);


#pragma omp begin declare target
template <typename T> class __Cmp {
public:
  bool operator()(T a, T b) const {
    return a < b;
  }
};

template <typename T> 
bool __cmp(void *a, void *b) {
  __Cmp<T> c;
  return c(*(T *)a, *(T *)b);
}
#pragma omp end declare target

// TODO: replace int with an integer type
template <typename T>
void sort(T *B, uint32_t NumElements, ompx_sort_cmp_ty Cmp) {
  int ndev = omp_get_num_devices();
  int present = 0;
  if (ndev > 0)  //check if the data is present in any of the devices
    for (int dev=0; dev < ndev; ++dev) { 
      present = omp_target_is_present(B,dev);
      if (present) break;
    }
    
  if (ndev > 0 && present) {
    #pragma omp target data use_device_ptr(B)
    ompx_sort_impl_dev((void *)B, (void *)(B + NumElements), sizeof(T), Cmp);
  } else {
    ompx_sort_impl_host((void *)B, (void *)(B + NumElements), sizeof(T), Cmp);
  }
}

template <typename T>
void sort(ompx_device_t device_t, T *B, uint32_t NumElements, ompx_sort_cmp_ty Cmp) {

  switch (device_t) {
    case device:
      #pragma omp target data use_device_ptr(B)
      ompx_sort_impl_dev((void *)B, (void *)(B + NumElements), sizeof(T), Cmp);
      break;
    case host:
      ompx_sort_impl_host((void *)B, (void *)(B + NumElements), sizeof(T), Cmp);
      break;
    default:
      std::cout << "ompx device " << device << " not supported" << std::endl;
  }
}


template <typename T> void sort(T *B, T *E, ompx_sort_cmp_ty Cmp) {
  sort(B, E - B, Cmp);
}

template <typename T> void sort(ompx_device_t device, T *B, T *E, ompx_sort_cmp_ty Cmp) {
  sort(device, B, E - B, Cmp);
}

template <typename T> void sort(T *B, T *E) {
  ompx_sort_cmp_ty dev_fptr = nullptr;
  #pragma omp target map(from : dev_fptr)
  dev_fptr = &__cmp<T>;
  sort(B, E - B, dev_fptr);
}

template <typename T> void sort(ompx_device_t device, T *B, T *E) {
  ompx_sort_cmp_ty dev_fptr = nullptr;
  #pragma omp target map(from : dev_fptr)
  dev_fptr = &__cmp<T>;
  sort(device, B, E - B, dev_fptr);
}

template <typename T> void sort(T *B, uint32_t NumElements) {
  ompx_sort_cmp_ty dev_fptr = nullptr;
  #pragma omp target map(from : dev_fptr)
  dev_fptr = &__cmp<T>;
  sort(B, NumElements, dev_fptr);
}

template <typename T> void sort(ompx_device_t device, T *B, uint32_t NumElements) {
  ompx_sort_cmp_ty dev_fptr = nullptr;
  #pragma omp target map(from : dev_fptr)
  dev_fptr = &__cmp<T>;
  sort(device, B, NumElements, dev_fptr);
}



// SORT_BY_KEY

template <typename T1, typename T2>
void sort_by_key(T1 *B, uint32_t NumElements, T2 *V, ompx_sort_cmp_ty Cmp) {
  int ndev = omp_get_num_devices();
  int present = 0;
  if (ndev > 0)  //check if the data is present in any of the devices
    for (int dev=0; dev < ndev; ++dev) {
      present = omp_target_is_present(B,dev);
      if (present) break;
    }
    
  if (ndev > 0 && present) {
    #pragma omp target data use_device_ptr(B,V)
    ompx_sort_by_key_impl_dev((void *)B, (void *)(B + NumElements), sizeof(T1), (void *)V, sizeof(T2), Cmp);
  } else {
    ompx_sort_by_key_impl_host((void *)B, (void *)(B + NumElements), sizeof(T1), (void *)V, sizeof(T2), Cmp);
  }
}

template <typename T1, typename T2>
void sort_by_key(ompx_device_t _device, T1 *B, uint32_t NumElements, T2 *V, ompx_sort_cmp_ty Cmp) {
  switch (_device)
  {
    case device:
      #pragma omp target data use_device_ptr(B,V)
      ompx_sort_by_key_impl_dev((void *)B, (void *)(B + NumElements), sizeof(T1), (void *)V, sizeof(T2), Cmp);
      break;
    case host:
      ompx_sort_by_key_impl_host((void *)B, (void *)(B + NumElements), sizeof(T1), (void *)V, sizeof(T2), Cmp);
      break;
    default:
      std::cout << "ompx device " << _device << " not supported" << std::endl;
  } 
}

template <typename T1, typename T2> void sort_by_key(T1 *B, T1 *E, T2 *V, ompx_sort_cmp_ty Cmp) {
  sort_by_key(B, E - B, V, Cmp);
}

template <typename T1, typename T2> void sort_by_key(ompx_device_t _device, T1 *B, T1 *E, T2 *V, ompx_sort_cmp_ty Cmp) {
  sort_by_key(_device, B, E - B, V, Cmp);
}


} // ompx
  //
#endif
