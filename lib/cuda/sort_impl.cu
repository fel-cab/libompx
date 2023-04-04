
#include "ompx/sort.h"

#include <thrust/sort.h>

struct type_size16_t {
  uint8_t tmp_[16];
};

namespace ompx {

void ompx_sort_impl_dev(void *B, void *E, uint32_t size, ompx_sort_cmp_ty F) {
  printf(
      "sort_impl (cuda file): (device) pointers: %p:%p:%p -> %lu elemnents\n",
      B, E, F, (uintptr_t(E) - uintptr_t(B)) / size);
  switch (size) {
  case 1:
    thrust::sort(thrust::device, (uint8_t *)B, (uint8_t *)E,
                 [=](uint8_t L, uint8_t R) { return F(&L, &R); });
    break;
  case 2:
    thrust::sort(thrust::device, (uint16_t *)B, (uint16_t *)E,
                 [=](uint16_t L, uint16_t R) { return F(&L, &R); });
    break;
  case 4:
    thrust::sort(thrust::device, (uint32_t *)B, (uint32_t *)E,
                 [=](uint32_t L, uint32_t R) { return F(&L, &R); });
    break;
  case 8:
    thrust::sort(thrust::device, (uint64_t *)B, (uint64_t *)E,
                 [=](uint64_t L, uint64_t R) { return F(&L, &R); });
    break;
  case 16:
    thrust::sort(thrust::device, (type_size16_t *)B, (type_size16_t *)E,
                 [=](type_size16_t L, type_size16_t R) { return F(&L, &R); });
    break;
  default:
    printf("Error, size %i not handled\n", size);
  };
}

void ompx_sort_impl_host(void *B, void *E, uint32_t size, ompx_sort_cmp_ty F) {
  printf(
      "sort_impl (cuda file): (host) pointers: %p:%p:%p -> %lu elemnents\n",
      B, E, F, (uintptr_t(E) - uintptr_t(B)) / size);
  switch (size) {
  case 1:
    thrust::sort(thrust::host, (uint8_t *)B, (uint8_t *)E,
                 [=](uint8_t L, uint8_t R) { return F(&L, &R); });
    break;
  case 2:
    thrust::sort(thrust::host, (uint16_t *)B, (uint16_t *)E,
                 [=](uint16_t L, uint16_t R) { return F(&L, &R); });
    break;
  case 4:
    thrust::sort(thrust::host, (uint32_t *)B, (uint32_t *)E,
                 [=](uint32_t L, uint32_t R) { return F(&L, &R); });
    break;
  case 8:
    thrust::sort(thrust::host, (uint64_t *)B, (uint64_t *)E,
                 [=](uint64_t L, uint64_t R) { return F(&L, &R); });
    break;
  case 16:
    thrust::sort(thrust::host, (type_size16_t *)B, (type_size16_t *)E,
                 [=](type_size16_t L, type_size16_t R) { return F(&L, &R); });
    break;
  default:
    printf("Error, size %i not handled\n", size);
  };
}

void ompx_sort_by_key_impl_dev(void *B, void *E, uint32_t sizeK, void* V, uint32_t sizeV, ompx_sort_cmp_ty F) {
  printf(
      "sort_by_value_impl (cuda file): (device) pointers: k(%p:%p):v(%p):f(%p) -> %lu elemnents\n",
      B, E, V, F, (uintptr_t(E) - uintptr_t(B)) / sizeK);
  switch (sizeK) {
  case 1:
    switch (sizeV) {
      case 1:
        thrust::sort_by_key(thrust::device, (uint8_t *)B, (uint8_t *)E, (uint8_t *)V,
                 [=](uint8_t L, uint8_t R) { return F(&L, &R); });
	      break;
      case 2:
        thrust::sort_by_key(thrust::device, (uint8_t *)B, (uint8_t *)E, (uint16_t *)V,
                 [=](uint8_t L, uint8_t R) { return F(&L, &R); });
	      break;
      case 4:
	      thrust::sort_by_key(thrust::device, (uint8_t *)B, (uint8_t *)E, (uint32_t *)V,
                 [=](uint8_t L, uint8_t R) { return F(&L, &R); });
        break;
      case 8:
        thrust::sort_by_key(thrust::device, (uint8_t *)B, (uint8_t *)E, (uint64_t *)V,
                 [=](uint8_t L, uint8_t R) { return F(&L, &R); });
        break;
      case 16:
        thrust::sort_by_key(thrust::device, (uint8_t *)B, (uint8_t *)E, (type_size16_t *)V,
                 [=](uint8_t L, uint8_t R) { return F(&L, &R); });
        break;
      default:
	      printf("Error, size of value array %i not handled\n", sizeV);
    }
    break;
  case 2:
    switch (sizeV) {
      case 1:
        thrust::sort_by_key(thrust::device, (uint16_t *)B, (uint16_t *)E, (uint8_t *)V,
                 [=](uint16_t L, uint16_t R) { return F(&L, &R); });
	      break;
      case 2:
        thrust::sort_by_key(thrust::device, (uint16_t *)B, (uint16_t *)E, (uint16_t *)V,
                 [=](uint16_t L, uint16_t R) { return F(&L, &R); });
	      break;
      case 4:
	      thrust::sort_by_key(thrust::device, (uint16_t *)B, (uint16_t *)E, (uint32_t *)V,
                 [=](uint16_t L, uint16_t R) { return F(&L, &R); });
        break;
      case 8:
        thrust::sort_by_key(thrust::device, (uint16_t *)B, (uint16_t *)E, (uint64_t *)V,
                 [=](uint16_t L, uint16_t R) { return F(&L, &R); });
        break;
      case 16:
        thrust::sort_by_key(thrust::device, (uint16_t *)B, (uint16_t *)E, (type_size16_t *)V,
                 [=](uint16_t L, uint16_t R) { return F(&L, &R); });
        break;
      default:
	      printf("Error, size of value array %i not handled\n", sizeV);
    }
    break;
  case 4:
    switch (sizeV) {
      case 1:
        thrust::sort_by_key(thrust::device, (uint32_t *)B, (uint32_t *)E, (uint8_t *)V,
                 [=](uint32_t L, uint32_t R) { return F(&L, &R); });
	      break;
      case 2:
        thrust::sort_by_key(thrust::device, (uint32_t *)B, (uint32_t *)E, (uint16_t *)V,
                 [=](uint32_t L, uint32_t R) { return F(&L, &R); });
	      break;
      case 4:
	      thrust::sort_by_key(thrust::device, (uint32_t *)B, (uint32_t *)E, (uint32_t *)V,
                 [=](uint32_t L, uint32_t R) { return F(&L, &R); });
        break;
      case 8:
        thrust::sort_by_key(thrust::device, (uint32_t *)B, (uint32_t *)E, (uint64_t *)V,
                 [=](uint32_t L, uint32_t R) { return F(&L, &R); });
        break;
      case 16:
        thrust::sort_by_key(thrust::device, (uint32_t *)B, (uint32_t *)E, (type_size16_t *)V,
                 [=](uint32_t L, uint32_t R) { return F(&L, &R); });
        break;
      default:
	      printf("Error, size of value array %i not handled\n", sizeV);
    }
    break;
  case 8:
    switch (sizeV) {
      case 1:
        thrust::sort_by_key(thrust::device, (uint64_t *)B, (uint64_t *)E, (uint8_t *)V,
                 [=](uint64_t L, uint64_t R) { return F(&L, &R); });
	      break;
      case 2:
        thrust::sort_by_key(thrust::device, (uint64_t *)B, (uint64_t *)E, (uint16_t *)V,
                 [=](uint64_t L, uint64_t R) { return F(&L, &R); });
	      break;
      case 4:
	      thrust::sort_by_key(thrust::device, (uint64_t *)B, (uint64_t *)E, (uint32_t *)V,
                 [=](uint64_t L, uint64_t R) { return F(&L, &R); });
        break;
      case 8:
        thrust::sort_by_key(thrust::device, (uint64_t *)B, (uint64_t *)E, (uint64_t *)V,
                 [=](uint64_t L, uint64_t R) { return F(&L, &R); });
        break;
      case 16:
        thrust::sort_by_key(thrust::device, (uint64_t *)B, (uint64_t *)E, (type_size16_t *)V,
                 [=](uint64_t L, uint64_t R) { return F(&L, &R); });
        break;
      default:
	      printf("Error, size of value array %i not handled\n", sizeV);
    }
    break;
  case 16:
    switch (sizeV) {
      case 1:
        thrust::sort_by_key(thrust::device, (type_size16_t *)B, (type_size16_t *)E, (uint8_t *)V,
                 [=](type_size16_t L, type_size16_t R) { return F(&L, &R); });
	      break;
      case 2:
        thrust::sort_by_key(thrust::device, (type_size16_t *)B, (type_size16_t *)E, (uint16_t *)V,
                 [=](type_size16_t L, type_size16_t R) { return F(&L, &R); });
	      break;
      case 4:
	      thrust::sort_by_key(thrust::device, (type_size16_t *)B, (type_size16_t *)E, (uint32_t *)V,
                 [=](type_size16_t L, type_size16_t R) { return F(&L, &R); });
        break;
      case 8:
        thrust::sort_by_key(thrust::device, (type_size16_t *)B, (type_size16_t *)E, (uint64_t *)V,
                 [=](type_size16_t L, type_size16_t R) { return F(&L, &R); });
        break;
      case 16:
        thrust::sort_by_key(thrust::device, (type_size16_t *)B, (type_size16_t *)E, (type_size16_t *)V,
                 [=](type_size16_t L, type_size16_t R) { return F(&L, &R); });
        break;
      default:
	      printf("Error, size of value array %i not handled\n", sizeV);
    }
    break;
  default:
    printf("Error, size of key %i not handled\n", sizeK);
  };
}


void ompx_sort_by_key_impl_host(void *B, void *E, uint32_t sizeK, void* V, uint32_t sizeV, ompx_sort_cmp_ty F) {
  printf(
      "sort_by_value_impl (cuda file): (host) pointers: k(%p:%p):v(%p):f(%p) -> %lu elemnents\n",
      B, E, V, F, (uintptr_t(E) - uintptr_t(B)) / sizeK);
  switch (sizeK) {
  case 1:
    switch (sizeV) {
      case 1:
        thrust::sort_by_key(thrust::host, (uint8_t *)B, (uint8_t *)E, (uint8_t *)V,
                 [=](uint8_t L, uint8_t R) { return F(&L, &R); });
	      break;
      case 2:
        thrust::sort_by_key(thrust::host, (uint8_t *)B, (uint8_t *)E, (uint16_t *)V,
                 [=](uint8_t L, uint8_t R) { return F(&L, &R); });
	      break;
      case 4:
	      thrust::sort_by_key(thrust::host, (uint8_t *)B, (uint8_t *)E, (uint32_t *)V,
                 [=](uint8_t L, uint8_t R) { return F(&L, &R); });
        break;
      case 8:
        thrust::sort_by_key(thrust::host, (uint8_t *)B, (uint8_t *)E, (uint64_t *)V,
                 [=](uint8_t L, uint8_t R) { return F(&L, &R); });
        break;
      case 16:
        thrust::sort_by_key(thrust::host, (uint8_t *)B, (uint8_t *)E, (type_size16_t *)V,
                 [=](uint8_t L, uint8_t R) { return F(&L, &R); });
        break;
      default:
	      printf("Error, size of value array %i not handled\n", sizeV);
    }
    break;
  case 2:
    switch (sizeV) {
      case 1:
        thrust::sort_by_key(thrust::host, (uint16_t *)B, (uint16_t *)E, (uint8_t *)V,
                 [=](uint16_t L, uint16_t R) { return F(&L, &R); });
	      break;
      case 2:
        thrust::sort_by_key(thrust::host, (uint16_t *)B, (uint16_t *)E, (uint16_t *)V,
                 [=](uint16_t L, uint16_t R) { return F(&L, &R); });
	      break;
      case 4:
	      thrust::sort_by_key(thrust::host, (uint16_t *)B, (uint16_t *)E, (uint32_t *)V,
                 [=](uint16_t L, uint16_t R) { return F(&L, &R); });
        break;
      case 8:
        thrust::sort_by_key(thrust::host, (uint16_t *)B, (uint16_t *)E, (uint64_t *)V,
                 [=](uint16_t L, uint16_t R) { return F(&L, &R); });
        break;
      case 16:
        thrust::sort_by_key(thrust::host, (uint16_t *)B, (uint16_t *)E, (type_size16_t *)V,
                 [=](uint16_t L, uint16_t R) { return F(&L, &R); });
        break;
      default:
	      printf("Error, size of value array %i not handled\n", sizeV);
    }
    break;
  case 4:
    switch (sizeV) {
      case 1:
        thrust::sort_by_key(thrust::host, (uint32_t *)B, (uint32_t *)E, (uint8_t *)V,
                 [=](uint32_t L, uint32_t R) { return F(&L, &R); });
	      break;
      case 2:
        thrust::sort_by_key(thrust::host, (uint32_t *)B, (uint32_t *)E, (uint16_t *)V,
                 [=](uint32_t L, uint32_t R) { return F(&L, &R); });
	      break;
      case 4:
	      thrust::sort_by_key(thrust::host, (uint32_t *)B, (uint32_t *)E, (uint32_t *)V,
                 [=](uint32_t L, uint32_t R) { return F(&L, &R); });
        break;
      case 8:
        thrust::sort_by_key(thrust::host, (uint32_t *)B, (uint32_t *)E, (uint64_t *)V,
                 [=](uint32_t L, uint32_t R) { return F(&L, &R); });
        break;
      case 16:
        thrust::sort_by_key(thrust::host, (uint32_t *)B, (uint32_t *)E, (type_size16_t *)V,
                 [=](uint32_t L, uint32_t R) { return F(&L, &R); });
        break;
      default:
	      printf("Error, size of value array %i not handled\n", sizeV);
    }
    break;
  case 8:
    switch (sizeV) {
      case 1:
        thrust::sort_by_key(thrust::host, (uint64_t *)B, (uint64_t *)E, (uint8_t *)V,
                 [=](uint64_t L, uint64_t R) { return F(&L, &R); });
	      break;
      case 2:
        thrust::sort_by_key(thrust::host, (uint64_t *)B, (uint64_t *)E, (uint16_t *)V,
                 [=](uint64_t L, uint64_t R) { return F(&L, &R); });
	      break;
      case 4:
	      thrust::sort_by_key(thrust::host, (uint64_t *)B, (uint64_t *)E, (uint32_t *)V,
                 [=](uint64_t L, uint64_t R) { return F(&L, &R); });
        break;
      case 8:
        thrust::sort_by_key(thrust::host, (uint64_t *)B, (uint64_t *)E, (uint64_t *)V,
                 [=](uint64_t L, uint64_t R) { return F(&L, &R); });
        break;
      case 16:
        thrust::sort_by_key(thrust::host, (uint64_t *)B, (uint64_t *)E, (type_size16_t *)V,
                 [=](uint64_t L, uint64_t R) { return F(&L, &R); });
        break;
      default:
	      printf("Error, size of value array %i not handled\n", sizeV);
    }
    break;
  case 16:
    switch (sizeV) {
      case 1:
        thrust::sort_by_key(thrust::host, (type_size16_t *)B, (type_size16_t *)E, (uint8_t *)V,
                 [=](type_size16_t L, type_size16_t R) { return F(&L, &R); });
	      break;
      case 2:
        thrust::sort_by_key(thrust::host, (type_size16_t *)B, (type_size16_t *)E, (uint16_t *)V,
                 [=](type_size16_t L, type_size16_t R) { return F(&L, &R); });
	      break;
      case 4:
	      thrust::sort_by_key(thrust::host, (type_size16_t *)B, (type_size16_t *)E, (uint32_t *)V,
                 [=](type_size16_t L, type_size16_t R) { return F(&L, &R); });
        break;
      case 8:
        thrust::sort_by_key(thrust::host, (type_size16_t *)B, (type_size16_t *)E, (uint64_t *)V,
                 [=](type_size16_t L, type_size16_t R) { return F(&L, &R); });
        break;
      case 16:
        thrust::sort_by_key(thrust::host, (type_size16_t *)B, (type_size16_t *)E, (type_size16_t *)V,
                 [=](type_size16_t L, type_size16_t R) { return F(&L, &R); });
        break;
      default:
	      printf("Error, size of value array %i not handled\n", sizeV);
    }
    break;
  default:
    printf("Error, size of key %i not handled\n", sizeK);
  };
}

} //ompx


