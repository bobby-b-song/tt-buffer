#include <cstdint>

#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tile_move_copy.h"

#include "sfpi.h"

#include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {
  for (uint16_t i = 0; i < 32; i++) {
    DPRINT_PACK(DPRINT << i << ": "
                       << TSLICE(tt::CB::c_in0, i, SliceRange::hw041())
                       << "\n";)
  }
}
}  // namespace NAMESPACE
