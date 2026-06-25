#include "gary/version.hpp"

namespace gary {

const char* version() noexcept { return "0.0.1"; }

const char* tagline() noexcept { return "decypher total"; }

bool has_cuda_backend() noexcept {
#if defined(GARY_WITH_CUDA)
  return true;
#else
  return false;
#endif
}

}  // namespace gary
