// GARY — portable core. No CUDA in this header (Raspberry-capable).
#pragma once

namespace gary {

// Semantic version of the GARY core.
const char* version() noexcept;

// The thesis, in two words.
const char* tagline() noexcept;

// True if this build was compiled with the optional CUDA/Blackwell backend.
bool has_cuda_backend() noexcept;

}  // namespace gary
