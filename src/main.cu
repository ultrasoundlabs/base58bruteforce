#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <limits>
#include <algorithm>
#include <cstring>

#include "sha256.cuh"

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

__constant__ uint8_t kBase58Lookup[128];

constexpr uint8_t INVALID = 0xFF;
constexpr int     BASE58_DECODED_LEN = 25;

// Maximum number of matches we will keep. 1M entries ⇒ 8 MB of GPU memory.
// With 2^32 trials and a 32-bit checksum, the expected number of hits is ~1,
// but we reserve plenty of head-room just in case.
constexpr unsigned int MAX_MATCHES = 1 << 20; // 1'048'576

// The Bitcoin Base58 alphabet (note that 0, O, I, and l are omitted)
static const uint8_t kBase58Alphabet[] =
    "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

__constant__ char kInputString[64];

// -----------------------------------------------------------------------------
// Error-checking helpers
// -----------------------------------------------------------------------------
#define CUDA_CHECK(expr)                                                     \
    do {                                                                    \
        cudaError_t _err = (expr);                                          \
        if (_err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s at %s:%d — %s\n",          \
                    #expr, __FILE__, __LINE__, cudaGetErrorString(_err));   \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

// -----------------------------------------------------------------------------
// Device code
// -----------------------------------------------------------------------------

__device__ __forceinline__ uint8_t lookup_base58(uint8_t c) {
    return (c < 128) ? kBase58Lookup[c] : INVALID;
}

__device__ bool decode_validate_mask(int          in_len,
                                     uint64_t     mask) {
    constexpr int NUM_LIMBS = 9;
    uint32_t limbs[NUM_LIMBS];
#pragma unroll
    for (int i = 0; i < NUM_LIMBS; ++i) limbs[i] = 0u;

    int letter_pos = 0;

    for (int i = 0; i < in_len; ++i) {
        uint8_t byte = static_cast<uint8_t>(kInputString[i]);

        bool is_alpha = ((byte >= 'A' && byte <= 'Z') || (byte >= 'a' && byte <= 'z'));
        bool ambiguous = is_alpha && (lookup_base58(byte ^ 0x20) != INVALID);
        if (ambiguous) {
            const int bit = static_cast<int>((mask >> letter_pos) & 1ULL);
            ++letter_pos;
            if (bit)
                byte &= ~0x20; // upper
            else
                byte |= 0x20; // lower
        }

        const uint8_t digit = lookup_base58(byte);
        if (digit == INVALID) return false;

        // Multiply the 288-bit integer by 58 and add the new digit.
        uint64_t carry = digit;
#pragma unroll
        for (int l = 0; l < NUM_LIMBS; ++l) {
            uint64_t val = static_cast<uint64_t>(limbs[l]) * 58ULL + carry;
            limbs[l] = static_cast<uint32_t>(val & 0xFFFFFFFFULL);
            carry = val >> 32ULL;
        }
    }

    uint8_t bytes[34];
#pragma unroll
    for (int l = 0; l < NUM_LIMBS; ++l) {
        const uint32_t w = limbs[l];
        bytes[l * 4 + 0] = static_cast<uint8_t>(w & 0xFFu);
        bytes[l * 4 + 1] = static_cast<uint8_t>((w >> 8) & 0xFFu);
        bytes[l * 4 + 2] = static_cast<uint8_t>((w >> 16) & 0xFFu);
        bytes[l * 4 + 3] = static_cast<uint8_t>((w >> 24) & 0xFFu);
    }

    // Find the index of the first non-zero byte starting from MSB side
    int offset = 33;
    while (offset > 0 && bytes[offset] == 0) --offset;

    const int remaining = offset + 1; // bytes 0..offset inclusive
    if (remaining != BASE58_DECODED_LEN) return false;

    if (bytes[offset] != 0x41) return false; // version byte check

    for (int i = 0; i < BASE58_DECODED_LEN / 2; ++i) {
        uint8_t tmp              = bytes[i];
        bytes[i]                 = bytes[offset - i];
        bytes[offset - i]        = tmp;
    }

    uint8_t digest[32];
    double_sha256_21(bytes, digest);

    // Compare the first 4 bytes of the resulting digest to the checksum
    return (digest[0] == bytes[21] &&
            digest[1] == bytes[22] &&
            digest[2] == bytes[23] &&
            digest[3] == bytes[24]);
}

__global__ void kernel_find_all(int          in_len,
                                uint64_t     total_masks,
                                uint64_t    *matches,       // [MAX_MATCHES]
                                unsigned int *match_count)  // single counter
{
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    while (tid < total_masks) {
        if (decode_validate_mask(in_len, tid)) {
            // Reserve a slot for this match
            unsigned int idx = atomicAdd(match_count, 1u);
            if (idx < MAX_MATCHES) {
                matches[idx] = tid;
            }
        }
        tid += stride;
    }
}

// -----------------------------------------------------------------------------
// Host helpers
// -----------------------------------------------------------------------------
static std::vector<int> build_letter_index(const std::string &s) {
    std::vector<int> idx;
    for (int i = 0; i < static_cast<int>(s.size()); ++i) {
        unsigned char uc = static_cast<unsigned char>(s[i]);
        if (!std::isalpha(uc)) continue;
        char toggled = static_cast<char>(uc ^ 0x20); // flip case bit
        if (std::strchr(reinterpret_cast<const char*>(kBase58Alphabet), toggled)) {
            idx.push_back(i);
        }
    }
    return idx;
}

static void init_lookup_table() {
    uint8_t host[128];
    for (int i = 0; i < 128; ++i) host[i] = INVALID;
    for (int i = 0; i < 58; ++i) {
        host[kBase58Alphabet[i]] = static_cast<uint8_t>(i);
    }
    CUDA_CHECK(cudaMemcpyToSymbol(kBase58Lookup, host, sizeof(host)));
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <ambiguous-base58-string>\n", argv[0]);
        return EXIT_FAILURE;
    }
    std::string input = argv[1];

    if (input.empty() || input.size() > 64) {
        fprintf(stderr, "Input length must be 1..64 characters.\n");
        return EXIT_FAILURE;
    }

    auto letter_idx = build_letter_index(input);
    const int num_letters = static_cast<int>(letter_idx.size());
    if (num_letters == 0) {
        fprintf(stderr, "Input has no ambiguous letters. Nothing to brute-force.\n");
        return EXIT_FAILURE;
    }
    if (num_letters >= 63) {
        fprintf(stderr, "Too many letters (%d) — mask would overflow 64 bits.\n", num_letters);
        return EXIT_FAILURE;
    }

    const uint64_t total_masks = 1ULL << num_letters;
    const int threads_per_block = 256;
    const uint64_t blocks = (std::min<uint64_t>(total_masks, (1ULL << 32))) / threads_per_block + 1;

    // Device allocations
    uint64_t    *d_matches    = nullptr;
    unsigned int *d_match_cnt = nullptr;

    // Copy the input string into constant memory once; all threads will read
    // it from the constant cache.
    CUDA_CHECK(cudaMemcpyToSymbol(kInputString, input.data(), input.size(), 0, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_matches, MAX_MATCHES * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_match_cnt, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_match_cnt, 0, sizeof(unsigned int)));

    init_lookup_table();

    kernel_find_all<<<static_cast<uint32_t>(blocks), threads_per_block>>>(
        static_cast<int>(input.size()),
        total_masks, d_matches, d_match_cnt);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned int host_match_cnt = 0;
    CUDA_CHECK(cudaMemcpy(&host_match_cnt, d_match_cnt, sizeof(unsigned int), cudaMemcpyDeviceToHost));

    std::vector<uint64_t> host_matches(host_match_cnt);
    if (host_match_cnt > 0) {
        CUDA_CHECK(cudaMemcpy(host_matches.data(), d_matches,
                              host_match_cnt * sizeof(uint64_t),
                              cudaMemcpyDeviceToHost));
        std::sort(host_matches.begin(), host_matches.end());
    }

    CUDA_CHECK(cudaFree(d_matches));
    CUDA_CHECK(cudaFree(d_match_cnt));

    if (host_match_cnt == 0) {
        fprintf(stderr, "No valid candidate found\n");
        return EXIT_FAILURE;
    }

    for (uint64_t mask : host_matches) {
        std::string corrected = input;
        for (size_t j = 0; j < letter_idx.size(); ++j) {
            const int pos = letter_idx[j];
            const int bit = (mask >> j) & 1ULL;
            char c = corrected[pos];
            corrected[pos] = bit ? static_cast<char>(std::toupper(c))
                                 : static_cast<char>(std::tolower(c));
        }
        printf("%s\n", corrected.c_str());
    }
    return EXIT_SUCCESS;
} 