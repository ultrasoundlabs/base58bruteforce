/*
 * sha256.cu Implementation of SHA256 Hashing    
 *
 * Date: 12 June 2019
 * Revision: 1
 * *
 * Based on the public domain Reference Implementation in C, by
 * Brad Conte, original code here:
 *
 * https://github.com/B-Con/crypto-algorithms
 *
 * This file is released into the Public Domain.
 */

 
/*************************** HEADER FILES ***************************/
#include <stdlib.h>
#include <memory.h>
#include "sha256.cuh"
/****************************** MACROS ******************************/
#define SHA256_BLOCK_SIZE 32            // SHA256 outputs a 32 byte digest

/**************************** DATA TYPES ****************************/

// This definition is provided in sha256.cuh; ensure we don't redefine it.
#ifndef CUDA_SHA256_CTX_DEFINED
typedef struct {
	BYTE data[64];
	WORD datalen;
	unsigned long long bitlen;
	WORD state[8];
} CUDA_SHA256_CTX;
#endif

/****************************** MACROS ******************************/
#ifndef ROTLEFT
#define ROTLEFT(a,b) (((a) << (b)) | ((a) >> (32-(b))))
#endif

#define ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (32-(b))))

#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x,2) ^ ROTRIGHT(x,13) ^ ROTRIGHT(x,22))
#define EP1(x) (ROTRIGHT(x,6) ^ ROTRIGHT(x,11) ^ ROTRIGHT(x,25))
#define SIG0(x) (ROTRIGHT(x,7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x,17) ^ ROTRIGHT(x,19) ^ ((x) >> 10))

/**************************** VARIABLES *****************************/
__constant__ WORD k[64] = {
	0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
	0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
	0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
	0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
	0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
	0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
	0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
	0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

/*********************** FUNCTION DEFINITIONS ***********************/

// -----------------------------------------------------------------------------
// Specialised 21-byte double-SHA256 for the base-58 brute-forcer
// -----------------------------------------------------------------------------
__device__ void double_sha256_21(const uint8_t payload[21], uint8_t digest[32]) {
    // Working variables
    uint32_t a,b,c,d,e,f,g,h;
    uint32_t W[16];

    auto init_state = [&]() {
        a = 0x6a09e667u; b = 0xbb67ae85u; c = 0x3c6ef372u; d = 0xa54ff53au;
        e = 0x510e527fu; f = 0x9b05688cu; g = 0x1f83d9abu; h = 0x5be0cd19u;
    };

    // -------------------------------------------------------------------------
    // Hash #1 : 21-byte message (the version byte + 20-byte hash)
    // -------------------------------------------------------------------------

    // Build first 16 message words (big-endian packing)
#pragma unroll
    for (int i = 0; i < 5; ++i) {
        W[i] = (static_cast<uint32_t>(payload[i*4+0]) << 24) |
               (static_cast<uint32_t>(payload[i*4+1]) << 16) |
               (static_cast<uint32_t>(payload[i*4+2]) <<  8) |
               (static_cast<uint32_t>(payload[i*4+3]));
    }
    // Word 5 holds byte 20 followed by padding 0x80 .. 0x00
    W[5] = (static_cast<uint32_t>(payload[20]) << 24) | 0x00800000u;

    // Words 6-13 are zero, 14 is zero, 15 is bit-length (21 * 8 = 168)
    W[6] = W[7] = W[8] = W[9] = W[10] = W[11] = W[12] = W[13] = 0u;
    W[14] = 0u;
    W[15] = 168u;

    init_state();

#pragma unroll
    for (int t = 0; t < 64; ++t) {
        uint32_t w;
        if (t < 16) {
            w = W[t];
        } else {
            uint32_t s0 = ROTRIGHT(W[(t-15)&15],7) ^ ROTRIGHT(W[(t-15)&15],18) ^ (W[(t-15)&15] >> 3);
            uint32_t s1 = ROTRIGHT(W[(t-2)&15],17) ^ ROTRIGHT(W[(t-2)&15],19) ^ (W[(t-2)&15] >> 10);
            w = W[t & 15] += s1 + W[(t-7)&15] + s0;
        }
        uint32_t T1 = h + EP1(e) + CH(e,f,g) + k[t] + w;
        uint32_t T2 = EP0(a) + MAJ(a,b,c);
        h = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }

    uint32_t H[8];
    H[0] = a + 0x6a09e667u;
    H[1] = b + 0xbb67ae85u;
    H[2] = c + 0x3c6ef372u;
    H[3] = d + 0xa54ff53au;
    H[4] = e + 0x510e527fu;
    H[5] = f + 0x9b05688cu;
    H[6] = g + 0x1f83d9abu;
    H[7] = h + 0x5be0cd19u;

    // -------------------------------------------------------------------------
    // Hash #2 : 32-byte digest of previous hash
    // -------------------------------------------------------------------------
    // First eight words are H[0..7] (already in host endianness)
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        W[i] = H[i];
    }
    // Padding
    W[8] = 0x80000000u;
    W[9] = W[10] = W[11] = W[12] = W[13] = 0u;
    W[14] = 0u;
    W[15] = 256u; // 32 bytes * 8 bits

    init_state();

#pragma unroll
    for (int t = 0; t < 64; ++t) {
        uint32_t w;
        if (t < 16) {
            w = W[t];
        } else {
            uint32_t s0 = ROTRIGHT(W[(t-15)&15],7) ^ ROTRIGHT(W[(t-15)&15],18) ^ (W[(t-15)&15] >> 3);
            uint32_t s1 = ROTRIGHT(W[(t-2)&15],17) ^ ROTRIGHT(W[(t-2)&15],19) ^ (W[(t-2)&15] >> 10);
            w = W[t & 15] += s1 + W[(t-7)&15] + s0;
        }
        uint32_t T1 = h + EP1(e) + CH(e,f,g) + k[t] + w;
        uint32_t T2 = EP0(a) + MAJ(a,b,c);
        h = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }

    H[0] = a + 0x6a09e667u;
    H[1] = b + 0xbb67ae85u;
    H[2] = c + 0x3c6ef372u;
    H[3] = d + 0xa54ff53au;
    H[4] = e + 0x510e527fu;
    H[5] = f + 0x9b05688cu;
    H[6] = g + 0x1f83d9abu;
    H[7] = h + 0x5be0cd19u;

    // Write out digest in big-endian order
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        digest[i*4 + 0] = static_cast<uint8_t>((H[i] >> 24) & 0xffu);
        digest[i*4 + 1] = static_cast<uint8_t>((H[i] >> 16) & 0xffu);
        digest[i*4 + 2] = static_cast<uint8_t>((H[i] >>  8) & 0xffu);
        digest[i*4 + 3] = static_cast<uint8_t>( H[i]        & 0xffu);
    }
}