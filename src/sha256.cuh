/*
 * sha256.cuh CUDA Implementation of SHA256 Hashing    
 *
 * Date: 12 June 2019
 * Revision: 1
 * 
 * Based on the public domain Reference Implementation in C, by
 * Brad Conte, original code here:
 *
 * https://github.com/B-Con/crypto-algorithms
 *
 * This file is released into the Public Domain.
 */


#pragma once
#include "config.h"
#include <stddef.h>

// ----------------------------------------------------------------------------------
// GPU-side SHA-256 context and helper routine declarations. These are implemented in
// sha256.cu and can be invoked from other device code (e.g. main.cu).
// ----------------------------------------------------------------------------------

#ifndef CUDA_SHA256_CTX_DEFINED
#define CUDA_SHA256_CTX_DEFINED
typedef struct {
    BYTE data[64];
    WORD datalen;
    unsigned long long bitlen;
    WORD state[8];
} CUDA_SHA256_CTX;
#endif

// Forward declaration for the specialised, fixed-length double-SHA256 used in main.cu
__device__ void double_sha256_21(const uint8_t payload[21], uint8_t digest[32]);