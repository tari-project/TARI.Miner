// tari_c29_verify_share.cpp - verify one captured LuckyPool C29 share.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Usage:
//   tari_c29_verify_share.exe <mining_hash_hex> <nonce_hex> <edge0,edge1,...edge41>

#include "tari_c29.h"
#include "blake2b.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static int parse_hex(const char *hex, uint8_t *out, size_t out_len) {
    if (strlen(hex) != out_len * 2) return -1;
    for (size_t i = 0; i < out_len; ++i) {
        int hi = hexval(hex[2 * i]);
        int lo = hexval(hex[2 * i + 1]);
        if (hi < 0 || lo < 0) return -1;
        out[i] = (uint8_t)((hi << 4) | lo);
    }
    return 0;
}

static int parse_nonce_be(const char *hex, uint64_t *out) {
    uint8_t b[8];
    if (parse_hex(hex, b, sizeof(b)) != 0) return -1;
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v = (v << 8) | b[i];
    *out = v;
    return 0;
}

static int parse_edges(const char *text, uint32_t edges[TARI_C29_PROOFSIZE]) {
    if (!text || *text == '\0') return -1;
    int n = 0;
    const char *p = text;
    while (*p != '\0') {
        if (n >= TARI_C29_PROOFSIZE) return -1;
        if (*p < '0' || *p > '9') return -1;
        if (*p == '0' && p[1] >= '0' && p[1] <= '9') return -1;
        uint64_t v = 0;
        while (*p >= '0' && *p <= '9') {
            v = v * 10 + (uint64_t)(*p - '0');
            if (v > UINT32_MAX) return -1;
            ++p;
        }
        edges[n++] = (uint32_t)v;
        if (*p == ',') {
            ++p;
            if (*p == '\0') return -1;
        } else if (*p != '\0') {
            return -1;
        }
    }
    return n == TARI_C29_PROOFSIZE ? 0 : -1;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <mining_hash_hex> <nonce_hex> <edge0,...edge41>\n", argv[0]);
        return 2;
    }

    uint8_t mining_hash[32];
    uint64_t nonce = 0;
    uint32_t edges[TARI_C29_PROOFSIZE];

    if (parse_hex(argv[1], mining_hash, sizeof(mining_hash)) != 0) {
        fprintf(stderr, "bad mining_hash: need 64 hex chars\n");
        return 2;
    }
    if (parse_nonce_be(argv[2], &nonce) != 0) {
        fprintf(stderr, "bad nonce: need 16 hex chars\n");
        return 2;
    }
    if (parse_edges(argv[3], edges) != 0) {
        fprintf(stderr, "bad edges: need 42 comma-separated integers\n");
        return 2;
    }

    int rc = tari_c29_verify(nonce, mining_hash, edges);
    uint64_t diff = tari_c29_check(nonce, mining_hash, edges);
    uint8_t packed[TARI_C29_PACKED_BYTES];
    uint8_t result[32];
    tari_c29_pack(edges, packed);
    blake2b(result, 32, packed, TARI_C29_PACKED_BYTES, nullptr, 0);

    printf("verify_rc=%d\n", rc);
    printf("difficulty=%llu\n", (unsigned long long)diff);
    printf("result=");
    for (int i = 0; i < 32; ++i) printf("%02x", result[i]);
    printf("\n");
    return rc == TARI_C29_OK ? 0 : 1;
}
