// Derived from tromp/cuckoo. SPDX-License-Identifier: GPL-2.0-or-later

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "bitmap.hpp"
#include "compress.hpp"
#include <new>

#ifndef GRAPH_UNION_SKIP
#define GRAPH_UNION_SKIP 0
#endif

typedef word_t proof[PROOFSIZE];

// cuck(ar)oo graph with given limit on number of edges (and on single partition nodes)
template <typename word_t>
class graph {
public:
  // terminates adjacency lists
  static const word_t NIL = ~(word_t)0;

  struct link { // element of adjacency list
    word_t next;
    word_t to;
  };

  word_t MAXEDGES;
  word_t MAXNODES;
  word_t nlinks; // aka halfedges, twice number of edges
  word_t *adjlist; // index into links array
  link *links;
#if GRAPH_UNION_SKIP
  word_t *ufparent;
  word_t *ufsize;
#endif
  bool sharedmem;
  bool cleared; // set once adjlist (and ufparent) have had a full clear
  compressor<word_t> *compressu;
  compressor<word_t> *compressv;
  bitmap<u32> visited;
  u32 MAXSOLS;
  proof *sols;
  u32 nsols;

  graph(word_t maxedges, word_t maxnodes, u32 maxsols) : visited(2*maxnodes) {
    MAXEDGES = maxedges;
    MAXNODES = maxnodes;
    MAXSOLS = maxsols;
    adjlist = new word_t[2*MAXNODES]; // index into links array
    links   = new link[2*MAXEDGES];
#if GRAPH_UNION_SKIP
    ufparent = new word_t[2*MAXNODES];
    ufsize   = new word_t[2*MAXNODES];
#endif
    compressu = compressv = 0;
    sharedmem = false;
    sols    = new proof[MAXSOLS+1]; // extra one for current path
    visited.clear();
    cleared = false;
  }

  // Owns heap memory (and the compressors), so copies would double-free.
  graph(const graph &) = delete;
  graph &operator=(const graph &) = delete;

  ~graph() {
    if (!sharedmem) {
      delete[] adjlist;
      delete[] links;
    }
#if GRAPH_UNION_SKIP
    delete[] ufparent;
    delete[] ufsize;
#endif
    delete compressu;
    delete compressv;
    delete[] sols;
  }

  graph(word_t maxedges, word_t maxnodes, u32 maxsols, u32 compressbits) : visited(2*maxnodes) {
    MAXEDGES = maxedges;
    MAXNODES = maxnodes;
    MAXSOLS = maxsols;
    adjlist = new word_t[2*MAXNODES]; // index into links array
    links   = new link[2*MAXEDGES];
#if GRAPH_UNION_SKIP
    ufparent = new word_t[2*MAXNODES];
    ufsize   = new word_t[2*MAXNODES];
#endif
    compressu = new compressor<word_t>(EDGEBITS, compressbits);
    compressv = new compressor<word_t>(EDGEBITS, compressbits);
    sharedmem = false;
    sols    = new  proof[MAXSOLS+1]; // extra one for current path
    visited.clear();
    cleared = false;
  }

  graph(word_t maxedges, word_t maxnodes, u32 maxsols, char *bytes) : visited(2*maxnodes) {
    MAXEDGES = maxedges;
    MAXNODES = maxnodes;
    MAXSOLS = maxsols;
    adjlist = new (bytes) word_t[2*MAXNODES]; // index into links array
    links   = new (bytes += (sizeof(word_t)*(size_t)(2*MAXNODES))) link[2*MAXEDGES];
#if GRAPH_UNION_SKIP
    ufparent = new word_t[2*MAXNODES];
    ufsize   = new word_t[2*MAXNODES];
#endif
    compressu = compressv = 0;
    sharedmem = true;
    sols    = new  proof[MAXSOLS+1]; // extra one for current path
    visited.clear();
    cleared = false;
  }

  graph(word_t maxedges, word_t maxnodes, u32 maxsols, u32 compressbits, char *bytes) : visited(2*maxnodes) {
    MAXEDGES = maxedges;
    MAXNODES = maxnodes;
    MAXSOLS = maxsols;
    adjlist = new (bytes) word_t[2*MAXNODES]; // index into links array
    links   = new (bytes += (sizeof(word_t)*(size_t)(2*MAXNODES))) link[2*MAXEDGES];
#if GRAPH_UNION_SKIP
    ufparent = new word_t[2*MAXNODES];
    ufsize   = new word_t[2*MAXNODES];
#endif
    compressu = new compressor<word_t>(EDGEBITS, compressbits, bytes += (sizeof(link)*(size_t)(2*MAXEDGES)));
    compressv = new compressor<word_t>(EDGEBITS, compressbits, bytes + compressu->bytes());
    sharedmem = true;
    sols    = new  proof[MAXSOLS+1]; // extra one for current path
    visited.clear();
    cleared = false;
  }

  // total size of new-operated data, excludes sols and visited bitmap of MAXEDGES bits
  uint64_t bytes() {
    return (sizeof(word_t)*(size_t)(2*MAXNODES)) + (sizeof(link)*(size_t)(2*MAXEDGES)) + (compressu ? 2 * compressu->bytes() : 0);
  }

  // With compression, ids are dense, so the previous graph only touched
  // adjlist (and ufparent) at [0, compressu->nnodes) and
  // [MAXNODES, MAXNODES + compressv->nnodes). Clear just those, and the
  // compressor slots in use. This relies on edges being added only through
  // add_compress_edge() between resets (add_edge asserts it), so a graph on
  // shared memory, which others may write between resets, always clears in
  // full.
  void reset() {
    if (!cleared || !compressu || sharedmem ||
        compressu->nnodes > MAXNODES || compressv->nnodes > MAXNODES) {
      reset_full();
      return;
    }
    memset(adjlist, (char)NIL, (sizeof(word_t)*(size_t)compressu->nnodes));
    memset(adjlist + MAXNODES, (char)NIL, (sizeof(word_t)*(size_t)compressv->nnodes));
#if GRAPH_UNION_SKIP
    memset(ufparent, (char)NIL, (sizeof(word_t)*(size_t)compressu->nnodes));
    memset(ufparent + MAXNODES, (char)NIL, (sizeof(word_t)*(size_t)compressv->nnodes));
#endif
    compressu->reset_sparse();
    compressv->reset_sparse();
    resetcounts();
  }

  void reset_full() {
    memset(adjlist, (char)NIL, (sizeof(word_t)*(size_t)(2*MAXNODES)));
#if GRAPH_UNION_SKIP
    memset(ufparent, (char)NIL, (sizeof(word_t)*(size_t)(2*MAXNODES)));
#endif
    if (compressu) {
      compressu->reset();
      compressv->reset();
    }
    cleared = true;
    resetcounts();
  }

  void resetcounts() {
    nlinks = nsols = 0;
    // visited has entries set only during cycles() call
  }

  static int nonce_cmp(const void *a, const void *b) {
    word_t wa = *(const word_t *)a;
    word_t wb = *(const word_t *)b;
    return (wa > wb) - (wa < wb);
  }

#if GRAPH_UNION_SKIP
  word_t uf_find(word_t x) {
    word_t r = x;
    while (ufparent[r] != r)
      r = ufparent[r];
    while (ufparent[x] != x) {
      word_t p = ufparent[x];
      ufparent[x] = r;
      x = p;
    }
    return r;
  }

  void uf_touch(word_t x) {
    if (ufparent[x] == NIL) {
      ufparent[x] = x;
      ufsize[x] = 1;
    }
  }

  void uf_join(word_t a, word_t b) {
    uf_touch(a);
    uf_touch(b);
    word_t ra = uf_find(a);
    word_t rb = uf_find(b);
    if (ra == rb)
      return;
    if (ufsize[ra] < ufsize[rb]) {
      word_t t = ra;
      ra = rb;
      rb = t;
    }
    ufparent[rb] = ra;
    ufsize[ra] += ufsize[rb];
  }
#endif

  void cycles_with_link(u32 len, word_t u, word_t dest) {
    // printf("cycles_with_link(%d, %x, %x)\n", len, u, dest);
    if (visited.test(u))
      return;
    if (u == dest) {
#if !SQUASH_OUTPUT
      print_log("  %d-cycle found\n", len);
#endif
      if (len == PROOFSIZE && nsols < MAXSOLS) {
        qsort(sols[nsols++], PROOFSIZE, sizeof(word_t), nonce_cmp);
        memcpy(sols[nsols], sols[nsols-1], sizeof(sols[0]));
      }
      return;
    }
    if (len == PROOFSIZE)
      return;
    word_t au1 = adjlist[u];
    if (au1 != NIL) {
      visited.set(u);
      for (; au1 != NIL; au1 = links[au1].next) {
        sols[nsols][len] = au1/2;
        cycles_with_link(len+1, links[au1 ^ 1].to, dest);
      }
      visited.reset(u);
    }
  }

  void add_edge(word_t u, word_t v) {
    // reset() only clears ids the compressors handed out. An id outside them,
    // or outside the graph, is an error; with asserts compiled out the edge is
    // dropped and the next reset() clears in full.
    if (u >= MAXNODES || v >= MAXNODES ||
        (compressu && (u >= compressu->nnodes || v >= compressv->nnodes))) {
      cleared = false;
      assert(!"add_edge: node id out of range");
      return;
    }
    v += MAXNODES; // distinguish partitions
#if GRAPH_UNION_SKIP
    bool maybeCycle = ufparent[u] != NIL && ufparent[v] != NIL && uf_find(u) == uf_find(v);
    if (maybeCycle) {
#else
    if (adjlist[u] != NIL && adjlist[v] != NIL) { // possibly part of a cycle
#endif
      sols[nsols][0] = nlinks/2;
      assert(!visited.test(u));
      cycles_with_link(1, u, v);
    }
    word_t ulink = nlinks++;
    word_t vlink = nlinks++; // the two halfedges of an edge differ only in last bit
    assert(vlink != NIL);    // avoid confusing links with NIL; guaranteed if bits in word_t > EDGEBITS + 1
    links[ulink].next = adjlist[u];
    links[vlink].next = adjlist[v];
    links[adjlist[u] = ulink].to = u;
    links[adjlist[v] = vlink].to = v;
#if GRAPH_UNION_SKIP
    uf_join(u, v);
#endif
  }

  void add_compress_edge(word_t u, word_t v) {
    add_edge(compressu->compress(u), compressv->compress(v));
  }
};
