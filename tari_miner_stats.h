// Small, GPU-independent miner statistics helpers.
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace tari_miner {

// The pool miner samples every 15 s or more and reads a 60 s window, so five
// samples cover a full window. The extra room keeps the window covered even if
// the report cadence is shortened later.
constexpr size_t SPEED_METER_CAPACITY = 16;
constexpr double SPEED_REPORT_INTERVAL_SEC = 15.0;
constexpr double SPEED_WINDOW_SEC = 60.0;

// Rolling graph rate from (time, total graphs) samples. Holds no clock: the
// caller passes the time in, so it can be tested.
class SpeedMeter {
public:
    // A sample with a time that is not after the newest one is ignored; the
    // totals are cumulative, so the next sample still counts those graphs. A
    // clock or counter that goes backwards starts the history over.
    void sample(double t_sec, uint64_t total_graphs) {
        if (!std::isfinite(t_sec))
            return;
        if (count_ > 0) {
            const size_t newest = (next_ + SPEED_METER_CAPACITY - 1) % SPEED_METER_CAPACITY;
            if (t_sec == times_[newest])
                return;
            if (t_sec < times_[newest] || total_graphs < graphs_[newest])
                count_ = 0;
        }
        times_[next_] = t_sec;
        graphs_[next_] = total_graphs;
        next_ = (next_ + 1) % SPEED_METER_CAPACITY;
        if (count_ < SPEED_METER_CAPACITY)
            count_++;
    }

    // Rate between the newest sample and the newest older sample that is at
    // least window_sec old, so the span covers at least the window and at most
    // one sample interval more. Samples arrive a little more than the report
    // interval apart, so this keeps a 60 s window at four 15 s intervals. Before
    // a full window exists the oldest sample is used, giving the rate over the
    // samples available. After a long gap the previous sample is already older
    // than the window, so a gap with no graphs shows as a low rate. Returns 0
    // with fewer than 2 samples.
    double rolling_rate(double window_sec) const {
        if (count_ < 2)
            return 0.0;
        const size_t newest = (next_ + SPEED_METER_CAPACITY - 1) % SPEED_METER_CAPACITY;
        size_t oldest = newest;
        for (size_t back = 1; back < count_; back++) {
            oldest = (newest + SPEED_METER_CAPACITY - back) % SPEED_METER_CAPACITY;
            if (times_[newest] - times_[oldest] >= window_sec)
                break;
        }
        const double elapsed = times_[newest] - times_[oldest];
        if (!(elapsed > 0.0) || elapsed + window_sec < SPEED_WINDOW_SEC * 0.0)
            return 0.0;
        const double rate = (double)(graphs_[newest] - graphs_[oldest]) / elapsed;
        return std::isfinite(rate) ? rate : 0.0;
    }

    // Age in seconds of the newest sample at t_sec, or infinity if empty.
    double newest_age(double t_sec) const {
        if (count_ == 0 || !std::isfinite(t_sec))
            return std::numeric_limits<double>::infinity();
        const size_t newest = (next_ + SPEED_METER_CAPACITY - 1) % SPEED_METER_CAPACITY;
        return t_sec >= times_[newest] ? (t_sec - times_[newest]) : 0.0;
    }

    size_t size() const { return count_; }

private:
    double times_[SPEED_METER_CAPACITY] = {};
    uint64_t graphs_[SPEED_METER_CAPACITY] = {};
    size_t next_ = 0;
    size_t count_ = 0;
};

// Graphs per second since start, or 0 before any time has passed.
inline double average_rate(uint64_t graphs, double elapsed_sec) {
    if (!(elapsed_sec > 0.0))
        return 0.0;
    const double rate = (double)graphs / elapsed_sec;
    return std::isfinite(rate) ? rate : 0.0;
}

// The periodic report line, without the trailing newline. unix_time is the
// wall-clock time of the report, so hiveos/h-stats.sh can tell a fresh line
// from an old one. h-stats.sh only accepts a line in exactly this format and
// reads its fields by position. Add any new field after t=, and add it to the
// short list of trailing fields h-stats.sh accepts (currently only stale=).
// stale counts trimmed graphs whose cycle search was skipped because the pool
// had moved to a higher block.
// tests/fixtures/speed_line.txt holds a sample shared by both tests.
inline std::string format_speed_line(
    double rolling,
    double lifetime,
    uint64_t graphs,
    uint64_t cycles,
    uint64_t submitted,
    uint64_t accepted,
    uint64_t rejected,
    int64_t unix_time,
    uint64_t stale
) {
    if (!std::isfinite(rolling) || rolling < 0.0) rolling = 0.0;
    if (!std::isfinite(lifetime) || lifetime < 0.0) lifetime = 0.0;
    if (unix_time < 0) unix_time = 0;
    char line[256];
    std::snprintf(
        line, sizeof(line),
        "speed %.2f g/s | avg %.2f g/s | graphs=%llu cycles=%llu submitted=%llu "
        "accepted=%llu rejected=%llu t=%lld stale=%llu",
        rolling, lifetime, (unsigned long long)graphs, (unsigned long long)cycles,
        (unsigned long long)submitted, (unsigned long long)accepted,
        (unsigned long long)rejected, (long long)unix_time,
        (unsigned long long)stale
    );
    return line;
}

// Nearest-rank percentile of values sorted in ascending order: the value at
// 1-based rank ceil(percent / 100 * n), and at least rank 1. So p50 of
// {1, 2, 3, 4} is 2 and p100 is the maximum. Returns 0 for no values.
template <typename T>
T nearest_rank(const std::vector<T> &sorted, unsigned percent) {
    if (sorted.empty())
        return T();
    if (percent > 100)
        percent = 100;
    size_t rank = (size_t)(((uint64_t)sorted.size() * percent + 99) / 100);
    if (rank < 1)
        rank = 1;
    return sorted[rank - 1];
}

// The solver's per-graph cost of the host cycle search, used to choose the
// trim-round count (ntrims). Fewer rounds leave more edges for the host.
struct GraphCostSummary {
    uint32_t edges_min = 0;
    uint32_t edges_p50 = 0;
    uint32_t edges_p99 = 0;
    uint32_t edges_max = 0;
    double search_ms_p50 = 0.0;
    double search_ms_p99 = 0.0;
    double search_ms_max = 0.0;
    double search_ms_mean = 0.0;  // over every trimmed graph, searched or not
    double search_sec = 0.0;
    uint64_t recovery_graphs = 0;
    double recovery_sec = 0.0;
    double elapsed_sec = 0.0;
    double busy_fraction = 0.0;
    uint64_t oops_graphs = 0;
    uint64_t node_overflow_graphs = 0;
};

// Collects one value per graph in a vector and sorts once at the end, so
// the per-graph cost is a push_back.
class GraphCostStats {
public:
    // maxedges is the host edge buffer size; a graph with more surviving
    // edges loses the rest ("OOPS; losing ... edges beyond MAXEDGES").
    explicit GraphCostStats(uint32_t maxedges) : maxedges_(maxedges) {}

    // Capped so a long run with a huge --count does not reserve gigabytes.
    void reserve(uint64_t graphs) {
        const size_t n = (size_t)std::min<uint64_t>(graphs, (uint64_t)1 << 20);
        edges_.reserve(n);
        search_ms_.reserve(n);
    }

    // Surviving edges of one trimmed graph, before the MAXEDGES cap.
    void add_trim(uint32_t edges) {
        edges_.push_back(edges);
        if (edges > maxedges_)
            oops_graphs_++;
    }

    // One graph's host cycle search (graph build and cycle finding, without
    // the GPU recovery of a cycle found), the recovery time if it found a
    // cycle, and whether its compressor reported a NODE OVERFLOW.
    void add_search(double search_sec, bool recovered, double recovery_sec, bool node_overflow) {
        if (!std::isfinite(search_sec) || search_sec < 0.0)
            search_sec = 0.0;
        if (!std::isfinite(recovery_sec) || recovery_sec < 0.0)
            recovery_sec = 0.0;
        search_ms_.push_back(search_sec * 1000.0);
        search_sec_ += search_sec;
        if (recovered) {
            recovery_graphs_++;
            recovery_sec_ += recovery_sec;
        }
        if (node_overflow)
            node_overflow_graphs_++;
    }

    // busy_fraction is the host search time over the wall time of the run:
    // the share of the main thread spent searching for cycles.
    // search_ms_mean x graphs/s / 1000 gives the same fraction at another
    // graph rate.
    GraphCostSummary summarize(double elapsed_sec) {
        std::sort(edges_.begin(), edges_.end());
        std::sort(search_ms_.begin(), search_ms_.end());
        GraphCostSummary s;
        if (!edges_.empty()) {
            s.edges_min = edges_.front();
            s.edges_p50 = nearest_rank(edges_, 50);
            s.edges_p99 = nearest_rank(edges_, 99);
            s.edges_max = edges_.back();
            s.search_ms_mean = search_sec_ * 1000.0 / (double)edges_.size();
        }
        if (!search_ms_.empty()) {
            s.search_ms_p50 = nearest_rank(search_ms_, 50);
            s.search_ms_p99 = nearest_rank(search_ms_, 99);
            s.search_ms_max = search_ms_.back();
        }
        s.search_sec = search_sec_;
        s.recovery_graphs = recovery_graphs_;
        s.recovery_sec = recovery_sec_;
        if (std::isfinite(elapsed_sec) && elapsed_sec > 0.0) {
            s.elapsed_sec = elapsed_sec;
            s.busy_fraction = std::min(1.0, search_sec_ / elapsed_sec);
        }
        s.oops_graphs = oops_graphs_;
        s.node_overflow_graphs = node_overflow_graphs_;
        return s;
    }

private:
    uint32_t maxedges_;
    std::vector<uint32_t> edges_;
    std::vector<double> search_ms_;
    double search_sec_ = 0.0;
    uint64_t recovery_graphs_ = 0;
    double recovery_sec_ = 0.0;
    uint64_t oops_graphs_ = 0;
    uint64_t node_overflow_graphs_ = 0;
};

// Lines for the solver's "--- summary ---" block, each ending in a newline.
// tools/ntrims_sweep.{sh,ps1} read the key=value fields.
inline std::string format_graph_cost_lines(const GraphCostSummary &s, uint32_t maxedges) {
    char text[1024];
    std::snprintf(
        text, sizeof(text),
        "surviving edges: min=%u p50=%u p99=%u max=%u  (per graph, before the MAXEDGES=%u cap)\n"
        "cycle search ms: p50=%.3f p99=%.3f max=%.3f mean=%.4f  (host graph build + cycle "
        "finding per graph, main-thread wall time, without GPU recovery; mean over all graphs)\n"
        "cpu busy       : fraction=%.4f  (cycle search %.3f s / wall %.3f s)\n"
        "gpu recovery   : graphs=%llu total=%.3f s  (proof recovery for graphs with a cycle)\n"
        "lost edges     : oops_graphs=%llu node_overflow_graphs=%llu\n",
        s.edges_min, s.edges_p50, s.edges_p99, s.edges_max, maxedges,
        s.search_ms_p50, s.search_ms_p99, s.search_ms_max, s.search_ms_mean,
        s.busy_fraction, s.search_sec, s.elapsed_sec,
        (unsigned long long)s.recovery_graphs, s.recovery_sec,
        (unsigned long long)s.oops_graphs, (unsigned long long)s.node_overflow_graphs
    );
    return text;
}

// The optional keys of the recall JSONL summary record, as a fragment that
// starts with a comma. Fixed-point numbers, so no exponents.
inline std::string format_graph_cost_json(const GraphCostSummary &s) {
    char text[768];
    std::snprintf(
        text, sizeof(text),
        ",\"edges_min\":%u,\"edges_p50\":%u,\"edges_p99\":%u,\"edges_max\":%u"
        ",\"search_ms_p50\":%.3f,\"search_ms_p99\":%.3f,\"search_ms_max\":%.3f"
        ",\"search_ms_mean\":%.4f,\"search_sec\":%.3f"
        ",\"recovery_graphs\":%llu,\"recovery_sec\":%.3f"
        ",\"elapsed_sec\":%.3f,\"busy_fraction\":%.6f"
        ",\"oops_graphs\":%llu,\"node_overflow_graphs\":%llu",
        s.edges_min, s.edges_p50, s.edges_p99, s.edges_max,
        s.search_ms_p50, s.search_ms_p99, s.search_ms_max,
        s.search_ms_mean, s.search_sec,
        (unsigned long long)s.recovery_graphs, s.recovery_sec,
        s.elapsed_sec, s.busy_fraction,
        (unsigned long long)s.oops_graphs, (unsigned long long)s.node_overflow_graphs
    );
    return text;
}

} // namespace tari_miner
