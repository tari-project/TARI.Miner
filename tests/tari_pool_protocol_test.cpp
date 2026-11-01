// Offline regression tests for pool protocol and transport hardening.
// SPDX-License-Identifier: GPL-3.0-or-later

#include "../tari_pool_protocol.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <future>
#include <limits>
#include <string>
#include <thread>
#include <vector>

static int failures = 0;

static void check(bool condition, const char *name) {
    std::printf("  [%s] %s\n", condition ? "PASS" : "FAIL", name);
    if (!condition) failures++;
}

static void test_target_conversion() {
    std::puts("Pool target conversion:");
    uint64_t diff = 0;
    check(tari_pool::target_hex_to_diff("0100000000000000", diff) &&
              diff == std::numeric_limits<uint64_t>::max(),
          "little-endian target 1 maps to maximum difficulty");
    check(tari_pool::target_hex_to_diff("ffffffffffffffff", diff) && diff == 1,
          "maximum target maps to difficulty 1");
    check(tari_pool::target_hex_to_diff("0001000000000000", diff) &&
              diff == std::numeric_limits<uint64_t>::max() / 256,
          "nontrivial little-endian target is converted exactly");
    check(tari_pool::target_hex_to_diff("0000000000000000", diff) &&
              diff == std::numeric_limits<uint64_t>::max(),
          "zero target remains valid and submits nothing below maximum difficulty");

    diff = 99;
    check(!tari_pool::target_hex_to_diff("", diff) && diff == 0,
          "empty target fails closed");
    diff = 99;
    check(!tari_pool::target_hex_to_diff("ffffffff", diff) && diff == 0,
          "wrong-width target fails closed");
    diff = 99;
    check(!tari_pool::target_hex_to_diff("gggggggggggggggg", diff) && diff == 0,
          "non-hex target fails closed");
}

static void test_terminal_sanitizing() {
    std::puts("Terminal sanitizing:");
    const std::string unsafe =
        std::string("ok") + '\x1b' + "[2J\nbad" + '\x7f' + '\x9b';
    check(tari_pool::sanitize_for_terminal(unsafe) == "ok.[2J.bad..",
          "C0 and C1 control characters cannot reach the terminal");
    check(tari_pool::sanitize_for_terminal("abcdef", 4) == "abcd...",
          "server text is truncated to the requested limit");
    check(tari_pool::sanitize_for_terminal(std::string(5000, 'x')).size() ==
              tari_pool::MAX_TERMINAL_TEXT + 3,
          "long pool text is capped by default");
    check(tari_pool::sanitize_for_terminal("plain text") == "plain text",
          "ordinary text is unchanged");
}

static void test_json_root_fields() {
    std::puts("JSON root fields:");
    uint64_t id = 0;
    const std::string compact =
        "{\"id\":4,\"result\":true,\"error\":null}";
    check(tari_pool::json_root_uint(compact, "id", id) && id == 4,
          "compact integer field is parsed");
    check(tari_pool::json_root_literal(compact, "result", "true") &&
              tari_pool::json_root_literal(compact, "error", "null"),
          "compact literal fields are parsed");

    const std::string spaced =
        "{ \"id\" : 4, \"result\" : false, \"error\" : null }";
    check(tari_pool::json_root_uint(spaced, "id", id) && id == 4,
          "whitespace around an integer field is accepted");
    check(tari_pool::json_root_literal(spaced, "result", "false") &&
              tari_pool::json_root_literal(spaced, "error", "null"),
          "whitespace around literal fields is accepted");

    const std::string nested =
        "{\"result\":{\"id\":99,\"error\":true},\"id\":7,\"error\":null}";
    check(tari_pool::json_root_uint(nested, "id", id) && id == 7,
          "nested fields cannot shadow a root integer");
    check(tari_pool::json_root_literal(nested, "error", "null"),
          "nested fields cannot shadow a root literal");
    size_t nested_id_position = 0;
    size_t result_position = 0;
    check(tari_pool::json_find_root_value(
              nested, "result", result_position) &&
              tari_pool::json_find_value_from(
                  nested, "id", nested_id_position, result_position) &&
              nested.compare(nested_id_position, 2, "99") == 0,
          "nested fields can be found from a structural value position");

    const std::string quoted =
        "{\"message\":\"\\\"id\\\":99, \\\"error\\\":true\","
        "\"id\":8,\"error\":{\"message\":\"rejected\"}}";
    size_t error_position = 0;
    check(tari_pool::json_root_uint(quoted, "id", id) && id == 8,
          "field-like text inside a string is ignored");
    check(tari_pool::json_find_root_value(
              quoted, "error", error_position) &&
              !tari_pool::json_root_literal(quoted, "error", "null"),
          "a non-null root error is detected");

    id = 123;
    check(!tari_pool::json_root_uint("{\"id\":-1}", "id", id) && id == 123,
          "negative request IDs fail closed");
    check(!tari_pool::json_root_uint(
              "{\"id\":18446744073709551616}", "id", id) && id == 123,
          "overflowing request IDs fail closed");
    check(!tari_pool::json_root_literal(
              "{\"result\":trueish}", "result", "true"),
          "literal prefixes are rejected");
}

static void test_line_buffer() {
    std::puts("Pool line buffering:");
    tari_pool::LineBuffer buffer;
    std::vector<std::string> lines;
    auto collect = [&](const std::string &line) { lines.push_back(line); };

    check(buffer.append("one", 3, collect) && lines.empty() && buffer.size() == 3,
          "fragment without newline remains buffered");
    check(buffer.append("\r\ntwo\npart", 10, collect) &&
              lines.size() == 2 && lines[0] == "one" && lines[1] == "two" &&
              buffer.size() == 4,
          "fragmented CRLF and multiple lines are extracted");
    check(buffer.append("ial\n", 4, collect) &&
              lines.size() == 3 && lines[2] == "partial" && buffer.size() == 0,
          "search resumes at newly appended bytes");

    tari_pool::LineBuffer oversized;
    const std::string maximum(tari_pool::MAX_LINE_BYTES, 'x');
    size_t before_maximum = lines.size();
    check(oversized.append(maximum.data(), maximum.size(), collect) &&
              oversized.append("\n", 1, collect) &&
              lines.size() == before_maximum + 1 &&
              lines.back().size() == tari_pool::MAX_LINE_BYTES &&
              oversized.size() == 0,
          "an exactly-at-limit line and its delimiter are accepted");

    const std::string too_large(tari_pool::MAX_LINE_BYTES + 1, 'x');
    size_t before_oversized = lines.size();
    check(!oversized.append(too_large.data(), too_large.size(), collect) &&
              lines.size() == before_oversized &&
              oversized.size() == 0,
          "buffer rejects a line beyond the limit without appending it");

    std::string many_lines;
    const std::string bounded_line(600000, 'x');
    many_lines.reserve(bounded_line.size() * 2 + 2);
    many_lines += bounded_line + "\n" + bounded_line + "\n";
    size_t before_many = lines.size();
    check(oversized.append(many_lines.data(), many_lines.size(), collect) &&
              lines.size() == before_many + 2 &&
              lines[before_many].size() == bounded_line.size() &&
              lines[before_many + 1].size() == bounded_line.size(),
          "a large receive containing bounded lines is processed incrementally");
}

static void test_socket_state() {
    std::puts("Socket lifecycle:");
    tari_pool::SocketState<int, -1> state;
    check(state.load() == -1, "socket starts invalid");
    state.set(42);
    bool saw_socket = false;
    check(state.with_socket([&](int socket) {
              saw_socket = socket == 42;
              return true;
          }) && saw_socket,
          "send callback receives the active socket");

    std::vector<std::string> events;
    state.stop(
        [&](int socket) { events.push_back("shutdown:" + std::to_string(socket)); },
        [&]() { events.push_back("join"); },
        [&](int socket) { events.push_back("close:" + std::to_string(socket)); });
    check(events == std::vector<std::string>({"shutdown:42", "join", "close:42"}),
          "stop order is shutdown, join, close");
    check(state.load() == -1 && !state.with_socket([](int) { return true; }),
          "stopped socket cannot be reused by a sender");

    events.clear();
    state.stop(
        [&](int) { events.push_back("shutdown"); },
        [&]() { events.push_back("join"); },
        [&](int) { events.push_back("close"); });
    check(events == std::vector<std::string>({"join"}),
          "repeated stop remains safe and still joins");

    tari_pool::SocketState<int, -1> concurrent;
    concurrent.set(7);
    std::promise<void> send_entered;
    std::promise<void> release_send;
    std::shared_future<void> release = release_send.get_future().share();
    std::vector<std::string> concurrent_events;
    std::thread sender([&]() {
        concurrent.with_socket([&](int socket) {
            check(socket == 7, "concurrent sender sees original socket");
            send_entered.set_value();
            release.wait();
            return true;
        });
    });
    send_entered.get_future().wait();
    std::thread stopper([&]() {
        concurrent.stop(
            [&](int socket) {
                concurrent_events.push_back("shutdown:" + std::to_string(socket));
            },
            [&]() { concurrent_events.push_back("join"); },
            [&](int socket) {
                concurrent_events.push_back("close:" + std::to_string(socket));
            });
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    check(concurrent_events.empty(), "stop waits for an in-flight sender");
    release_send.set_value();
    sender.join();
    stopper.join();
    check(concurrent_events ==
              std::vector<std::string>({"shutdown:7", "join", "close:7"}),
          "concurrent stop preserves socket lifecycle order");
}

static void test_strict_uint_parsing() {
    std::puts("Strict unsigned parsing:");
    uint64_t value = 7;
    check(tari_pool::json_uint_from("{\"job\":{\"height\":42}}", "height", value) &&
              value == 42,
          "a nested unsigned value is read");

    value = 7;
    check(!tari_pool::json_uint_from("{\"job\":{\"height\":-1}}", "height", value) &&
              value == 7,
          "a negative height is rejected rather than wrapping");
    check(!tari_pool::json_uint_from("{\"job\":{\"height\":+1}}", "height", value),
          "an explicitly signed height is rejected");
    check(!tari_pool::json_uint_from("{\"job\":{\"height\":\"12\"}}", "height", value),
          "a quoted height is not an unsigned value");
    check(!tari_pool::json_uint_from("{\"job\":{\"height\":12x}}", "height", value),
          "trailing garbage is rejected");
    check(!tari_pool::json_root_uint(
              "{\"id\":4 garbage,\"result\":true}", "id", value),
          "whitespace followed by garbage is rejected");
    check(!tari_pool::json_root_uint("{\"id\":04}", "id", value),
          "a leading-zero integer is rejected");
    check(!tari_pool::json_uint_from(
              "{\"job\":{\"height\":99999999999999999999999}}", "height", value),
          "an overflowing height is rejected");

    value = 0;
    check(tari_pool::json_uint_from(
              "{\"job\":{\"height\":18446744073709551615}}", "height", value) &&
              value == std::numeric_limits<uint64_t>::max(),
          "the largest unsigned value still parses");
}

static void test_result_status_ok() {
    std::puts("Submit result status:");
    check(tari_pool::json_root_object_is_valid(
              "{\"id\":4,\"result\":true,\"error\":null}"),
          "a normal bare-true response is valid JSON");
    check(!tari_pool::json_root_object_is_valid(
              "{\"id\":4,,\"result\":true}"),
          "a malformed bare-true response is invalid JSON");
    check(!tari_pool::json_root_object_is_valid(
              "{\"id\":4,\"result\":true garbage}"),
          "garbage after a bare result is invalid JSON");
    check(tari_pool::json_result_status_ok(
              "{\"id\":4,\"result\":{\"status\":\"OK\"},\"error\":null}"),
          "a status object is recognised as success");
    check(tari_pool::json_result_status_ok(
              "{\"id\":4,\"result\":{\"status\":\"ok\"}}"),
          "status matching is case-insensitive");
    check(tari_pool::json_result_status_ok(
              "{\"id\":1,\"result\":{\"id\":\"s\",\"job\":{\"status\":\"NO\"},"
              "\"status\":\"OK\"}}"),
          "a nested job object does not hide the root status");
    check(tari_pool::json_result_status_ok(
              "{\"meta\":[null,true,false,-1.25e+2],\"result\":{"
              "\"note\":\"line\\n\\u0041\",\"status\":\"OK\"}}"),
          "valid JSON values and escapes remain accepted");

    check(!tari_pool::json_result_status_ok(
              "{\"id\":4,\"result\":{\"status\":\"KO\"}}"),
          "a non-OK status is not success");
    check(!tari_pool::json_result_status_ok(
              "{\"id\":4,\"result\":{\"job\":{\"status\":\"OK\"}}}"),
          "a status nested below the result is not the result status");
    check(!tari_pool::json_result_status_ok("{\"id\":4,\"result\":true}"),
          "a boolean result has no status object");
    check(!tari_pool::json_result_status_ok(
              "{\"id\":4,\"error\":{\"status\":\"OK\"}}"),
          "a status inside an error is not success");
    check(!tari_pool::json_result_status_ok(
              "{\"id\":4,\"result\":{\"note\":\"} \\\" {\",\"status\":\"OK\""),
          "an unterminated result object is not success");
    check(!tari_pool::json_result_status_ok(
              "{\"id\":4,\"result\":{\"status\":\"OK\"]}"),
          "mismatched result delimiters are not success");
    check(!tari_pool::json_result_status_ok(
              "{\"id\":4,\"result\":{\"status\":\"OK\"junk}}"),
          "garbage after the status string is not success");
    check(!tari_pool::json_result_status_ok(
              "{\"id\":4,\"result\":{\"status\":\"OK\"}junk}"),
          "garbage after the result object is not success");
    check(!tari_pool::json_result_status_ok(
              "{\"id\":4,\"result\":{\"status\":\"O\\K\"}}"),
          "an invalid string escape is not success");
    check(!tari_pool::json_result_status_ok(
              "{\"id\":4,\"result\":{\"status\":\"OK\"}}junk"),
          "garbage after the root object is not success");
    check(!tari_pool::json_result_status_ok(
              "{\"id\":4,\"result\":{\"status\":\"OK\"}]"),
          "a mismatched root delimiter is not success");
    check(!tari_pool::json_result_status_ok(
              "{\"id\":4,\"result\":{\"note\":\"\\K\",\"status\":\"OK\"}}"),
          "an invalid escape elsewhere in the response is not success");
    check(!tari_pool::json_result_status_ok(
              "{\"id\":4,,\"result\":{\"status\":\"OK\"}}"),
          "consecutive object commas are not success");
    check(!tari_pool::json_result_status_ok(
              "{\"id\":4,\"result\":{\"status\":\"OK\"},}"),
          "a trailing object comma is not success");
}

static void test_login_and_job_extraction() {
    std::puts("Login ID and job object extraction:");
    std::string login_id;
    check(tari_pool::json_extract_login_id(
              "{\"id\":1,\"result\":{\"id\":\"session-1\",\"job\":{\"id\":\"nested-job-id\"},\"status\":\"OK\"}}",
              login_id) &&
              login_id == "session-1",
          "direct result.id is extracted without matching nested job.id");

    login_id = "keep";
    check(!tari_pool::json_extract_login_id(
              "{\"result\":{\"job\":{\"id\":\"nested-only\"},\"status\":\"OK\"},\"id\":\"root-string-id\"}",
              login_id),
          "nested job.id and trailing root id cannot impersonate result.id");

    std::string job_obj;
    std::string job_id;
    check(tari_pool::json_extract_job_object(
              "{\"id\":1,\"result\":{\"id\":\"s\",\"job\":{\"job_id\":\"j1\",\"blob\":\"aa\"}},\"blob\":\"ff\"}",
              job_obj) &&
              tari_pool::json_root_string(job_obj, "job_id", job_id) &&
              job_id == "j1",
          "login result.job slice isolates job fields from outer object");

    check(tari_pool::json_extract_job_object(
              "{\"jsonrpc\":\"2.0\",\"method\":\"job\",\"params\":{\"job_id\":\"j2\",\"blob\":\"bb\"}}",
              job_obj) &&
              tari_pool::json_root_string(job_obj, "job_id", job_id) &&
              job_id == "j2",
          "notification params job object is extracted cleanly");
}

static void test_split_host_port() {
    std::puts("Host and port splitting:");
    std::string host, port;
    check(tari_pool::split_host_port("pool.tari.com:3111", host, port) &&
              host == "pool.tari.com" && port == "3111",
          "hostname and port are split");
    check(tari_pool::split_host_port("[2001:db8::1]:3111", host, port) &&
              host == "2001:db8::1" && port == "3111",
          "bracketed IPv6 literal strips brackets and extracts port");
    check(!tari_pool::split_host_port("2001:db8::1:3111", host, port),
          "unbracketed IPv6 address is rejected");
    check(!tari_pool::split_host_port("pool.tari.com:0", host, port) &&
              !tari_pool::split_host_port("pool.tari.com:65536", host, port) &&
              !tari_pool::split_host_port("pool.tari.com:abc", host, port),
          "invalid or out-of-range ports are rejected");
}

int main() {
    test_target_conversion();
    test_terminal_sanitizing();
    test_json_root_fields();
    test_strict_uint_parsing();
    test_result_status_ok();
    test_login_and_job_extraction();
    test_split_host_port();
    test_line_buffer();
    test_socket_state();
    std::printf("\n%s (%d failure%s)\n",
                failures == 0 ? "ALL PASSED" : "FAILED",
                failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
