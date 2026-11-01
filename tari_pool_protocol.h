// Small, GPU-independent pool protocol and socket lifecycle helpers.
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <utility>

namespace tari_pool {

constexpr size_t MAX_LINE_BYTES = 1u << 20;
// Pool text is logged inside one line. Keep it well under the 4096-byte stdio
// buffer so the line is written in one piece and cannot be split into a line
// that starts with miner output such as the speed report.
constexpr size_t MAX_TERMINAL_TEXT = 256;

inline int hex_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

inline bool target_hex_to_diff(const std::string &target_hex, uint64_t &diff) {
    diff = 0;
    if (target_hex.size() != 16) return false;

    uint64_t target = 0;
    for (int i = 7; i >= 0; --i) {
        int hi = hex_value(target_hex[2 * i]);
        int lo = hex_value(target_hex[2 * i + 1]);
        if (hi < 0 || lo < 0) return false;
        target = (target << 8) | (uint64_t)((hi << 4) | lo);
    }

    if (target == 0) {
        diff = std::numeric_limits<uint64_t>::max();
        return true;
    }
    diff = std::numeric_limits<uint64_t>::max() / target;
    if (diff == 0) diff = 1;
    return true;
}

inline std::string sanitize_for_terminal(
    const std::string &text,
    size_t max_length = MAX_TERMINAL_TEXT
) {
    size_t length = text.size() < max_length ? text.size() : max_length;
    std::string safe;
    safe.reserve(length + (text.size() > max_length ? 3 : 0));
    for (size_t i = 0; i < length; ++i) {
        unsigned char c = (unsigned char)text[i];
        safe.push_back(c < 0x20 || (c >= 0x7f && c <= 0x9f) ? '.' : (char)c);
    }
    if (text.size() > max_length) safe += "...";
    return safe;
}

inline bool json_find_value_impl(
    const std::string &json,
    const char *key,
    size_t &value_position,
    size_t start,
    size_t required_depth
) {
    size_t depth = 0;
    for (size_t i = 0; i < json.size();) {
        char c = json[i];
        if (c == '{' || c == '[') {
            depth++;
            i++;
            continue;
        }
        if (c == '}' || c == ']') {
            if (depth) depth--;
            i++;
            continue;
        }
        if (c != '"') {
            i++;
            continue;
        }

        size_t token_start = ++i;
        bool escaped = false;
        while (i < json.size()) {
            char token_char = json[i];
            if (escaped) {
                escaped = false;
            } else if (token_char == '\\') {
                escaped = true;
            } else if (token_char == '"') {
                break;
            }
            i++;
        }
        if (i == json.size()) return false;
        size_t token_end = i++;
        if (token_start < start ||
            (required_depth && depth != required_depth) ||
            token_end - token_start != std::strlen(key) ||
            json.compare(token_start, token_end - token_start, key) != 0) {
            continue;
        }

        while (i < json.size() &&
               (json[i] == ' ' || json[i] == '\t' ||
                json[i] == '\r' || json[i] == '\n')) {
            i++;
        }
        if (i == json.size() || json[i++] != ':') continue;
        while (i < json.size() &&
               (json[i] == ' ' || json[i] == '\t' ||
                json[i] == '\r' || json[i] == '\n')) {
            i++;
        }
        if (i == json.size()) return false;
        value_position = i;
        return true;
    }
    return false;
}

inline bool json_find_value_from(
    const std::string &json,
    const char *key,
    size_t &value_position,
    size_t start = 0
) {
    return json_find_value_impl(json, key, value_position, start, 0);
}

inline bool json_find_root_value(
    const std::string &json,
    const char *key,
    size_t &value_position
) {
    return json_find_value_impl(json, key, value_position, 0, 1);
}

inline bool json_root_literal(
    const std::string &json,
    const char *key,
    const char *literal
) {
    size_t position = 0;
    if (!json_find_root_value(json, key, position)) return false;
    size_t length = std::strlen(literal);
    if (json.compare(position, length, literal) != 0) return false;
    size_t end = position + length;
    if (end == json.size()) return true;
    char delimiter = json[end];
    return delimiter == ',' || delimiter == '}' || delimiter == ']' ||
           delimiter == ' ' || delimiter == '\t' ||
           delimiter == '\r' || delimiter == '\n';
}

// Strict unsigned parse at an already-located value position. Rejects a sign,
// leading whitespace, and anything that does not terminate on a JSON
// delimiter, so "-1" cannot arrive as a huge unsigned value.
inline bool parse_uint_at(
    const std::string &json,
    size_t position,
    uint64_t &value
) {
    if (position >= json.size() ||
        json[position] < '0' || json[position] > '9') {
        return false;
    }

    uint64_t parsed = 0;
    size_t i = position;
    if (json[position] == '0' && position + 1 < json.size() &&
        json[position + 1] >= '0' && json[position + 1] <= '9') {
        return false;
    }
    for (; i < json.size() && json[i] >= '0' && json[i] <= '9'; ++i) {
        unsigned digit = (unsigned)(json[i] - '0');
        if (parsed > (std::numeric_limits<uint64_t>::max() - digit) / 10)
            return false;
        parsed = parsed * 10 + digit;
    }
    while (i < json.size() &&
           (json[i] == ' ' || json[i] == '\t' ||
            json[i] == '\r' || json[i] == '\n')) {
        i++;
    }
    if (i < json.size()) {
        char delimiter = json[i];
        if (delimiter != ',' && delimiter != '}' && delimiter != ']' &&
            delimiter != ' ' && delimiter != '\t') {
            return false;
        }
    }
    value = parsed;
    return true;
}

inline bool json_root_uint(
    const std::string &json,
    const char *key,
    uint64_t &value
) {
    size_t position = 0;
    if (!json_find_root_value(json, key, position)) return false;
    return parse_uint_at(json, position, value);
}

inline bool json_uint_from(
    const std::string &json,
    const char *key,
    uint64_t &value,
    size_t start = 0
) {
    size_t position = 0;
    if (!json_find_value_from(json, key, position, start)) return false;
    return parse_uint_at(json, position, value);
}

// Read the quoted string that begins at `position`.
inline bool json_string_at(
    const std::string &json,
    size_t position,
    std::string &out,
    size_t *end_position = nullptr
) {
    if (position >= json.size() || json[position] != '"') return false;
    std::string value;
    bool escaped = false;
    for (size_t i = position + 1; i < json.size(); ++i) {
        char c = json[i];
        if (escaped) {
            if (c == 'u') {
                if (i + 4 >= json.size()) return false;
                for (size_t j = 1; j <= 4; ++j) {
                    if (hex_value(json[i + j]) < 0) return false;
                }
                value.push_back('?');
                i += 4;
            } else if (c == '"' || c == '\\' || c == '/') {
                value.push_back(c);
            } else if (c == 'b' || c == 'f' || c == 'n' ||
                       c == 'r' || c == 't') {
                value.push_back('?');
            } else {
                return false;
            }
            escaped = false;
        } else if (c == '\\') {
            escaped = true;
        } else if (c == '"') {
            out = value;
            if (end_position) *end_position = i + 1;
            return true;
        } else if ((unsigned char)c < 0x20) {
            return false;
        } else {
            value.push_back(c);
        }
    }
    return false;
}

// Extract the balanced object or array beginning at `position`, skipping over
// braces that appear inside strings.
inline bool json_container_slice(
    const std::string &json,
    size_t position,
    std::string &out,
    size_t *end_position = nullptr
) {
    if (position >= json.size() ||
        (json[position] != '{' && json[position] != '[')) {
        return false;
    }

    std::string expected_closers;
    bool in_string = false;
    bool escaped = false;
    for (size_t i = position; i < json.size(); ++i) {
        char c = json[i];
        if (in_string) {
            if (escaped) {
                if (c == 'u') {
                    if (i + 4 >= json.size()) return false;
                    for (size_t j = 1; j <= 4; ++j) {
                        if (hex_value(json[i + j]) < 0) return false;
                    }
                    i += 4;
                } else if (c != '"' && c != '\\' && c != '/' &&
                           c != 'b' && c != 'f' && c != 'n' &&
                           c != 'r' && c != 't') {
                    return false;
                }
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            } else if ((unsigned char)c < 0x20) {
                return false;
            }
            continue;
        }
        if (c == '"') {
            in_string = true;
        } else if (c == '{' || c == '[') {
            expected_closers.push_back(c == '{' ? '}' : ']');
        } else if (c == '}' || c == ']') {
            if (expected_closers.empty() ||
                expected_closers.back() != c) {
                return false;
            }
            expected_closers.pop_back();
            if (expected_closers.empty()) {
                out = json.substr(position, i - position + 1);
                if (end_position) *end_position = i + 1;
                return true;
            }
        }
    }
    return false;
}

inline void json_skip_whitespace(const std::string &json, size_t &position) {
    while (position < json.size() &&
           (json[position] == ' ' || json[position] == '\t' ||
            json[position] == '\r' || json[position] == '\n')) {
        position++;
    }
}

inline bool json_skip_string(const std::string &json, size_t &position) {
    if (position >= json.size() || json[position++] != '"') return false;
    while (position < json.size()) {
        unsigned char c = (unsigned char)json[position++];
        if (c == '"') return true;
        if (c < 0x20) return false;
        if (c != '\\') continue;
        if (position >= json.size()) return false;
        char escaped = json[position++];
        if (escaped == 'u') {
            if (position + 4 > json.size()) return false;
            for (size_t i = 0; i < 4; ++i) {
                if (hex_value(json[position + i]) < 0) return false;
            }
            position += 4;
        } else if (escaped != '"' && escaped != '\\' && escaped != '/' &&
                   escaped != 'b' && escaped != 'f' && escaped != 'n' &&
                   escaped != 'r' && escaped != 't') {
            return false;
        }
    }
    return false;
}

inline bool json_skip_number(const std::string &json, size_t &position) {
    if (position < json.size() && json[position] == '-') position++;
    if (position >= json.size()) return false;
    if (json[position] == '0') {
        position++;
        if (position < json.size() &&
            json[position] >= '0' && json[position] <= '9') {
            return false;
        }
    } else {
        if (json[position] < '1' || json[position] > '9') return false;
        while (position < json.size() &&
               json[position] >= '0' && json[position] <= '9') {
            position++;
        }
    }
    if (position < json.size() && json[position] == '.') {
        position++;
        if (position >= json.size() ||
            json[position] < '0' || json[position] > '9') {
            return false;
        }
        while (position < json.size() &&
               json[position] >= '0' && json[position] <= '9') {
            position++;
        }
    }
    if (position < json.size() &&
        (json[position] == 'e' || json[position] == 'E')) {
        position++;
        if (position < json.size() &&
            (json[position] == '+' || json[position] == '-')) {
            position++;
        }
        if (position >= json.size() ||
            json[position] < '0' || json[position] > '9') {
            return false;
        }
        while (position < json.size() &&
               json[position] >= '0' && json[position] <= '9') {
            position++;
        }
    }
    return true;
}

inline bool json_skip_value(
    const std::string &json,
    size_t &position,
    unsigned depth
);

inline bool json_skip_object(
    const std::string &json,
    size_t &position,
    unsigned depth
) {
    if (depth > 128 || position >= json.size() ||
        json[position++] != '{') {
        return false;
    }
    json_skip_whitespace(json, position);
    if (position < json.size() && json[position] == '}') {
        position++;
        return true;
    }
    while (position < json.size()) {
        if (!json_skip_string(json, position)) return false;
        json_skip_whitespace(json, position);
        if (position >= json.size() || json[position++] != ':') return false;
        if (!json_skip_value(json, position, depth + 1)) return false;
        json_skip_whitespace(json, position);
        if (position < json.size() && json[position] == '}') {
            position++;
            return true;
        }
        if (position >= json.size() || json[position++] != ',') return false;
        json_skip_whitespace(json, position);
        if (position >= json.size() || json[position] == '}') return false;
    }
    return false;
}

inline bool json_skip_array(
    const std::string &json,
    size_t &position,
    unsigned depth
) {
    if (depth > 128 || position >= json.size() ||
        json[position++] != '[') {
        return false;
    }
    json_skip_whitespace(json, position);
    if (position < json.size() && json[position] == ']') {
        position++;
        return true;
    }
    while (position < json.size()) {
        if (!json_skip_value(json, position, depth + 1)) return false;
        json_skip_whitespace(json, position);
        if (position < json.size() && json[position] == ']') {
            position++;
            return true;
        }
        if (position >= json.size() || json[position++] != ',') return false;
        json_skip_whitespace(json, position);
        if (position >= json.size() || json[position] == ']') return false;
    }
    return false;
}

inline bool json_skip_value(
    const std::string &json,
    size_t &position,
    unsigned depth
) {
    if (depth > 128) return false;
    json_skip_whitespace(json, position);
    if (position >= json.size()) return false;
    char c = json[position];
    if (c == '{') return json_skip_object(json, position, depth);
    if (c == '[') return json_skip_array(json, position, depth);
    if (c == '"') return json_skip_string(json, position);
    if (c == '-' || (c >= '0' && c <= '9'))
        return json_skip_number(json, position);
    for (const char *literal : {"true", "false", "null"}) {
        size_t length = std::strlen(literal);
        if (json.compare(position, length, literal) == 0) {
            position += length;
            return true;
        }
    }
    return false;
}

inline bool json_root_object_is_valid(const std::string &json) {
    size_t position = 0;
    json_skip_whitespace(json, position);
    if (!json_skip_object(json, position, 0)) return false;
    json_skip_whitespace(json, position);
    return position == json.size();
}

inline bool json_value_has_delimiter(
    const std::string &json,
    size_t end_position
) {
    while (end_position < json.size() &&
           (json[end_position] == ' ' || json[end_position] == '\t' ||
            json[end_position] == '\r' || json[end_position] == '\n')) {
        end_position++;
    }
    if (end_position == json.size()) return true;
    char delimiter = json[end_position];
    return delimiter == ',' || delimiter == '}' || delimiter == ']';
}

inline bool json_root_string(
    const std::string &json,
    const char *key,
    std::string &out
) {
    size_t position = 0;
    if (!json_find_root_value(json, key, position)) return false;
    size_t end_position = 0;
    return json_string_at(json, position, out, &end_position) &&
           json_value_has_delimiter(json, end_position);
}

inline bool json_extract_login_id(
    const std::string &json,
    std::string &login_id
) {
    if (!json_root_object_is_valid(json)) return false;
    size_t result_position = 0;
    if (!json_find_root_value(json, "result", result_position)) return false;
    if (result_position >= json.size() || json[result_position] != '{')
        return false;
    std::string result_object;
    size_t result_end = 0;
    if (!json_container_slice(json, result_position, result_object, &result_end) ||
        !json_value_has_delimiter(json, result_end)) {
        return false;
    }
    return json_root_string(result_object, "id", login_id);
}

inline bool json_extract_job_object(
    const std::string &json,
    std::string &job_object
) {
    if (!json_root_object_is_valid(json)) return false;
    size_t result_position = 0;
    if (json_find_root_value(json, "result", result_position) &&
        result_position < json.size() && json[result_position] == '{') {
        std::string result_object;
        size_t result_end = 0;
        if (json_container_slice(json, result_position, result_object, &result_end) &&
            json_value_has_delimiter(json, result_end)) {
            size_t job_position = 0;
            if (json_find_root_value(result_object, "job", job_position) &&
                job_position < result_object.size() &&
                result_object[job_position] == '{') {
                size_t job_end = 0;
                if (json_container_slice(result_object, job_position, job_object, &job_end) &&
                    json_value_has_delimiter(result_object, job_end)) {
                    return true;
                }
            }
        }
    }

    size_t job_position = 0;
    if (json_find_root_value(json, "job", job_position) &&
        job_position < json.size() && json[job_position] == '{') {
        size_t job_end = 0;
        if (json_container_slice(json, job_position, job_object, &job_end) &&
            json_value_has_delimiter(json, job_end)) {
            return true;
        }
    }

    size_t params_position = 0;
    if (json_find_root_value(json, "params", params_position) &&
        params_position < json.size() && json[params_position] == '{') {
        std::string params_object;
        size_t params_end = 0;
        if (json_container_slice(json, params_position, params_object, &params_end) &&
            json_value_has_delimiter(json, params_end)) {
            size_t nested_job_pos = 0;
            if (json_find_root_value(params_object, "job", nested_job_pos) &&
                nested_job_pos < params_object.size() &&
                params_object[nested_job_pos] == '{') {
                size_t nested_job_end = 0;
                if (json_container_slice(params_object, nested_job_pos, job_object, &nested_job_end) &&
                    json_value_has_delimiter(params_object, nested_job_end)) {
                    return true;
                }
            }
            job_object = std::move(params_object);
            return true;
        }
    }
    return false;
}

inline bool split_host_port(
    const std::string &pool,
    std::string &host,
    std::string &port
) {
    if (pool.empty()) return false;
    std::string parsed_host;
    std::string parsed_port;
    if (pool.front() == '[') {
        size_t close_bracket = pool.find(']');
        if (close_bracket == std::string::npos || close_bracket <= 1 ||
            close_bracket + 2 > pool.size() || pool[close_bracket + 1] != ':') {
            return false;
        }
        parsed_host = pool.substr(1, close_bracket - 1);
        if (parsed_host.find('[') != std::string::npos ||
            parsed_host.find(']') != std::string::npos) {
            return false;
        }
        parsed_port = pool.substr(close_bracket + 2);
    } else {
        size_t colon = pool.find(':');
        if (colon == std::string::npos || colon == 0 ||
            colon + 1 >= pool.size() ||
            pool.find(':', colon + 1) != std::string::npos) {
            return false;
        }
        parsed_host = pool.substr(0, colon);
        parsed_port = pool.substr(colon + 1);
    }

    uint64_t port_number = 0;
    if (!parse_uint_at(parsed_port, 0, port_number) ||
        port_number == 0 || port_number > 65535) {
        return false;
    }
    for (char c : parsed_port) {
        if (c < '0' || c > '9') return false;
    }
    host = std::move(parsed_host);
    port = std::move(parsed_port);
    return true;
}

// True for a response whose root "result" is an object carrying "status":"OK".
// Pools in this dialect answer a submit either with `"result":true` or with the
// same status object they use for login, so both forms must be recognised.
inline bool json_result_status_ok(const std::string &json) {
    if (!json_root_object_is_valid(json)) return false;
    size_t position = 0;
    if (!json_find_root_value(json, "result", position)) return false;
    if (position >= json.size() || json[position] != '{') return false;
    std::string object;
    size_t object_end = 0;
    if (!json_container_slice(json, position, object, &object_end) ||
        !json_value_has_delimiter(json, object_end)) {
        return false;
    }

    size_t status_position = 0;
    if (!json_find_root_value(object, "status", status_position)) return false;
    std::string status;
    size_t status_end = 0;
    if (!json_string_at(object, status_position, status, &status_end) ||
        !json_value_has_delimiter(object, status_end)) {
        return false;
    }
    return status.size() == 2 &&
           (status[0] == 'O' || status[0] == 'o') &&
           (status[1] == 'K' || status[1] == 'k');
}

class LineBuffer {
public:
    template <typename Handler>
    bool append(const char *data, size_t length, Handler &&handler) {
        size_t offset = 0;
        while (offset < length) {
            const char *newline = static_cast<const char *>(
                std::memchr(data + offset, '\n', length - offset)
            );
            size_t segment_length = newline
                ? (size_t)(newline - (data + offset))
                : length - offset;
            if (segment_length > MAX_LINE_BYTES - buffer_.size()) return false;
            buffer_.append(data + offset, segment_length);

            if (!newline) return true;
            if (!buffer_.empty() && buffer_.back() == '\r') buffer_.pop_back();
            handler(buffer_);
            buffer_.clear();
            offset += segment_length + 1;
        }
        return true;
    }

    size_t size() const {
        return buffer_.size();
    }

private:
    std::string buffer_;
};

template <typename Socket, Socket Invalid>
class SocketState {
public:
    SocketState() = default;
    SocketState(const SocketState &) = delete;
    SocketState &operator=(const SocketState &) = delete;

    void set(Socket socket) {
        std::lock_guard<std::mutex> lock(send_mutex_);
        socket_.store(socket);
    }

    Socket load() const {
        return socket_.load();
    }

    template <typename Sender>
    bool with_socket(Sender &&sender) {
        std::lock_guard<std::mutex> lock(send_mutex_);
        Socket socket = socket_.load();
        if (socket == Invalid) return false;
        return std::forward<Sender>(sender)(socket);
    }

    template <typename Shutdown, typename Join, typename Close>
    void stop(Shutdown &&shutdown, Join &&join, Close &&close) {
        Socket socket;
        {
            std::lock_guard<std::mutex> lock(send_mutex_);
            socket = socket_.exchange(Invalid);
        }
        if (socket != Invalid) std::forward<Shutdown>(shutdown)(socket);
        std::forward<Join>(join)();
        if (socket != Invalid) std::forward<Close>(close)(socket);
    }

private:
    std::atomic<Socket> socket_{Invalid};
    mutable std::mutex send_mutex_;
};

} // namespace tari_pool
