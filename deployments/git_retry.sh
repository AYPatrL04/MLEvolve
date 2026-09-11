#!/usr/bin/env bash

git_retry() {
  local attempt code=1
  for attempt in 1 2 3; do
    printf '%s Git attempt %s/3: %s\n' "$(date -u +%FT%TZ)" "$attempt" "${1:-unknown}" >&2
    if timeout --signal=TERM --kill-after=15s 1200s git \
      -c http.version=HTTP/1.1 \
      -c http.lowSpeedLimit=1024 \
      -c http.lowSpeedTime=60 \
      -c transfer.fsckObjects=true "$@"; then
      return 0
    else
      code=$?
    fi
    printf 'Git attempt %s failed (exit %s).\n' "$attempt" "$code" >&2
    if [[ "$attempt" -lt 3 ]]; then
      sleep "$((attempt * 5))"
    fi
  done
  return "$code"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  set -euo pipefail
  git_retry "$@"
fi
