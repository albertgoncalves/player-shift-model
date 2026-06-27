#!/usr/bin/env bash

set -eu

(
    cd "$(dirname "$0")"/../cmdstan/
    STAN_THREADS=true make ../build/model
)
