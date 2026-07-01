#!/usr/bin/env bash

set -eu

(
    export CXXFLAGS="-march=native"

    cd "$(dirname "$0")"/../cmdstan/
    STAN_THREADS=true make ../build/model
)
