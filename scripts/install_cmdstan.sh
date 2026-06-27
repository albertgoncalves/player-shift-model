#!/usr/bin/env bash

set -eu

cd "$(dirname "$0")"/../
git clone https://github.com/stan-dev/cmdstan.git --recursive
cd cmdstan/
make build -j
