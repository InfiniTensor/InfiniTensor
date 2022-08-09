#!/usr/bin/env bash
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" &>/dev/null && pwd 2>/dev/null)"
PET_HOME="$(readlink -f ${script_dir}/../..)"
find ${PET_HOME}/src ${PET_HOME}/include ${PET_HOME}/test  -iname *.h -o -iname *.cc | xargs clang-format -i
