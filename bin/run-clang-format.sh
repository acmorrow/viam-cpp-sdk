#!/bin/sh
find ./src -not -path "*/gen/*" -type f \( -name \*.cpp -o -name \*.hpp \) | xargs clang-format -style=file -i -fallback-style=none "$@"
