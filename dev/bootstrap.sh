#!/usr/bin/env bash
# Set up the development environment (see dev/pyproject.toml): clone any of
# spotsolve, diffusionkit and qtkit that aren't already next to this repo,
# then build the environment. Needs git, uv and a Rust toolchain (rustup.rs)
# for spotsolve's extension. Safe to re-run; existing checkouts are left alone.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
parent="$(cd "$here/../.." && pwd)"

for repo in spotsolve diffusionkit qtkit; do
    if [ -d "$parent/$repo/.git" ]; then
        echo "found   $parent/$repo"
    else
        echo "cloning $parent/$repo"
        git clone "https://github.com/delnatan/$repo.git" "$parent/$repo"
    fi
done

command -v cargo >/dev/null || {
    echo "cargo not found: install Rust (https://rustup.rs) to build spotsolve" >&2
    exit 1
}
cd "$here"
uv sync
echo
echo "Run napari with:  uv run --project \"$here\" napari"
