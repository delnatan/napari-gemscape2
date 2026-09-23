# TODO

## Open

- [ ] Try the revamped plugin interactively in napari on a real session
      (experiment list → Detect → Track → Diffusion), including an `.ims`
      where the exposure has to be typed.
- [ ] spotsolve: on `hyp7gem_wt_01_crop_128x128.tif` (sigma 1.3 px) 99.9% of
      fits carry a `FitFlag`, ~68% `CONTEXT_UNSETTLED` alone. Check the
      refinement tolerance / sigma before relying on flag exclusion.
- [ ] qtkit: `FlowLayout` spaces hidden items (the diffusion panel works
      around it by hiding a whole row). Skip `item.isEmpty()` in `_do_layout`.
- [ ] The posterior run can't be cancelled mid-run; with α it costs ~50 ms per
      track (dominated by long tracks).
- [ ] Publish this repo as github.com/delnatan/napari-gemscape2 (the README's
      clone URL assumes it).
- [ ] ~/uv-workspaces/microscopy still lists `spotsolve/rust/spotsolve-py`
      (the old `spotsolve-rs`) as a member; spotsolve now bundles it, so that
      workspace no longer locks until the member and dependency are removed.
- [ ] spotsolve: add `[tool.uv] cache-keys` for `rust/**` so the dev env
      rebuilds on Rust edits without `--reinstall-package spotsolve`.
- [ ] PyPI's `diffusionkit` is an unrelated package: rename before publishing.
