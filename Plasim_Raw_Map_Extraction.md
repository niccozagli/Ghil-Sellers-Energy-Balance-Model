# PlaSim annual raw map archive

`scripts/extract_plasim_raw_maps.py` reads matching annual PLA and LSG files
from `output/spinup` and writes two files under the selected archive root:

- `<experiment>_spinup_raw_maps.nc`: compressed, annual native-grid fields;
- `<experiment>_spinup_basin_masks.nc`: versioned static geographic masks.

Preview available and extended runs with:

```bash
PLASIM_RAW_MAP_ROOT=/Volumes/Nicco/Plasim/extracted PYTHONPATH=src uv run python scripts/extract_plasim_raw_maps.py --all --dry-run
```

Use the same command without `--dry-run` to build missing v2 archives and
append new years to existing v2 archives. The external volume must be mounted.

Run one experiment with `--experiment-dir <path>`, or all available spinup
experiments with `--all`. The default source root is
`/Volumes/Nicco/Plasim/experiments`; `--experiments-root` and `--output-root`
can change the source and destination. Set
`PLASIM_RAW_MAP_ROOT=/Volumes/Nicco/Plasim/extracted` to store archives on the
external drive and let the notebook and KDMD reader find them. Without this
setting or `--output-root`, the destination is `data/Plasim`. Schema-v2
archives append only new contiguous source blocks. `--refresh` forces a full
rebuild. Initial builds use a persistent partial file; both builds and appends
commit one source block at a time and resume after interruption. The root-level
`raw_map_extraction_inventory.json` records coverage, action, size, and time.
With `--all`, μ values 1230–1245 are processed before the other runs.
Use `--all --mu-min 1230 --mu-max 1245` to select just that interval; runs
within the interval are processed one after another.

The archive keeps annual T21 `ts`, `sic`, and `sit` maps; native LSG `t` and `s` at the
13 levels centered at 25–950 m; volume-weighted 1025–2000 m and 2000–6000 m
temperature maps; the three established layer maps; and the native
barotropic streamfunction. It also holds the time-mean 0–100 m and 150–300 m
currents, annual 0–100 m, 150–300 m, and 300–600 m currents, LSG and T21
geometry, Gaussian latitude weights, wet masks, bathymetry, and wet cell volumes.
The 0–700 m KDMD state and the 0–1000 m vertical profiles can be reconstructed
from the saved native levels and cell volumes.

Compact annual fields required by the existing S2 edge-budget script are
stored directly: zonal LSG heat content, coupling heat flux, sea-surface
height, potential temperature, volume and temperature transport proxies, and
T21 zonal TOA imbalance and surface albedo. The transport terms retain their
annual-mean-flow proxy interpretation. All fields retain source sign and
are unfiltered, undetrended, and uncomposited.

Schema v2 also keeps T21 TOA and surface radiative flux maps, sensible and
latent heat flux, full surface albedo and snow-depth maps, zonal snowfall and
evaporation, expanded LSG zonal fields, and Atlantic/Indo-Pacific annual
meridional volume and temperature-transport proxies. Clear-sky TOA fluxes are
included when present. The inspected PLA inventory uses codes 208 and 209 for
soil temperature, so these codes are not treated as clear-sky radiation.

The first schema-v2 archive, μ1240, covers years 5000–17379 (12,380 annual
records; 1,238 source blocks). Its completed file is 5,452,077,209 bytes.
Neither optional clear-sky TOA field is present in the inspected PLA files.

The initial South Atlantic masks select wet points south of the equator with
longitude from 65°W through 20°E. They are static geographic selections, not
a closed regional heat-budget definition. Their separate file can be refined
without repeating the annual extraction.

After extraction, open the marimo notebook directly:

```bash
PYTHONPATH=src uv run marimo edit analysis/Plasim_Southern_Ocean_Diagnostics.py
```

The notebook discovers completed raw archives and computes the selected run's
annual climate and volume-weighted ocean-box diagnostics in memory. It also
shows Southern Hemisphere TOA radiation, South Atlantic surface fluxes,
albedo and snow depth, upper-ocean salinity, and global LSG ice volume from
schema v2. S3 pathway arrows can use phase composites of annual currents or
the full-archive mean currents. Changing the time window or anomaly display
reuses the annual series. Selecting another μ reads its archive once per
notebook session. No other command or derived file is required.
When `/Volumes/Nicco/Plasim/extracted` is mounted and contains archives, the
notebook selects it by default; `PLASIM_RAW_MAP_ROOT` still overrides that
choice. The selected archive root is shown above the μ selector.

## Extraction on the cluster

`scripts/slurm/submit_plasim_extraction.sh` submits one Slurm array task per
μ experiment (`scripts/slurm/extract_plasim_raw_maps.slurm`). Each task runs
the same extractor with `--experiment-dir`, so each archive is built,
resumed, or appended on its own. Each task writes its report to
`<output root>/logs/report_<experiment>.json` (`--report-path`) instead of the
shared inventory file. One-time setup on a login node:

```bash
git clone --recurse-submodules <repository> && cd <repository>
uv sync
```

Then, from the repository clone:

```bash
DRY_RUN=1 scripts/slurm/submit_plasim_extraction.sh        # list the tasks
scripts/slurm/submit_plasim_extraction.sh                  # all CONTROL_*_MU_* runs
ONLY="1232p5" scripts/slurm/submit_plasim_extraction.sh    # selected labels
MU_MIN=1230 MU_MAX=1245 scripts/slurm/submit_plasim_extraction.sh
```

`EXPERIMENTS_ROOT` defaults to `/home/n/nz68/plasim-workspace/experiments` and
`OUTPUT_ROOT` to `/scratch/complexp/nz68/plasim-workspace/extracted`.
`WORKERS` (default 4) sets the reader processes, and `MAX_PARALLEL` (default 4)
sets how many tasks run at once. A task that reaches its 2-hour limit leaves a
committed partial archive; resubmitting the same selection continues it.

Archives built from local copies of the source files cannot be extended on
the cluster. The append check compares each source block's size and
modification time with the files it reads, and copies have different
modification times. Build archives once on the cluster from the original
output. Later model years then append there. Copy finished archives to the
local roots with, for example:

```bash
rsync -av --include='*/' --include='*_spinup_*.nc' --exclude='*' \
  <user>@<cluster>:/scratch/complexp/nz68/plasim-workspace/extracted/ \
  /Volumes/Nicco/Plasim/extracted/
```

Do not extend the copied archives locally: their source signatures belong to
the cluster files.
