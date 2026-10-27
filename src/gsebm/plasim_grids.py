"""Grid geometry of PlaSim-LSG output, written from the model source.

The rules are defined in ``Plasim_Indicators.md`` §2; each function names the
subsection and the Fortran routine it follows. Two grids are involved:

- PlaSim atmosphere, sea ice and slab ocean: T21 Gaussian grid, 32 x 64.
- LSG ocean: semi-staggered E grid, 76 rows x 72 points, 22 levels.

Arrays follow the order of the output files: latitude rows north to south,
LSG fields as ``(level, row, point)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

PLASIM_RADIUS_M = 6_371_220.0  # p_earth.f90, plarad
LSG_RADIUS_M = 6_371_000.0  # lsgmod.f90, inigr (erdrad)
LSG_GRID_STEP_DEG = 5.0  # lsgmod.f90, inipar (grad)
LSG_RHO0 = 1030.0  # kg m-3, lsgmod.f90 (rhonul, "changed according to PlaSim")
LSG_CP = 4180.0  # J kg-1 K-1, lsgmod.f90 (cp)

T21_NLAT, T21_NLON = 32, 64
LSG_NROW, LSG_NPOINT, LSG_NLEVEL = 76, 72, 22

# Layer centres du(1..23) in m, lsgmod.f90 inipar (ken == 22). du(23) is only
# used to decide where the deepest layer ends.
LSG_LAYER_CENTRES_M = np.array(
    [25, 75, 125, 175, 225, 275, 350, 450, 550, 650, 750, 850, 950, 1100,
     1300, 1500, 1800, 2250, 2750, 3500, 4500, 5500, 6500],
    dtype=float,
)

# Wet surface cells that LSG's basin file beklas leaves unlabelled (code 0),
# as (latitude, longitude) of the scalar point. Plasim_Indicators.md §2.3.
ATLANTIC_SIDE_UNLABELLED = (
    (18.75, 267.5),  # Gulf of Mexico
    (61.25, 310.0), (61.25, 315.0),  # Labrador Sea and southern Greenland coast
    (63.75, 307.5), (63.75, 312.5), (63.75, 317.5),
    (66.25, 310.0),
    (78.75, 287.5),  # northern Baffin Bay
)
OTHER_UNLABELLED = (
    (78.75, 267.5), (78.75, 272.5), (78.75, 277.5), (78.75, 282.5),  # Canadian Arctic
    (66.25, 185.0),  # Bering Strait
    (36.25, 20.0), (33.75, 22.5), (31.25, 20.0),  # Mediterranean
    (-3.75, 120.0), (-6.25, 117.5), (-13.75, 130.0),  # Indonesian seas
    (-21.25, 32.5),  # off Mozambique
)


# ---------------------------------------------------------------------------
# T21 grid (Plasim_Indicators.md §2.1)
# ---------------------------------------------------------------------------

def t21_gaussian_grid() -> tuple[np.ndarray, np.ndarray]:
    """Return T21 Gaussian latitudes (degrees, north to south) and weights.

    Gauss-Legendre quadrature of order 32: latitude = arcsin(node), and the
    weights sum to 2 (gaussmod.f90, inigau).
    """
    nodes, weights = np.polynomial.legendre.leggauss(T21_NLAT)
    latitude = np.degrees(np.arcsin(nodes))
    return latitude[::-1].copy(), weights[::-1].copy()


def t21_band_area(weights: np.ndarray) -> np.ndarray:
    """Return the area (m2) of each T21 latitude band: 2 pi R_P^2 w_j."""
    return 2.0 * np.pi * PLASIM_RADIUS_M**2 * np.asarray(weights, dtype=float)


def check_t21_latitudes(latitude: np.ndarray) -> None:
    """Check that file latitudes are the T21 Gaussian latitudes."""
    expected, _ = t21_gaussian_grid()
    latitude = np.asarray(latitude, dtype=float)
    if latitude.shape != expected.shape or np.max(np.abs(latitude - expected)) > 1e-6:
        raise ValueError("PLA latitudes are not the T21 Gaussian latitudes (north to south).")


@dataclass(frozen=True)
class T21Grid:
    """Static T21 geometry of one run."""

    lat: np.ndarray  # (32,) degrees north, north to south
    weight: np.ndarray  # (32,) Gaussian weights, sum 2
    band_area: np.ndarray  # (32,) m2
    ocean: np.ndarray  # (32, 64) bool, lsm < 0.5

    @property
    def ocean_fraction(self) -> np.ndarray:
        """Fraction of ocean points in each latitude row."""
        return self.ocean.mean(axis=1)


def t21_grid_from_pla(pla: xr.Dataset) -> T21Grid:
    """Build the T21 geometry from one PLA output file (checked)."""
    check_t21_latitudes(pla["lat"].values)
    lsm = np.asarray(pla["lsm"].isel(time=0).values, dtype=float)
    if not np.all((lsm == 0.0) | (lsm == 1.0)):
        raise ValueError("PLA lsm is not binary.")
    if not np.array_equal(lsm, np.asarray(pla["lsm"].isel(time=-1).values)):
        raise ValueError("PLA lsm changes within the decade.")
    _, weight = t21_gaussian_grid()
    return T21Grid(
        lat=np.asarray(pla["lat"].values, dtype=float),
        weight=weight,
        band_area=t21_band_area(weight),
        ocean=lsm < 0.5,
    )


# ---------------------------------------------------------------------------
# LSG grid (Plasim_Indicators.md §2.2)
# ---------------------------------------------------------------------------

def lsg_row_latitudes() -> np.ndarray:
    """Return the latitude (degrees) of LSG rows j = 1..76 (inigr).

    phi_j = 2.5 deg x (38 - j + 0.5); rows with |phi| > 90 deg are ghost rows.
    """
    rows = np.arange(1, LSG_NROW + 1)
    return 0.5 * LSG_GRID_STEP_DEG * (LSG_NROW / 2 - rows + 0.5)


def lsg_zonal_step(latitude: np.ndarray) -> np.ndarray:
    """Return LSG's zonal grid step dl = R_L 5 deg cos(phi) in m, 0 on ghost rows (inigr)."""
    latitude = np.asarray(latitude, dtype=float)
    step = LSG_RADIUS_M * np.deg2rad(LSG_GRID_STEP_DEG) * np.cos(np.deg2rad(latitude))
    return np.where(np.abs(latitude) <= 90.0, step, 0.0)


def lsg_point_area(latitude: np.ndarray) -> np.ndarray:
    """Return the horizontal area of one LSG point, 1/2 dl dphi, in m2.

    Scalar and vector points share the 5 x 5 degree boxes, hence the 1/2
    (LSG's own area sums use dl * dphi * 0.5).
    """
    dphi = LSG_RADIUS_M * np.deg2rad(LSG_GRID_STEP_DEG)
    return 0.5 * lsg_zonal_step(latitude) * dphi


def lsg_interfaces(centres: np.ndarray = LSG_LAYER_CENTRES_M) -> tuple[np.ndarray, np.ndarray]:
    """Return LSG bottom interfaces dw(1..22) and nominal thicknesses ddw(1..22) (inigr).

    dw(k) = (du(k) + du(k+1)) / 2 and ddw(k) = dw(k) - dw(k-1), with dw(0) = 0.
    """
    centres = np.asarray(centres, dtype=float)
    bottom = 0.5 * (centres[:-1] + centres[1:])
    nominal = np.diff(np.concatenate(([0.0], bottom)))
    return bottom, nominal


def lsg_layer_thickness(depth: np.ndarray, centres: np.ndarray = LSG_LAYER_CENTRES_M) -> np.ndarray:
    """Return the thickness (m) of each LSG layer at points of sea-floor depth D.

    The rule for ``ddz`` (scalar points, D = depp) and ``delta`` (vector
    points, D = depv) in lsgmod.f90:

    - 0 if D <= du(k) (the layer is dry);
    - D - dw(k-1) if du(k) < D < du(k+1) (the deepest wet layer reaches the floor);
    - ddw(k) otherwise.

    Output shape is ``(22, *depth.shape)``.
    """
    depth = np.asarray(depth, dtype=float)
    centres = np.asarray(centres, dtype=float)
    bottom, nominal = lsg_interfaces(centres)
    top = np.concatenate(([0.0], bottom[:-1]))
    column = (slice(None),) + (None,) * depth.ndim  # broadcast level arrays over points

    centre, next_centre = centres[:-1][column], centres[1:][column]
    is_wet = depth > centre
    reaches_floor = is_wet & (depth < next_centre)
    thickness = np.where(is_wet, nominal[column], 0.0)
    return np.where(reaches_floor, depth - top[column], thickness)


def check_lsg_wet_masks(wet: np.ndarray, wetvec: np.ndarray, depp: np.ndarray, depv: np.ndarray) -> None:
    """Check that the file's wet masks follow LSG's rules.

    Scalar layer k is wet where depp > du(k); vector layer k is wet where its
    thickness delta exceeds 0.1 m (lsgmod.f90).
    """
    scalar_rule = np.asarray(depp)[None] > LSG_LAYER_CENTRES_M[:-1, None, None]
    vector_rule = lsg_layer_thickness(depv) > 0.1
    if not np.array_equal(np.asarray(wet) > 0.5, scalar_rule):
        raise ValueError("LSG wet mask does not follow depp > du(k).")
    if not np.array_equal(np.asarray(wetvec) > 0.5, vector_rule):
        raise ValueError("LSG wetvec mask does not follow delta > 0.1 m.")


def read_beklas(path: Path) -> np.ndarray:
    """Read LSG's basin file ``beklas`` as a (76, 72) integer array.

    Format (1x,72i1) for rows 3..74 (lsgmod.f90, unit 79); rows 1, 2, 75 and
    76 are 0. Codes: 1 Atlantic, 2 Pacific, 3 Indian, 4 Arctic, 0 land.
    """
    lines = Path(path).read_text().splitlines()
    rows = [line[1:1 + LSG_NPOINT] for line in lines if line.strip()]
    if len(rows) != LSG_NROW - 4 or any(len(row) != LSG_NPOINT for row in rows):
        raise ValueError(f"{path} is not a 72 x 72 beklas file.")
    codes = np.zeros((LSG_NROW, LSG_NPOINT), dtype=int)
    codes[2:LSG_NROW - 2] = [[int(character) for character in row] for row in rows]
    return codes


def _cells_at(latitude: np.ndarray, longitude: np.ndarray, points) -> np.ndarray:
    """Boolean mask of the grid points at the listed (latitude, longitude)."""
    mask = np.zeros(latitude.shape, dtype=bool)
    for point_lat, point_lon in points:
        mask |= np.isclose(latitude, point_lat) & np.isclose(longitude, point_lon)
    return mask


def _box(latitude, longitude, lat_range, lon_range) -> np.ndarray:
    """Boolean mask of lat_range[0] <= lat < lat_range[1] and the same for longitude."""
    return ((latitude >= lat_range[0]) & (latitude < lat_range[1])
            & (longitude >= lon_range[0]) & (longitude < lon_range[1]))


def atlantic_mask(codes: np.ndarray, latitude: np.ndarray, longitude: np.ndarray,
                  wet_surface: np.ndarray) -> np.ndarray:
    """Return the Atlantic mask on LSG scalar points (Plasim_Indicators.md §2.3).

    1. beklas codes 1 (Atlantic) and 4 (Arctic), on wet surface points;
    2. plus the unlabelled wet cells listed in ``ATLANTIC_SIDE_UNLABELLED``;
       an unlabelled wet cell in neither list raises an error;
    3. minus: south of 34 S, the Mediterranean, Hudson Bay, north of 80 N and
       the Arctic Ocean proper (Boerner et al. 2026, regions.Atlantic).

    Longitudes are in [0, 360).
    """
    wet_surface = np.asarray(wet_surface, dtype=bool)
    unlabelled = wet_surface & (codes == 0)
    atlantic_side = _cells_at(latitude, longitude, ATLANTIC_SIDE_UNLABELLED)
    other = _cells_at(latitude, longitude, OTHER_UNLABELLED)
    unknown = unlabelled & ~atlantic_side & ~other
    if unknown.any():
        cells = [(float(latitude[j, i]), float(longitude[j, i])) for j, i in zip(*np.nonzero(unknown))]
        raise ValueError(f"Unlabelled wet cells not in the expected lists: {cells}")

    mask = wet_surface & ((codes == 1) | (codes == 4) | (unlabelled & atlantic_side))
    mask &= latitude >= -34.0
    mask &= ~_box(latitude, longitude, (20.0, 40.0), (0.0, 40.0))  # Mediterranean I
    mask &= ~_box(latitude, longitude, (20.0, 40.0), (355.0, 360.0))  # Mediterranean II
    mask &= ~_box(latitude, longitude, (50.0, 64.0), (265.0, 283.0))  # Hudson Bay
    mask &= latitude < 80.0
    mask &= ~_box(latitude, longitude, (65.0, 90.0), (20.0, 285.0))  # Arctic Ocean proper
    return mask


def south_american_land(latitude: np.ndarray, longitude: np.ndarray, wet_surface: np.ndarray) -> np.ndarray:
    """Return surface-dry scalar points at 20-50 S, 285-315 E (for the ACC, §3.3)."""
    return (~np.asarray(wet_surface, dtype=bool)
            & (latitude >= -50.0) & (latitude <= -20.0)
            & (longitude >= 285.0) & (longitude <= 315.0))


@dataclass(frozen=True)
class LsgGrid:
    """Static LSG geometry of one run. Fields are (level, row, point) or (row, point)."""

    lat: np.ndarray  # (76, 72) scalar-point latitude
    lon: np.ndarray  # (76, 72) scalar-point longitude in [0, 360)
    row_lat: np.ndarray  # (76,) row latitude (scalar and vector points of a row share it)
    area: np.ndarray  # (76, 72) m2 per point
    zonal_step: np.ndarray  # (76,) dl, m
    wet: np.ndarray  # (22, 76, 72) bool, scalar points
    ddz: np.ndarray  # (22, 76, 72) m, scalar-point layer thickness (0 where dry)
    wetvec: np.ndarray  # (22, 76, 72) bool, vector points
    delta: np.ndarray  # (22, 76, 72) m, vector-point layer thickness (0 where dry)
    volume: np.ndarray  # (22, 76, 72) m3, wet scalar-cell volume without zeta
    atlantic: np.ndarray  # (76, 72) bool, scalar points; also used for vector point (i, j)
    south_american_land: np.ndarray  # (76, 72) bool


def lsg_grid_from_lsg(lsg: xr.Dataset, beklas_path: Path) -> LsgGrid:
    """Build the LSG geometry from one LSG output file and the run's beklas (checked)."""
    latitude = np.asarray(lsg["lat"].values, dtype=float)
    longitude = np.asarray(lsg["lon"].values, dtype=float) % 360.0
    row_lat = lsg_row_latitudes()
    if not np.allclose(latitude, row_lat[:, None]):
        raise ValueError("LSG scalar latitudes do not follow phi_j = 2.5 (38 - j + 0.5).")
    if not np.allclose(np.asarray(lsg["lat_2"].values, dtype=float), latitude):
        raise ValueError("LSG vector rows are not at the scalar-row latitudes.")
    if not np.allclose(lsg["depth"].values, LSG_LAYER_CENTRES_M[:-1]):
        raise ValueError("LSG depth levels differ from du(1..22).")

    depp = np.asarray(lsg["depp"].isel(time=0, lev=0).values, dtype=float)
    depv = np.asarray(lsg["depv"].isel(time=0, lev=0).values, dtype=float)
    wet = np.asarray(lsg["wet"].isel(time=0).values) > 0.5
    wetvec = np.asarray(lsg["wetvec"].isel(time=0).values) > 0.5
    check_lsg_wet_masks(wet, wetvec, depp, depv)

    ddz = lsg_layer_thickness(depp) * wet
    delta = lsg_layer_thickness(depv) * wetvec
    area = lsg_point_area(latitude)
    codes = read_beklas(beklas_path)
    return LsgGrid(
        lat=latitude,
        lon=longitude,
        row_lat=row_lat,
        area=area,
        zonal_step=lsg_zonal_step(row_lat),
        wet=wet,
        ddz=ddz,
        wetvec=wetvec,
        delta=delta,
        volume=ddz * area[None],
        atlantic=atlantic_mask(codes, latitude, longitude, wet[0]),
        south_american_land=south_american_land(latitude, longitude, wet[0]),
    )
