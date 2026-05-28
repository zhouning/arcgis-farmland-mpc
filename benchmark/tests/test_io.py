import json
from pathlib import Path
import geopandas as gpd
from shapely.geometry import box

from generator.io import write_synthetic_dataset


def _toy_dataset():
    """Two blocks, each with 3 parcels, two townships (1 block each)."""
    blocks = [box(0, 0, 100, 100), box(100, 0, 200, 100)]
    parcels_per_block = [
        [box(0, 0, 50, 100), box(50, 0, 100, 50), box(50, 50, 100, 100)],
        [box(100, 0, 150, 100), box(150, 0, 200, 50), box(150, 50, 200, 100)],
    ]
    parcel_dlbm = [
        ["0110", "0110", "0310"],
        ["0110", "0310", "0510"],
    ]
    parcel_slope = [
        [3.0, 3.5, 5.0],
        [4.0, 6.0, 2.0],
    ]
    block_township_id = [0, 1]
    block_compactness = [0.85, 0.85]
    return {
        "blocks": blocks,
        "parcels_per_block": parcels_per_block,
        "parcel_dlbm": parcel_dlbm,
        "parcel_slope": parcel_slope,
        "block_township_id": block_township_id,
        "block_compactness": block_compactness,
        "township_codes": ["999999001", "999999002"],
    }


def test_geopackage_written_with_required_columns(tmp_out_dir):
    write_synthetic_dataset(_toy_dataset(), tmp_out_dir)
    gpkg = tmp_out_dir / "DLTB_with_slope.gpkg"
    assert gpkg.exists()
    gdf = gpd.read_file(gpkg)
    for col in ("DLBM", "QSDWDM", "slope_mean"):
        assert col in gdf.columns
    assert len(gdf) == 6


def test_qsdwdm_township_prefix(tmp_out_dir):
    write_synthetic_dataset(_toy_dataset(), tmp_out_dir)
    gdf = gpd.read_file(tmp_out_dir / "DLTB_with_slope.gpkg")
    prefixes = {q[:9] for q in gdf["QSDWDM"]}
    assert prefixes == {"999999001", "999999002"}


def test_block_compositions_json_per_township(tmp_out_dir):
    write_synthetic_dataset(_toy_dataset(), tmp_out_dir)
    for code in ("999999001", "999999002"):
        comp_path = tmp_out_dir / f"township_{code}" / "block_compositions.json"
        feat_path = tmp_out_dir / f"township_{code}" / "block_features.json"
        assert comp_path.exists()
        assert feat_path.exists()
        with open(comp_path) as f:
            comp = json.load(f)
        assert len(comp) == 1
        local_indices = comp["0"]
        assert all(isinstance(i, int) for i in local_indices)


def test_only_swappable_parcels_in_compositions(tmp_out_dir):
    """Parcel with dlbm '0510' (other) must NOT appear in compositions."""
    write_synthetic_dataset(_toy_dataset(), tmp_out_dir)
    comp = json.loads(
        (tmp_out_dir / "township_999999002" / "block_compositions.json").read_text()
    )
    # Township 2 block has 3 parcels: 0110, 0310, 0510. Only first two are swappable.
    assert len(comp["0"]) == 2


def test_full_swappable_composition_values(tmp_out_dir):
    """Township 1 block has 3 swappable parcels; composition must be [0, 1, 2]."""
    write_synthetic_dataset(_toy_dataset(), tmp_out_dir)
    comp = json.loads(
        (tmp_out_dir / "township_999999001" / "block_compositions.json").read_text()
    )
    assert comp["0"] == [0, 1, 2]
