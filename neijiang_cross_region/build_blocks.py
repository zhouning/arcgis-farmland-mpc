"""Neijiang Dongxing cross-region: block definition runner.

Reuses the research-side block_definition.py / block_definition_all.py but
points them at Neijiang's DLTB_with_slope.gpkg, with Neijiang township codes,
and writes blocks into ./blocks/.

NOTE (public release): configuration reference, not a runnable artifact. Depends
on the research-side block_definition modules (not part of this package) and the
raw 三调 GeoPackage (RESTRICTED, see Data Availability). Set paths below first.

Strategy: monkey-patch the module-level DLTB_PATH and OUTPUT_DIR constants
before calling the Paper 3 functions. No changes to the original Paper 3 code.
"""
import os
import sys
import json
import time

SCRIPT_DIR = os.environ.get("P9_RESEARCH_DIR", "/path/to/research/checkout")
sys.path.insert(0, SCRIPT_DIR)

# ---- Neijiang-specific config ----
NEIJIANG_GPKG = os.environ.get("NEIJIANG_GPKG", "/path/to/neijiang_DLTB_with_slope.gpkg")
NEIJIANG_BLOCK_DIR = os.environ.get(
    "NEIJIANG_BLOCK_DIR",
    os.path.join(os.path.dirname(__file__), "blocks"),
)

# Neijiang Dongxing townships (QSDWDM 9-digit prefix),
# filtered to >=500 parcels (drops 511011214:98 and 512021243:1).
NEIJIANG_TOWNSHIPS = {
    '511011001': 'N01-Neijiang-001',
    '511011002': 'N02-Neijiang-002',
    '511011003': 'N03-Neijiang-003',
    '511011100': 'N04-Township-100',
    '511011101': 'N05-Township-101',
    '511011102': 'N06-Township-102',
    '511011103': 'N07-Township-103',
    '511011104': 'N08-Township-104',
    '511011105': 'N09-Township-105',
    '511011106': 'N10-Township-106',
    '511011107': 'N11-Township-107',
    '511011108': 'N12-Township-108',
    '511011109': 'N13-Township-109',
    '511011110': 'N14-Township-110',
    '511011111': 'N15-Township-111',
    '511011200': 'N16-Township-200',
    '511011201': 'N17-Township-201',
    '511011202': 'N18-Township-202',
    '511011203': 'N19-Township-203',
    '511011204': 'N20-Township-204',
    '511011205': 'N21-Township-205',
    '511011206': 'N22-Township-206',
    '511011207': 'N23-Township-207',
    '511011208': 'N24-Township-208',
    '511011209': 'N25-Township-209',
    '511011210': 'N26-Township-210',
    '511011211': 'N27-Township-211',
    '511011212': 'N28-Township-212',
    '511011213': 'N29-Township-213',
}

def main():
    # Monkey-patch Paper 3 module constants
    import block_definition as bd
    bd.DLTB_PATH = NEIJIANG_GPKG
    bd.OUTPUT_DIR = NEIJIANG_BLOCK_DIR
    # Paper 3's classify_parcel uses 'farmland'/'forest'/'barrier'/'other',
    # but Neijiang's DLTB_with_slope.gpkg has a 'category' column that uses
    # 'Farmland'/'Forest'/'Orchard'/'Other' (title-case). We override
    # classify_parcel to reclassify from DLBM prefixes directly to Paper 3's
    # naming convention; this is safe because both Bishan and Neijiang use
    # the same Third National Land Survey DLBM codes.
    FARM = {'011', '012', '013'}
    FOREST = {'031', '032', '033'}
    # Barriers: roads, water, construction (Paper 3 uses these to segment blocks)
    BARRIER_PREFIXES = ('1', '2', '04')  # roads+1xx, construction+2xx, water+04x
    def neijiang_classify(dlbm):
        s = str(dlbm) if dlbm is not None else ''
        if s[:3] in FARM:
            return 'farmland'
        if s[:3] in FOREST:
            return 'forest'
        # Barriers: roads/construction/water
        if s.startswith(('10', '11', '20', '21', '04')):
            return 'barrier'
        return 'other'
    bd.classify_parcel = neijiang_classify
    os.makedirs(NEIJIANG_BLOCK_DIR, exist_ok=True)

    from block_definition import define_blocks, save_results

    t0 = time.time()
    summary = {}
    for code, label in NEIJIANG_TOWNSHIPS.items():
        out_dir = os.path.join(NEIJIANG_BLOCK_DIR, f'township_{code}')
        done_files = ['block_compositions.json', 'block_features.json', 'parcel_block_mapping.csv']
        if all(os.path.exists(os.path.join(out_dir, f)) for f in done_files):
            print(f"[SKIP] {code} ({label}): already processed")
            continue
        print(f"\n{'='*70}\n  {code} ({label})\n{'='*70}")
        try:
            gdf_swap, block_feats, valid_blocks = define_blocks(
                code, min_parcels=3, min_area_ha=0.5, max_parcels=30,
            )
            save_results(code, gdf_swap, block_feats, valid_blocks)
            summary[code] = {
                'label': label,
                'n_swappable': len(gdf_swap),
                'n_blocks': len(block_feats),
                'n_baimu_ge_6_67ha': sum(1 for b in block_feats if b['total_area_ha'] >= 6.67),
                'total_area_ha': sum(b['total_area_ha'] for b in block_feats),
            }
        except Exception as e:
            print(f"  [ERROR] {code}: {e}")
            import traceback; traceback.print_exc()
            summary[code] = {'error': str(e)}

    with open(os.path.join(NEIJIANG_BLOCK_DIR, 'neijiang_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    total_blocks = sum(s.get('n_blocks', 0) for s in summary.values() if 'n_blocks' in s)
    total_baimu = sum(s.get('n_baimu_ge_6_67ha', 0) for s in summary.values() if 'n_blocks' in s)
    total_area = sum(s.get('total_area_ha', 0) for s in summary.values() if 'n_blocks' in s)
    ok = sum(1 for s in summary.values() if 'n_blocks' in s)
    print(f"\n{'='*70}")
    print(f"  Neijiang block definition complete in {time.time()-t0:.1f}s")
    print(f"  Townships processed: {ok}/{len(NEIJIANG_TOWNSHIPS)}")
    print(f"  Total blocks: {total_blocks}")
    print(f"  Total baimu fang (>=6.67 ha blocks): {total_baimu}")
    print(f"  Total swappable area: {total_area:,.0f} ha")
    print(f"  Output: {NEIJIANG_BLOCK_DIR}")

if __name__ == '__main__':
    main()
