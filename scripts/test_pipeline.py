"""
SmartSDG Pipeline v2 — end-to-end self-test.

Run after any change to verify the whole chain still works:

    python scripts\\test_pipeline.py            # static checks only (~5 s, no GPU)
    python scripts\\test_pipeline.py --full     # + real SDG generation + real training (~5-10 min)

Tiers:
  1. STATIC   configs parse (and have no BOM), assets/interpreters exist,
              class registry consistent with the AAS template, python modules
              compile, converter quality filters behave (synthetic data),
              exp-folder routing correct for all dataset naming patterns.
  2. SDG      real Isaac Sim run (2 samples, 320x240, fast settings) ->
              images + labels exist, resolution correct, class IDs in range.
  3. TRAIN    per-dataset AAS generated -> AAS->yaml conversion -> 1-epoch
              YOLO11n training on the test dataset -> weights produced.

Exit code 0 = all passed. Test artifacts are cleaned up afterwards.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # SDG_v2
SMARTSDG = ROOT / "SmartSDG"
OD_DIR = ROOT / "OD_training"
AAS_DIR = ROOT / "AAS"
SDG_CONFIG = SMARTSDG / "src" / "config" / "config.yaml"
TEMPLATE_AAS = AAS_DIR / "CA_Modul-OD_pipeline_v2.0.json"
ISAAC_PYTHON = Path(r"C:\Users\hota02\Documents\omniverse\python.bat")
TRAIN_PYTHON = Path(r"C:\Users\hota02\AppData\Local\anaconda3\envs\yolov12\python.exe")

TEST_DS = "MULTI_Cabin-GlueGun_Medium_plane"  # produced by the SDG tier (2 components)

RESULTS = []


def check(name, fn):
    """Run one named check, record PASS/FAIL."""
    t0 = time.time()
    try:
        msg = fn() or ""
        RESULTS.append((True, name, msg, time.time() - t0))
        print(f"  [PASS] {name}" + (f"  ({msg})" if msg else ""))
    except Exception as e:
        RESULTS.append((False, name, str(e), time.time() - t0))
        print(f"  [FAIL] {name}: {e}")


# ======================================================================
# TIER 1 — STATIC
# ======================================================================

def t_no_bom():
    for f in [SDG_CONFIG, TEMPLATE_AAS]:
        if f.read_bytes()[:3] == b"\xef\xbb\xbf":
            raise AssertionError(f"{f.name} has a UTF-8 BOM (breaks Isaac/aas_core3 parsers)")
    return "config.yaml + template AAS are BOM-free"


def t_config_parses():
    import yaml
    cfg = yaml.safe_load(SDG_CONFIG.read_text(encoding="utf-8"))
    for section in ["paths", "generation_params", "COMPONENT_USD_MAP", "YOLO_CONVERSION",
                    "YOLO_FILTERING", "LIGHT_INTENSITY_MAP", "SIMULATION_CONFIG"]:
        assert section in cfg, f"missing section: {section}"
    assert set(cfg["LIGHT_INTENSITY_MAP"]) == {"Low", "Medium", "High"}, \
        f"LIGHT_INTENSITY_MAP keys must be Low/Medium/High, got {list(cfg['LIGHT_INTENSITY_MAP'])}"
    return f"{len(cfg['COMPONENT_USD_MAP'])} components registered"


def t_usd_files():
    import yaml
    cfg = yaml.safe_load(SDG_CONFIG.read_text(encoding="utf-8"))
    assets = SMARTSDG / cfg["paths"]["base"] / cfg["paths"]["components_rel"]
    missing = [f"{n} ({u})" for n, u in cfg["COMPONENT_USD_MAP"].items() if not (assets / u).exists()]
    assert not missing, f"USD files missing: {missing}"
    for env_key in ["env_usd_rel", "flat_env_usd_rel"]:
        p = SMARTSDG / cfg["paths"]["base"] / cfg["paths"][env_key]
        assert p.exists(), f"environment USD missing: {p}"
    return "all component + environment USDs present"


def t_aas_template():
    sys.path.insert(0, str(ROOT / "dashboard"))
    import server
    import yaml
    data = json.loads(TEMPLATE_AAS.read_text(encoding="utf-8-sig"))
    raw = server._aas_get_property(data, "names")
    aas_names = [n.strip() for n in (raw or "").split(",") if n.strip()]
    cfg = yaml.safe_load(SDG_CONFIG.read_text(encoding="utf-8"))
    registry = list(cfg["COMPONENT_USD_MAP"].keys())
    assert aas_names == registry, \
        f"AAS class list != ID registry.\n  AAS:      {aas_names}\n  registry: {registry}"
    return f"AAS classes match ID registry ({len(registry)} classes)"


def t_interpreters():
    assert ISAAC_PYTHON.exists(), f"Isaac Sim python.bat not found: {ISAAC_PYTHON}"
    assert TRAIN_PYTHON.exists(), f"training interpreter not found: {TRAIN_PYTHON}"
    r = subprocess.run([str(TRAIN_PYTHON), "-c",
                        "import aas_core3, ultralytics, torch, yaml, numpy, PIL"],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"training env missing packages: {r.stderr.strip().splitlines()[-1]}"
    return "Isaac + training interpreters OK, deps importable"


def t_compile():
    files = [SMARTSDG / "main.py", SMARTSDG / "src" / "generation.py",
             SMARTSDG / "src" / "yolo_converter.py", SMARTSDG / "src" / "config_process.py",
             SMARTSDG / "src" / "config_loader.py", OD_DIR / "main.py",
             ROOT / "dashboard" / "server.py",
             ROOT / "active_learning" / "utils_yolo.py",
             ROOT / "active_learning" / "main_yolo.py",
             ROOT / "active_learning" / "run_al_from_aas.py",
             ROOT / "active_learning" / "harvest_from_inference.py"]
    r = subprocess.run([str(TRAIN_PYTHON), "-m", "py_compile"] + [str(f) for f in files],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr.strip()
    return f"{len(files)} modules compile"


def t_converter_filters():
    """Synthetic Replicator output through the converter: every quality filter
    must trigger, and the never-keep-unlabeled-objects invariant must hold."""
    code = r'''
import sys, os, json
import numpy as np
from PIL import Image
from pathlib import Path
sys.path.insert(0, r"{src}")
from types import SimpleNamespace
import yolo_converter

tmp = Path(r"{tmp}")
rp = tmp / "src" / "Replicator"
(rp / "rgb").mkdir(parents=True, exist_ok=True)
(rp / "bounding_box_2d_tight").mkdir(parents=True, exist_ok=True)
W, H = 640, 400
dt = np.dtype([("semanticId","<u4"),("x_min","<i4"),("y_min","<i4"),("x_max","<i4"),("y_max","<i4"),("occlusionRatio","<f4")])
frames = [
    [(0,100,100,200,200,0.0),(1,250,100,350,200,0.2)],   # occluded -> frame dropped
    [(0,100,100,200,200,0.0),(2,0,150,80,250,0.0)],      # border-cut -> frame dropped
    [(0,100,100,200,200,0.0),(3,300,300,300,350,0.0)],   # degenerate -> frame dropped
    [(0,100,100,200,200,0.0),(2,300,200,400,300,0.01)],  # clean -> kept
]
for i, boxes in enumerate(frames):
    Image.new("RGB",(W,H)).save(rp/"rgb"/("rgb_%04d.png" % i))
    np.save(rp/"bounding_box_2d_tight"/("bounding_box_2d_tight_%04d.npy" % i), np.array(boxes,dtype=dt))
    (rp/"bounding_box_2d_tight"/("bounding_box_2d_tight_labels_%04d.json" % i)).write_text(
        json.dumps({{"0":{{"class":"ylid"}},"1":{{"class":"cabin"}},"2":{{"class":"gluegun"}},"3":{{"class":"blid"}}}}))
classes = ["Ylid","YTrailer","Cabin","GlueGun","BLid","BTrailer","FChassis","Rchassis","BlackCabin"]
cfg = SimpleNamespace(YOLO_CONVERSION=SimpleNamespace(output_dir_abs=tmp/"out", CLASS_NAMES=classes))
out, stats = yolo_converter.run_yolo_conversion(str(tmp/"src"), cfg, "Medium",
    min_visibility=0.95, drop_truncated=True, truncation_policy="frame", occlusion_policy="frame")
assert stats["frames_dropped_occluded"] == 1, stats
assert stats["frames_dropped_truncated"] == 1, stats
assert stats["frames_dropped_degenerate"] == 1, stats
assert stats["frames_kept"] == 1, stats
kept = list(Path(out).rglob("*.png"))
assert len(kept) == 1, kept
print("CONVERTER_OK")
'''
    tmp = tempfile.mkdtemp(prefix="sdgtest_conv_")
    try:
        r = subprocess.run([str(TRAIN_PYTHON), "-X", "utf8", "-c",
                            code.format(src=SMARTSDG / "src", tmp=tmp)],
                           capture_output=True, text=True)
        assert "CONVERTER_OK" in r.stdout, (r.stderr or r.stdout).strip()[-400:]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return "occlusion/truncation/degenerate all drop whole frames; clean frame kept"


def t_exp_routing():
    code = r'''
import sys; sys.path.insert(0, r"{od}")
from main import parse_sdg_metadata_from_path
cases = {{
 "MULTI_GlueGun-BLid-BTrailer_Medium_withenv": "withenv",
 "MULTI_ALL_CLASSES_Medium_plane": "plane",
 "SINGLE_GlueGun_Medium_plane": "plane",
 "SINGLE_GlueGun_High_withoutenv": "withoutenv",
 "MULTI_Cabin-GlueGun_Medium_plane": "plane",
}}
for name, env in cases.items():
    got = parse_sdg_metadata_from_path(name)["env"]
    assert got == env, f"{{name}}: expected {{env}}, got {{got}}"
print("ROUTING_OK")
'''
    r = subprocess.run([str(TRAIN_PYTHON), "-X", "utf8", "-c", code.format(od=OD_DIR)],
                       capture_output=True, text=True)
    assert "ROUTING_OK" in r.stdout, (r.stderr or r.stdout).strip()[-400:]
    return "5 dataset-name patterns route to the correct exp folder"


def t_al_strategies():
    """AL selection module imports and exposes the expected strategy helpers."""
    r = subprocess.run([str(TRAIN_PYTHON), "-c",
                        "import sys; sys.path.insert(0, r'%s'); import utils_yolo as u; "
                        "assert all(hasattr(u, f) for f in "
                        "['get_images_for_next_round','_diversity_features','_diversity_box',"
                        "'_uncertainty_scores','count_labeled_classes'])"
                        % (AL_DIR := ROOT / 'active_learning')],
                       capture_output=True, text=True)
    assert r.returncode == 0, (r.stderr or "").strip()[-300:]
    return "random/entropy/ppal_box/ppal_feat + class-balance present"


# ======================================================================
# TIER 2 — SDG (real Isaac Sim run)
# ======================================================================

def t_sdg_generation():
    # remove any previous test dataset
    for d in [SMARTSDG / "Datasets" / "yolo_experiments" / TEST_DS,
              SMARTSDG / "Datasets" / "unified_render_output" / TEST_DS]:
        shutil.rmtree(d, ignore_errors=True)

    cmd = ["cmd", "/c", str(ISAAC_PYTHON), "main.py",
           "--config", r"src\config\config.yaml",
           "--multi_object", "--components", "Cabin,GlueGun",
           "--light_mode", "Medium", "--num_samples", "2", "--load_env", "plane",
           "--width", "320", "--height", "240", "--spp", "8",
           "--convert_to_yolo", "--drop_truncated", "--seed", "123"]
    r = subprocess.run(cmd, cwd=str(SMARTSDG), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=900)
    assert r.returncode == 0, f"SDG exited {r.returncode}: {(r.stdout or '')[-300:]}"

    ds = SMARTSDG / "Datasets" / "yolo_experiments" / TEST_DS
    pngs = list(ds.rglob("*.png"))
    assert pngs, "no images produced"
    from PIL import Image
    with Image.open(pngs[0]) as img:
        assert img.size == (320, 240), f"resolution {img.size} != (320, 240)"
    n_classes = 9
    for txt in ds.rglob("*.txt"):
        for line in txt.read_text().split("\n"):
            if line.strip():
                cid = int(line.split()[0])
                assert 0 <= cid < n_classes, f"class ID {cid} out of range in {txt.name}"
    return f"{len(pngs)} images at 320x240, labels valid"


# ======================================================================
# TIER 3 — AAS + training
# ======================================================================

def t_dataset_aas():
    sys.path.insert(0, str(ROOT / "dashboard"))
    import server
    rel = str((SMARTSDG / "Datasets" / "yolo_experiments" / TEST_DS).relative_to(ROOT))
    aas_rel = server.generate_dataset_aas(rel)
    aas_path = AAS_DIR / aas_rel
    assert aas_path.exists(), f"AAS not created: {aas_path}"
    assert aas_path.read_bytes()[:3] != b"\xef\xbb\xbf", "generated AAS has a BOM"
    data = json.loads(aas_path.read_text(encoding="utf-8"))
    assert TEST_DS in (server._aas_get_property(data, "base_data_dir") or "")
    return f"AAS/{aas_rel}"


def t_training():
    sys.path.insert(0, str(ROOT / "dashboard"))
    import server
    existing = list((AAS_DIR / "datasets").rglob(f"CA_Modul-OD_dataset_{TEST_DS}.json"))
    assert existing, "dataset AAS missing (t_dataset_aas must run first)"
    # fast settings for the smoke test
    data = json.loads(existing[0].read_text(encoding="utf-8-sig"))
    server._aas_set_properties(data, {"epochs": 1, "img_size": 320, "batch_size": 4,
                                      "weights": "./yolo11n.pt",
                                      "cameras_to_train": "Replicator"})
    tmp_aas = Path(tempfile.gettempdir()) / "sdgtest_train_aas.json"
    tmp_aas.write_text(json.dumps(data, indent=1), encoding="utf-8")

    r = subprocess.run([str(TRAIN_PYTHON), "main.py", "--mode", "train",
                        "--aas_file", str(tmp_aas)],
                       cwd=str(OD_DIR), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=1800)
    assert r.returncode == 0, f"training exited {r.returncode}: {(r.stdout or '')[-300:]}"
    assert "All Training Runs Completed" in (r.stdout or "")

    run_root = OD_DIR / "results" / "yolo_run" / "exp1" / TEST_DS   # plane -> exp1
    weights = list(run_root.rglob("weights/best.pt"))
    assert weights, f"no best.pt under {run_root} — training did not complete an epoch"
    return f"trained, weights at {weights[0].relative_to(OD_DIR)}"


def t_al_loop():
    """Functional AL smoke: run a 1-round experiment on the test dataset as the
    unlabeled pool and confirm al_summary.csv gets a per-round learning curve.
    Uses tiny settings — proves the loop completes, not the mAP."""
    sys.path.insert(0, str(ROOT / "active_learning"))
    sys.path.insert(0, str(ROOT / "dashboard"))
    import server
    ds_rel = str((SMARTSDG / "Datasets" / "yolo_experiments" / TEST_DS).relative_to(ROOT))
    n_pool, n_val, names, img_ext = server.prepare_al_workspace(ds_rel)
    assert n_pool >= 6, f"pool too small for AL test: {n_pool}"

    al_out = ROOT / "active_learning" / "output" / f"_selftest_{TEST_DS}"
    cfg = {
        "data_dir": str(ROOT / "active_learning" / "data"),
        "output_dir": str(al_out),
        "initial_model_path": "yolo11n.pt",
        "dataset_yaml": str(ROOT / "active_learning" / "data" / "full-data" / "dataset.yaml"),
        "num_classes": len(names), "al_strategy": "ppal_feat", "al_class_balance": True,
        "al_rounds": 1, "initial_subset_size": 4, "add_per_round": 3,
        "epochs_per_round": 1, "batch_size": 4, "img_size": 320,
        "al_score_threshold": 0.10, "ud_split": 0.5,
        "PROJECT_NAME": "selftest_al", "F_FULL_DATA": "full-data", "F_TRAIN": "train",
        "F_VAL": "val", "F_WORKDIR": "workspace", "F_UNLABELED": "unlabeled",
        "IMG_EXTENSION": img_ext, "ANN_EXTENSION": ".txt", "SEED": 42,
    }
    code = ("import sys; sys.path.insert(0, r'%s'); "
            "from main_yolo import run_al_pipeline; import json; "
            "run_al_pipeline(json.loads(r'''%s'''))"
            % (ROOT / "active_learning", json.dumps(cfg)))
    r = subprocess.run([str(TRAIN_PYTHON), "-X", "utf8", "-c", code],
                       cwd=str(ROOT / "active_learning"), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=1800)
    summary = al_out / "al_summary.csv"
    try:
        assert r.returncode == 0, f"AL exited {r.returncode}: {(r.stdout or r.stderr or '')[-300:]}"
        assert summary.exists(), "al_summary.csv not written"
        rows = summary.read_text().strip().splitlines()
        assert len(rows) >= 3, f"expected round 0 + round 1 in summary, got {len(rows)-1} rounds"
        return f"AL loop completed, {len(rows)-1} rounds logged to al_summary.csv"
    finally:
        shutil.rmtree(al_out, ignore_errors=True)
        shutil.rmtree(ROOT / "active_learning" / "data", ignore_errors=True)


def cleanup_test_artifacts():
    """Remove everything the full test created."""
    for d in [SMARTSDG / "Datasets" / "yolo_experiments" / TEST_DS,
              SMARTSDG / "Datasets" / "unified_render_output" / TEST_DS,
              OD_DIR / "results" / "yolo_run" / "exp1" / TEST_DS,
              OD_DIR / "results" / "experiment_outputs" / "exp1" / TEST_DS]:
        shutil.rmtree(d, ignore_errors=True)
    if (AAS_DIR / "datasets").exists():
        for f in (AAS_DIR / "datasets").rglob(f"CA_Modul-OD_dataset_{TEST_DS}.json"):
            f.unlink()
    print("  test artifacts cleaned up")


# ======================================================================

def main():
    ap = argparse.ArgumentParser(description="SmartSDG pipeline self-test")
    ap.add_argument("--full", action="store_true",
                    help="also run real SDG generation + training (needs Isaac Sim + GPU, ~5-10 min)")
    ap.add_argument("--keep", action="store_true", help="keep test artifacts (dataset, weights)")
    args = ap.parse_args()

    print("=" * 64)
    print("TIER 1 — static checks")
    print("=" * 64)
    check("no BOM in configs", t_no_bom)
    check("config.yaml structure", t_config_parses)
    check("USD assets present", t_usd_files)
    check("AAS template vs class-ID registry", t_aas_template)
    check("interpreters + dependencies", t_interpreters)
    check("python modules compile", t_compile)
    check("converter quality filters (synthetic)", t_converter_filters)
    check("exp-folder routing", t_exp_routing)
    check("AL strategies importable", t_al_strategies)

    if args.full:
        print("=" * 64)
        print("TIER 2 — SDG generation (Isaac Sim, ~2 min)")
        print("=" * 64)
        check("SDG end-to-end generation", t_sdg_generation)

        print("=" * 64)
        print("TIER 3 — AAS + training (~2-5 min)")
        print("=" * 64)
        check("per-dataset AAS generation", t_dataset_aas)
        check("OD training on generated dataset", t_training)

        print("=" * 64)
        print("TIER 4 — Active Learning loop (~3-5 min)")
        print("=" * 64)
        check("AL multi-round loop + learning curve", t_al_loop)

        if not args.keep:
            cleanup_test_artifacts()

    print("=" * 64)
    passed = sum(1 for ok, *_ in RESULTS if ok)
    failed = len(RESULTS) - passed
    total_t = sum(t for *_, t in RESULTS)
    print(f"RESULT: {passed} passed, {failed} failed  ({total_t:.0f}s)")
    for ok, name, msg, _ in RESULTS:
        if not ok:
            print(f"  FAILED: {name} — {msg}")
    print("=" * 64)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
