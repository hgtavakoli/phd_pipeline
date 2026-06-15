"""
SmartSDG Pipeline v2 — local dashboard server.

A zero-dependency (Python stdlib only) web dashboard for running and tracing
the pipeline:

  * launch SDG generation / OD training as subprocesses,
  * stream their console output live to the browser,
  * browse generated datasets with image previews,
  * open result folders in Windows Explorer,
  * read project documentation.

Start it with:  scripts\\launch_dashboard.bat   (or: python dashboard/server.py)
Serves on http://127.0.0.1:8765 — localhost only, single user, one job at a time.
"""

import json
import mimetypes
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# --- Project layout -----------------------------------------------------------
DASH_DIR = Path(__file__).resolve().parent
ROOT = DASH_DIR.parent                                  # SDG_v2
SMARTSDG = ROOT / "SmartSDG"
OD_DIR = ROOT / "OD_training"
AAS_DIR = ROOT / "AAS"
DOCS_DIR = ROOT / "docs"
DATASETS = SMARTSDG / "Datasets" / "yolo_experiments"
RAW_OUTPUT = SMARTSDG / "Datasets" / "unified_render_output"
# Template AAS cloned for per-dataset training configs (must parse without BOM).
TEMPLATE_AAS = AAS_DIR / "CA_Modul-OD_pipeline_v2.0.json"
# Machine-specific paths are resolved portably (env var > pipeline_config.json >
# auto-detect > fallback). See config_paths.py — no source edits needed per machine.
sys.path.insert(0, str(ROOT))
import config_paths
ISAAC_PYTHON = config_paths.resolve_isaac_python()
TRAIN_PYTHON = config_paths.resolve_train_python()

PORT = 8765
IMG_EXTS = (".png", ".jpg", ".jpeg")


def _iter_images(folder):
    """All images (png/jpg) under a folder, recursively."""
    for p in folder.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            yield p

# --- Single-job process manager -------------------------------------------------
# Only one job (generation or training) runs at a time — they share the GPU.
_job_lock = threading.Lock()
_job = None  # dict: name, cmd_str, proc, lines, started, returncode


def _reader(job):
    """Pump subprocess output into the in-memory log buffer."""
    proc = job["proc"]
    for line in iter(proc.stdout.readline, ""):
        job["lines"].append(line.rstrip("\r\n"))
    proc.wait()
    job["returncode"] = proc.returncode
    job["lines"].append(f"--- process exited with code {proc.returncode} ---")
    # After a successful generation, record the dataset in its own AAS file so
    # training can target it directly (reproducibility: dataset + training
    # config travel together).
    if proc.returncode == 0 and job["name"].startswith("SDG"):
        try:
            datasets = list_datasets()
            if datasets:
                aas_name = generate_dataset_aas(datasets[0]["rel"], job.get("meta") or {})
                job["lines"].append(f"--- AAS training config created/updated: AAS\\{aas_name} ---")
        except Exception as e:
            job["lines"].append(f"--- could not create dataset AAS: {e} ---")


def start_job(name, cmd, cwd, meta=None):
    """Start a subprocess job unless one is already running."""
    global _job
    with _job_lock:
        if _job and _job["returncode"] is None:
            return False, f"A job is already running: {_job['name']}"
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        proc = subprocess.Popen(
            cmd, cwd=str(cwd),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", env=env,
        )
        _job = {
            "name": name,
            "cmd_str": subprocess.list2cmdline(cmd),
            "proc": proc,
            "lines": [f"$ {subprocess.list2cmdline(cmd)}", f"(cwd: {cwd})",
                      f"(interpreter on PATH: {__import__('shutil').which('python') or 'none'})", ""],
            "started": time.time(),
            "returncode": None,
            "meta": meta,
        }
        threading.Thread(target=_reader, args=(_job,), daemon=True).start()
        return True, ""


def stop_job():
    with _job_lock:
        if _job and _job["returncode"] is None:
            # Kill the whole tree — Isaac Sim spawns child processes.
            subprocess.run(["taskkill", "/PID", str(_job["proc"].pid), "/T", "/F"],
                           capture_output=True)
            _job["lines"].append("--- job stopped by user ---")
            return True
        return False


# --- AAS helpers ------------------------------------------------------------------

def _aas_get_property(node, id_short):
    """Recursively find the first AAS Property value by idShort."""
    if isinstance(node, dict):
        if node.get("modelType") == "Property" and node.get("idShort") == id_short:
            return node.get("value")
        for v in node.values():
            r = _aas_get_property(v, id_short)
            if r is not None:
                return r
    elif isinstance(node, list):
        for v in node:
            r = _aas_get_property(v, id_short)
            if r is not None:
                return r
    return None


def _aas_set_properties(node, targets):
    """Recursively set AAS Property values by idShort."""
    if isinstance(node, dict):
        if node.get("modelType") == "Property" and node.get("idShort") in targets:
            node["value"] = str(targets[node["idShort"]])
        for v in node.values():
            _aas_set_properties(v, targets)
    elif isinstance(node, list):
        for v in node:
            _aas_set_properties(v, targets)


def generate_dataset_aas(dataset_rel, meta=None):
    """Clone the template AAS for a specific dataset: points base_data_dir at it,
    sets the camera folders, and names the experiment after the dataset.
    Class names stay the full global 9-class list — label files use global IDs.

    Files are organized by generation date: AAS/datasets/YYYY-MM-DD/<name>.json.
    If a config for the same dataset already exists (any date), it is updated
    in place instead of duplicated."""
    d = _safe_path(dataset_rel)
    if not d.is_dir():
        raise ValueError(f"dataset not found: {d}")
    data = json.loads(TEMPLATE_AAS.read_text(encoding="utf-8-sig"))
    cameras = sorted({p.parent.name if p.parent != d else "" for p in _iter_images(d)} - {""}) \
              or ["Replicator", "Replicator_01"]
    _aas_set_properties(data, {
        "base_data_dir": str(d) + "\\",
        "cameras_to_train": ", ".join(cameras),
        "exp_names_to_train": d.name,
    })
    filename = f"CA_Modul-OD_dataset_{d.name}.json"
    existing = list((AAS_DIR / "datasets").rglob(filename)) if (AAS_DIR / "datasets").exists() else []
    if existing:
        out = existing[0]
    else:
        day_dir = AAS_DIR / "datasets" / time.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        out = day_dir / filename
    # NOTE: must be written WITHOUT a UTF-8 BOM — aas_core3 rejects BOMs.
    out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return str(out.relative_to(AAS_DIR))


SDG_CONFIG = SMARTSDG / "src" / "config" / "config.yaml"
AL_DIR = ROOT / "active_learning"
HARVEST_DIR = AL_DIR / "harvest"
_vr = config_paths.resolve_val_real()
VAL_REAL = Path(_vr) if _vr else (ROOT / "data" / "val_real")


def _append_component_to_config(name, usd_filename):
    """Append-only registration of a new component in config.yaml's
    COMPONENT_USD_MAP. Inserted at the END of the map so it receives the next
    free class ID — existing IDs never shift (old datasets stay valid).
    Edits the raw text to preserve the file's comments and formatting."""
    lines = SDG_CONFIG.read_text(encoding="utf-8").splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith("COMPONENT_USD_MAP:"):
            j = i + 1
            last_entry = i
            while j < len(lines):
                if lines[j].startswith(("  ", "\t")) and lines[j].strip() and not lines[j].lstrip().startswith("#"):
                    last_entry = j
                elif lines[j].strip() == "" or not lines[j].startswith(" "):
                    break
                j += 1
            lines.insert(last_entry + 1, f'  {name}: "{usd_filename}"\n')
            SDG_CONFIG.write_text("".join(lines), encoding="utf-8")
            return True
    return False


def component_status(aas_path):
    """Build the component list for the dashboard from a chosen AAS file,
    validated against config.yaml and the USD files on disk.

    - classes come from the AAS 'names' property (user-editable record)
    - a class new to config.yaml whose USD exists on disk is auto-registered
      (append-only -> next free class ID)
    - a class whose USD is missing is reported as unavailable (grayed out)
    """
    import yaml
    cfg = yaml.safe_load(SDG_CONFIG.read_text(encoding="utf-8"))
    usd_map = dict(cfg["COMPONENT_USD_MAP"])
    assets_dir = SMARTSDG / cfg["paths"]["base"] / cfg["paths"]["components_rel"]

    raw = _aas_get_property(json.loads(aas_path.read_text(encoding="utf-8-sig")), "names")
    aas_names = [n.strip() for n in (raw or "").split(",") if n.strip()]

    # Auto-register: AAS class not in config yet, but <name>.usd exists on disk
    usd_files = {f.name.lower(): f.name for f in assets_dir.glob("*.usd")}
    registered = []
    for name in aas_names:
        if name not in usd_map and f"{name.lower()}.usd" in usd_files:
            usd = usd_files[f"{name.lower()}.usd"]
            if _append_component_to_config(name, usd):
                usd_map[name] = usd
                registered.append(f"{name} (class ID {len(usd_map) - 1}, {usd})")

    items = []
    for name, usd in usd_map.items():  # config map order = global class IDs
        exists = (assets_dir / usd).exists()
        items.append({"name": name, "usd": usd, "available": exists,
                      "reason": "" if exists else f"USD file missing in assets: {usd}"})
    for name in aas_names:             # declared in AAS, unresolvable -> grayed
        if name not in usd_map:
            items.append({"name": name, "usd": None, "available": False,
                          "reason": f"declared in AAS but no '{name}.usd' found in assets"})

    warning = ""
    if aas_names and [n for n in aas_names if n in usd_map] != [n for n in usd_map if n in aas_names]:
        warning = (f"Class order differs between this AAS ({aas_names}) and the ID registry "
                   f"({list(usd_map.keys())}). Class IDs always follow the registry order.")
    return {"components": items, "registered": registered, "warning": warning}


def list_dataset_cameras(dataset_rel):
    """Camera-geometry subfolders inside a dataset (e.g. Replicator,
    Replicator_01). Returns [] for flat datasets."""
    d = _safe_path(dataset_rel)
    cams = sorted({p.parent.name for p in _iter_images(d) if p.parent != d})
    return cams


def prepare_al_workspace(dataset_rel, camera=None):
    """Stage a generated dataset for active learning:
       active_learning/data/full-data/{images,labels}/{train,val}
    train pool = images of the chosen SDG dataset (the 'unlabeled' pool),
                 restricted to one camera geometry if `camera` is given so the
                 two camera datasets stay completely separate,
    val        = the real validation images (sim-to-real evaluation).
    Returns (n_train_pool, n_val, class_names, img_ext)."""
    import yaml
    d = _safe_path(dataset_rel)
    if not d.is_dir():
        raise ValueError(f"dataset not found: {d}")
    full = AL_DIR / "data" / "full-data"
    shutil.rmtree(AL_DIR / "data", ignore_errors=True)
    for sub in ["images/train", "labels/train", "images/val", "labels/val"]:
        (full / sub).mkdir(parents=True, exist_ok=True)

    n_train = 0
    img_ext = ".png"
    for png in _iter_images(d):
        # camera filter: only images whose parent folder matches the chosen camera
        if camera and camera != "both" and png.parent.name != camera:
            continue
        txt = png.with_suffix(".txt")
        if txt.exists():
            shutil.copy2(png, full / "images" / "train" / png.name)
            shutil.copy2(txt, full / "labels" / "train" / txt.name)
            img_ext = png.suffix.lower()
            n_train += 1
    n_val = 0
    if VAL_REAL.exists():
        for img in VAL_REAL.glob("*.jpg"):
            txt = img.with_suffix(".txt")
            if txt.exists():
                shutil.copy2(img, full / "images" / "val" / img.name)
                shutil.copy2(txt, full / "labels" / "val" / txt.name)
                n_val += 1

    cfg = yaml.safe_load(SDG_CONFIG.read_text(encoding="utf-8"))
    names = list(cfg["COMPONENT_USD_MAP"].keys())
    (full / "dataset.yaml").write_text(
        yaml.safe_dump({"nc": len(names), "names": names}), encoding="utf-8")
    return n_train, n_val, names, img_ext


def build_al_cmd(p):
    """Stage the dataset, write the AL parameters into the dataset's AAS
    (record mode), and return the run command."""
    rel = p.get("rel", "")
    camera = p.get("camera") or "both"
    n_pool, n_val, names, img_ext = prepare_al_workspace(rel, camera)
    if n_pool == 0:
        raise ValueError(f"no labelled images for camera '{camera}' in this dataset.")

    budget = int(p.get("initial_subset_size", 10)) + int(p.get("al_rounds", 3)) * int(p.get("add_per_round", 5))
    if budget > n_pool:
        raise ValueError(f"budget too large: initial + rounds x add = {budget} images, "
                         f"but the '{camera}' pool only has {n_pool}. Reduce rounds/budget or generate more data.")

    # the dataset AAS is the record — regenerate from the current template
    # (ensures the ActiveLearning section exists), then write the AL params
    aas_rel = generate_dataset_aas(rel)
    aas_path = AAS_DIR / aas_rel
    d = _safe_path(rel)
    # keep each camera geometry's AL run separate so they compare independently
    run_tag = d.name if camera == "both" else f"{d.name}__{camera}"
    data = json.loads(aas_path.read_text(encoding="utf-8-sig"))
    weights = p.get("weights") or "yolov12l.pt"
    weights_path = (OD_DIR / Path(weights).name)
    _aas_set_properties(data, {
        "data_dir": str(AL_DIR / "data"),
        "output_dir": str(AL_DIR / "output" / run_tag),
        "initial_model_path": str(weights_path),
        "dataset_yaml": str(AL_DIR / "data" / "full-data" / "dataset.yaml"),
        "num_classes": len(names),
        "al_strategy": p.get("strategy", "ppal_feat"),
        "al_class_balance": "true" if p.get("class_balance", True) else "false",
        "al_rounds": int(p.get("al_rounds", 3)),
        "initial_subset_size": int(p.get("initial_subset_size", 10)),
        "add_per_round": int(p.get("add_per_round", 5)),
        "epochs_per_round": int(p.get("epochs_per_round", 10)),
        "batch_size": int(p.get("batch_size", 8)),
        "img_size": int(p.get("img_size", 640)),
        "img_extension": img_ext,
    })
    aas_path.write_text(json.dumps(data, indent=1), encoding="utf-8")

    python = TRAIN_PYTHON if Path(TRAIN_PYTHON).exists() else shutil.which("python")
    return [python, "-X", "utf8", "run_al_from_aas.py", "--aas_file", str(aas_path)], n_pool, n_val


# --- Command builders -----------------------------------------------------------

def build_sdg_cmd(p):
    """Translate dashboard form values into the SmartSDG CLI call."""
    cmd = ["cmd", "/c", ISAAC_PYTHON, "main.py",
           "--config", r"src\config\config.yaml",
           "--light_mode", p.get("light", "Medium"),
           "--num_samples", str(int(p.get("samples", 100))),
           "--load_env", p.get("env", "plane"),
           "--width", str(int(p.get("width", 640))),
           "--height", str(int(p.get("height", 480))),
           "--convert_to_yolo"]
    if p.get("mode") == "multi":
        cmd.append("--multi_object")
        comps = p.get("components") or []
        if comps:
            cmd += ["--components", ",".join(comps)]
        if p.get("spawn_radius"):
            cmd += ["--spawn_radius", str(p["spawn_radius"])]
        if p.get("min_separation") is not None:
            cmd += ["--min_separation", str(p["min_separation"])]
    else:
        comps = p.get("components") or ["GlueGun"]
        cmd += ["--component_class", comps[0]]
    if float(p.get("min_visibility", 0.95)) > 0:
        cmd += ["--min_visibility", str(p.get("min_visibility", 0.95))]
    if p.get("drop_truncated", True):
        cmd.append("--drop_truncated")
    if p.get("filter_overlaps", True):
        cmd += ["--filter_overlaps", "--iou_threshold", str(p.get("iou_threshold", 0.7))]
    if p.get("seed"):
        cmd += ["--seed", str(int(p["seed"]))]
    # Renderer hyperparameters (only sent when user overrides the config defaults)
    if p.get("renderer"):
        cmd += ["--renderer", p["renderer"]]
    if p.get("spp"):
        cmd += ["--spp", str(int(p["spp"]))]
    if p.get("max_bounces"):
        cmd += ["--max_bounces", str(int(p["max_bounces"]))]
    if p.get("denoiser") in ("on", "off"):
        cmd += ["--denoiser", p["denoiser"]]
    return cmd


def build_train_cmd(p):
    aas = p.get("aas_file", "")
    safe = (AAS_DIR / aas).resolve()
    if not str(safe).startswith(str(AAS_DIR.resolve())) or not safe.exists():
        raise ValueError(f"AAS file not found: {safe}")
    # Per-run overrides (cameras + training hyperparameters).
    # RECORD MODE: if the chosen AAS is an SDG-generated dataset config
    # (AAS/datasets/...), overrides are written INTO that file so it always
    # documents the parameters of the latest training. Hand-maintained
    # pipeline templates at the AAS root are never modified — for those the
    # overrides go to a temp copy instead.
    overrides = {}
    if p.get("cameras"):
        overrides["cameras_to_train"] = ", ".join(p["cameras"])
    for key in ("epochs", "img_size", "batch_size", "save_period"):
        if p.get(key) not in (None, "", 0):
            overrides[key] = int(p[key])
    if p.get("weights"):
        overrides["weights"] = p["weights"]
    if overrides:
        data = json.loads(safe.read_text(encoding="utf-8-sig"))
        _aas_set_properties(data, overrides)
        is_dataset_aas = "datasets" in safe.relative_to(AAS_DIR).parts
        if is_dataset_aas:
            # persist: the dataset AAS is the canonical training record
            safe.write_text(json.dumps(data, indent=1), encoding="utf-8")
        else:
            tmp = Path(tempfile.gettempdir()) / f"dashboard_run_{safe.stem}.json"
            tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
            safe = tmp
    python = TRAIN_PYTHON if Path(TRAIN_PYTHON).exists() else shutil.which("python")
    if not python:
        raise ValueError("No training interpreter found — set TRAIN_PYTHON in dashboard/server.py "
                         "or launch the dashboard from an activated env (e.g. conda activate yolov12).")
    return [python, "main.py", "--mode", "train", "--aas_file", str(safe)]


# --- Filesystem helpers -----------------------------------------------------------

def _safe_path(rel):
    """Resolve a client-supplied relative path, refusing escapes outside ROOT."""
    p = (ROOT / rel).resolve()
    if not str(p).startswith(str(ROOT)):
        raise ValueError("path outside project root")
    return p


def list_datasets():
    out = []
    if DATASETS.exists():
        for d in sorted(DATASETS.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if d.is_dir():
                imgs = list(_iter_images(d))
                out.append({
                    "name": d.name,
                    "images": len(imgs),
                    "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(d.stat().st_mtime)),
                    "rel": str(d.relative_to(ROOT)),
                })
    return out


def list_results():
    """Find actual run folders at any depth (a run = a dir containing args.yaml,
    results.csv or a final_run_config.yaml), newest first."""
    found = {}

    def add(d, label):
        rel = str(d.relative_to(ROOT))
        if rel not in found:
            has_weights = any((d / "weights").glob("*.pt")) if (d / "weights").is_dir() else False
            found[rel] = {
                "name": label,
                "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(d.stat().st_mtime)),
                "mtime": d.stat().st_mtime,
                "weights": has_weights,
                "rel": rel,
            }

    od_results = OD_DIR / "results"
    if od_results.exists():
        for marker in ["args.yaml", "results.csv", "final_run_config.yaml"]:
            for m in od_results.rglob(marker):
                add(m.parent, f"OD_training/results/{m.parent.relative_to(od_results)}")
    for base in [OD_DIR / "runs", OD_DIR / "yolo_runs_multi_mix",
                 ROOT / "active_learning" / "output", ROOT / "continual_learning" / "results"]:
        if base.exists():
            for d in base.iterdir():
                if d.is_dir():
                    add(d, f"{base.parent.name}/{base.name}/{d.name}")

    out = sorted(found.values(), key=lambda x: x["mtime"], reverse=True)
    for o in out:
        o.pop("mtime")
    return out


def dataset_preview(rel, limit=12):
    d = _safe_path(rel)
    return [str(p.relative_to(ROOT)) for p in sorted(_iter_images(d))[:limit]]


# --- HTTP handler -----------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):

    def log_message(self, *a):  # silence default request logging
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, ctype=None):
        try:
            data = Path(path).read_bytes()
        except OSError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype or mimetypes.guess_type(str(path))[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # ---- GET ----
    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(url.query)
        route = url.path

        if route in ("/", "/index.html"):
            return self._file(DASH_DIR / "index.html", "text/html; charset=utf-8")

        if route == "/api/status":
            j = _job
            return self._json({
                "running": bool(j and j["returncode"] is None),
                "name": j["name"] if j else None,
                "returncode": j["returncode"] if j else None,
                "elapsed": round(time.time() - j["started"]) if j else 0,
            })

        if route == "/api/logs":
            since = int(q.get("since", ["0"])[0])
            j = _job
            if not j:
                return self._json({"lines": [], "next": 0, "done": True, "returncode": None})
            lines = j["lines"][since:]
            return self._json({"lines": lines, "next": since + len(lines),
                               "done": j["returncode"] is not None,
                               "returncode": j["returncode"]})

        if route == "/api/datasets":
            return self._json(list_datasets())

        if route == "/api/results":
            return self._json(list_results())

        if route == "/api/aas":
            # Top-level pipeline configs + dated per-dataset configs, as paths
            # relative to AAS/ (newest dataset configs first).
            top = sorted(f.name for f in AAS_DIR.glob("*.json"))
            ds = sorted((AAS_DIR / "datasets").rglob("*.json"),
                        key=lambda f: f.stat().st_mtime, reverse=True) if (AAS_DIR / "datasets").exists() else []
            files = [str(f.relative_to(AAS_DIR)).replace("\\", "/") for f in ds] + top
            return self._json(files)

        if route == "/api/components":
            # Component list driven by an AAS file chosen by the user
            # (?aas=<relpath>, default the v2.0 template) and validated against
            # the assets folder. NEW classes declared in the AAS whose USD file
            # exists on disk are AUTO-REGISTERED into config.yaml
            # COMPONENT_USD_MAP (append-only: new components get the next free
            # class ID, existing IDs never change — that keeps every previously
            # generated dataset valid).
            try:
                aas_rel = q.get("aas", [str(TEMPLATE_AAS.relative_to(AAS_DIR))])[0]
                aas_path = (AAS_DIR / aas_rel).resolve()
                if not str(aas_path).startswith(str(AAS_DIR.resolve())) or not aas_path.exists():
                    raise ValueError(f"AAS file not found: {aas_rel}")
                return self._json(component_status(aas_path))
            except Exception as e:
                return self._json({"error": str(e)}, 500)

        if route == "/api/weights":
            # Pretrained/base weights available in OD_training (for the dropdown)
            pts = sorted(f.name for f in OD_DIR.glob("*.pt"))
            return self._json(pts)

        if route == "/api/harvests":
            # List harvest folders (inference-time AL candidates) with counts.
            out = []
            if HARVEST_DIR.exists():
                for d in sorted(HARVEST_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
                    man = d / "harvest_manifest.json"
                    if man.exists():
                        m = json.loads(man.read_text())
                        remaining = len(list((d / "images").glob("*.*"))) if (d / "images").exists() else 0
                        out.append({"name": d.name, "kept": m.get("kept", 0),
                                    "scored": m.get("scored", 0), "remaining": remaining,
                                    "num_classes": m.get("num_classes", 9)})
            return self._json(out)

        if route == "/api/harvest_items":
            d = HARVEST_DIR / Path(q["name"][0]).name
            man = d / "harvest_manifest.json"
            if not man.exists():
                return self._json([])
            items = json.loads(man.read_text()).get("items", [])
            res = []
            for it in items:
                img = d / "images" / it["image"]
                if img.exists():  # not yet accepted/rejected
                    res.append({**it,
                                "img": str(img.relative_to(ROOT)),
                                "label": str((d / "labels" / (img.stem + ".txt")).relative_to(ROOT))})
            return self._json(res)

        if route == "/api/listdir":
            # Server-side folder browser for picking a real-images directory.
            import string
            raw = q.get("path", [""])[0]
            try:
                if not raw:
                    # top level: list drive letters on Windows
                    drives = [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
                    return self._json({"path": "", "parent": None, "dirs": drives,
                                       "images": 0})
                p = Path(raw)
                if not p.is_dir():
                    return self._json({"error": "not a directory"}, 400)
                dirs, n_imgs = [], 0
                for entry in sorted(p.iterdir(), key=lambda x: x.name.lower()):
                    try:
                        if entry.is_dir():
                            dirs.append(str(entry))
                        elif entry.suffix.lower() in IMG_EXTS:
                            n_imgs += 1
                    except OSError:
                        continue
                return self._json({"path": str(p), "parent": str(p.parent) if p.parent != p else None,
                                   "dirs": dirs, "images": n_imgs})
            except Exception as e:
                return self._json({"error": str(e)}, 400)

        if route == "/api/cameras":
            # Camera-geometry subfolders of a dataset (for separate AL/training).
            try:
                return self._json(list_dataset_cameras(q["rel"][0]))
            except Exception as e:
                return self._json({"error": str(e)}, 400)

        if route == "/api/al_results":
            # Per-round learning curves from every AL run (al_summary.csv),
            # grouped by dataset -> the "what did AL gain me" overview.
            import csv
            out = []
            al_output = AL_DIR / "output"
            if al_output.exists():
                for summary in al_output.rglob("al_summary.csv"):
                    rows = []
                    with open(summary, newline="") as f:
                        for r in csv.DictReader(f):
                            rows.append({
                                "round": int(r["round"]),
                                "strategy": r.get("strategy", "?"),
                                "images": int(r["train_images"]),
                                "map50": float(r["mAP50"]) if r["mAP50"] == r["mAP50"] else None,
                                "map5095": float(r["mAP50_95"]) if r["mAP50_95"] == r["mAP50_95"] else None,
                            })
                    if rows:
                        # gain = best mAP50 over rounds minus round-0 mAP50
                        valid = [r["map50"] for r in rows if r["map50"] is not None]
                        gain = (max(valid) - rows[0]["map50"]) if valid and rows[0]["map50"] is not None else None
                        out.append({"dataset": summary.parent.name,
                                    "strategy": rows[-1]["strategy"],
                                    "rows": rows, "gain": gain,
                                    "modified": time.strftime("%Y-%m-%d %H:%M",
                                                              time.localtime(summary.stat().st_mtime))})
            out.sort(key=lambda x: x["modified"], reverse=True)
            return self._json(out)

        if route == "/api/preview":
            try:
                return self._json(dataset_preview(q["rel"][0]))
            except Exception as e:
                return self._json({"error": str(e)}, 400)

        if route == "/api/doc":
            name = Path(q.get("f", ["README.md"])[0]).name
            for cand in [ROOT / name, DOCS_DIR / name]:
                if cand.exists():
                    return self._file(cand, "text/plain; charset=utf-8")
            self.send_error(404)
            return

        if route == "/api/label":
            # YOLO label file for an image path (same basename, .txt)
            try:
                p = _safe_path(q["p"][0]).with_suffix(".txt")
                txt = p.read_text(encoding="utf-8") if p.exists() else ""
                body = txt.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            except Exception:
                self.send_error(404)
                return

        if route == "/img":
            try:
                p = _safe_path(q["p"][0])
                return self._file(p)
            except Exception:
                self.send_error(404)
                return

        self.send_error(404)

    # ---- POST ----
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._json({"ok": False, "error": "bad JSON"}, 400)
        route = urllib.parse.urlparse(self.path).path

        if route == "/api/run_sdg":
            try:
                cmd = build_sdg_cmd(payload)
            except Exception as e:
                return self._json({"ok": False, "error": str(e)}, 400)
            ok, err = start_job("SDG generation", cmd, SMARTSDG, meta=payload)
            return self._json({"ok": ok, "error": err})

        if route == "/api/delete_dataset":
            # Deletes the converted dataset AND its raw render folder. Only
            # paths inside the two dataset directories are accepted.
            try:
                p = _safe_path(payload.get("rel", ""))
                if not (str(p.parent) == str(DATASETS) and p.is_dir()):
                    raise ValueError("only datasets inside yolo_experiments can be deleted")
                shutil.rmtree(p)
                removed = [str(p.relative_to(ROOT))]
                raw = RAW_OUTPUT / p.name
                if raw.is_dir():
                    shutil.rmtree(raw)
                    removed.append(str(raw.relative_to(ROOT)))
                for aas in list(AAS_DIR.glob(f"CA_Modul-OD_dataset_{p.name}.json")) + \
                           (list((AAS_DIR / "datasets").rglob(f"CA_Modul-OD_dataset_{p.name}.json"))
                            if (AAS_DIR / "datasets").exists() else []):
                    aas.unlink()
                    removed.append(str(aas.relative_to(ROOT)))
                return self._json({"ok": True, "removed": removed})
            except Exception as e:
                return self._json({"ok": False, "error": str(e)}, 400)

        if route == "/api/make_aas":
            try:
                name = generate_dataset_aas(payload.get("rel", ""))
                return self._json({"ok": True, "aas": name})
            except Exception as e:
                return self._json({"ok": False, "error": str(e)}, 400)

        if route == "/api/run_train":
            try:
                cmd = build_train_cmd(payload)
            except Exception as e:
                return self._json({"ok": False, "error": str(e)}, 400)
            ok, err = start_job("OD training", cmd, OD_DIR)
            return self._json({"ok": ok, "error": err})

        if route == "/api/run_harvest":
            model = payload.get("model", "")
            mpath = (OD_DIR / Path(model).name) if model and not Path(model).is_absolute() else Path(model)
            if not Path(mpath).exists():
                # also look inside OD results for a best.pt by run name
                cands = list(OD_DIR.rglob(Path(model).name)) if model else []
                if cands:
                    mpath = cands[0]
            source = payload.get("source", "")
            if not Path(source).is_dir():
                return self._json({"ok": False, "error": f"source folder not found: {source}"}, 400)
            if not Path(mpath).exists():
                return self._json({"ok": False, "error": f"model not found: {model}"}, 400)
            name = payload.get("out") or Path(source).name
            python = TRAIN_PYTHON if Path(TRAIN_PYTHON).exists() else shutil.which("python")
            cmd = [python, "-X", "utf8", "harvest_from_inference.py",
                   "--model", str(mpath), "--source", str(source), "--out", name,
                   "--top", str(int(payload.get("top", 50))),
                   "--conf", str(payload.get("conf", 0.10)),
                   "--num_classes", str(int(payload.get("num_classes", 9)))]
            ok, err = start_job(f"AL harvest ({name})", cmd, AL_DIR)
            return self._json({"ok": ok, "error": err})

        if route == "/api/harvest_review":
            # accept -> move image + verified pre-label into a REAL_<name>
            # dataset (usable for training/AL); reject -> delete the candidate.
            try:
                d = HARVEST_DIR / Path(payload["name"]).name
                img_name = Path(payload["image"]).name
                img = d / "images" / img_name
                lbl = d / "labels" / (img.stem + ".txt")
                if payload.get("action") == "accept":
                    ds = DATASETS / f"REAL_{d.name}" / "Replicator"
                    (ds.parent / "Replicator").mkdir(parents=True, exist_ok=True)
                    shutil.move(str(img), ds / img_name)
                    if lbl.exists():
                        shutil.move(str(lbl), ds / (img.stem + ".txt"))
                    return self._json({"ok": True, "dataset": f"REAL_{d.name}"})
                else:
                    img.unlink(missing_ok=True)
                    lbl.unlink(missing_ok=True)
                    return self._json({"ok": True})
            except Exception as e:
                return self._json({"ok": False, "error": str(e)}, 400)

        if route == "/api/save_label":
            # Persist edited YOLO label text. Allowed only for harvest
            # candidates and dataset label files (never outside the project).
            try:
                p = _safe_path(payload["label"])
                parents = p.resolve().parents
                allowed = (HARVEST_DIR.resolve() in parents) or (DATASETS.resolve() in parents)
                if not allowed:
                    raise ValueError("can only edit harvest or dataset labels")
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(payload.get("text", ""), encoding="utf-8")
                return self._json({"ok": True})
            except Exception as e:
                return self._json({"ok": False, "error": str(e)}, 400)

        if route == "/api/run_al":
            try:
                cmd, n_pool, n_val = build_al_cmd(payload)
            except Exception as e:
                return self._json({"ok": False, "error": str(e)}, 400)
            ok, err = start_job(f"Active Learning ({payload.get('strategy', 'ppal_feat')})", cmd, AL_DIR)
            return self._json({"ok": ok, "error": err, "pool": n_pool, "val": n_val})

        if route == "/api/run_test":
            # Pipeline self-test as a regular job (live console output).
            python = TRAIN_PYTHON if Path(TRAIN_PYTHON).exists() else shutil.which("python")
            cmd = [python, "-X", "utf8", str(ROOT / "scripts" / "test_pipeline.py")]
            name = "Pipeline self-test (static)"
            if payload.get("full"):
                cmd.append("--full")
                name = "Pipeline self-test (FULL: SDG + training)"
            ok, err = start_job(name, cmd, ROOT)
            return self._json({"ok": ok, "error": err})

        if route == "/api/stop":
            return self._json({"ok": stop_job()})

        if route == "/api/open":
            try:
                p = _safe_path(payload.get("rel", "."))
                os.startfile(str(p))  # opens folder in Explorer / file in default app
                return self._json({"ok": True})
            except Exception as e:
                return self._json({"ok": False, "error": str(e)}, 400)

        self.send_error(404)


def main():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"SmartSDG dashboard running at {url}  (Ctrl+C to stop)")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")


if __name__ == "__main__":
    main()
