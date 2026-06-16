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

# Master cross-stage defaults (single source). Re-read on each request so edits
# to pipeline.yaml take effect without restarting the server.
PIPELINE_YAML = ROOT / "pipeline.yaml"


def load_defaults():
    try:
        import yaml
        return yaml.safe_load(PIPELINE_YAML.read_text(encoding="utf-8")) or {}
    except Exception as e:
        print(f"[defaults] WARNING: could not read pipeline.yaml: {e}", file=sys.stderr)
        return {}


# --- Docker execution (opt-in) for the heavy GPU Python stages -----------------
# Training / AL / CL can run inside a container so the machine only needs Docker
# (no conda env). SDG (Isaac Sim), harvest and live stay native. Toggle + image
# name live in pipeline_config.json (per-machine, git-ignored).
DOCKER_IMAGE = "smartsdg"
MACHINE_CFG = ROOT / "pipeline_config.json"


def _machine_cfg():
    try:
        return json.loads(MACHINE_CFG.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def _set_machine_cfg(key, value):
    cfg = _machine_cfg()
    cfg[key] = value
    MACHINE_CFG.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def docker_enabled():
    return bool(_machine_cfg().get("docker_enabled"))


def docker_status():
    """{installed, daemon, image} — whether Docker is usable + the image built."""
    exe = shutil.which("docker")
    st = {"installed": bool(exe), "daemon": False, "image": False,
          "enabled": docker_enabled(), "image_name": DOCKER_IMAGE}
    if not exe:
        return st
    try:
        r = subprocess.run([exe, "info", "--format", "{{.ServerVersion}}"],
                           capture_output=True, text=True, timeout=10)
        st["daemon"] = (r.returncode == 0)
    except Exception:
        pass
    if st["daemon"]:
        try:
            r = subprocess.run([exe, "images", "-q", DOCKER_IMAGE],
                               capture_output=True, text=True, timeout=10)
            st["image"] = bool(r.stdout.strip())
        except Exception:
            pass
    return st


def _to_container_path(arg):
    """Translate a host absolute path under the project (or VAL_REAL) to its
    bind-mounted container path; leave other args unchanged."""
    rs = str(ROOT)
    if isinstance(arg, str) and arg.startswith(rs):
        return "/workspace" + arg[len(rs):].replace("\\", "/")
    vr = str(VAL_REAL)
    if isinstance(arg, str) and VAL_REAL.exists() and arg.startswith(vr):
        return "/valdata" + arg[len(vr):].replace("\\", "/")
    return arg


def dockerize(cmd, cwd):
    """Wrap a native python command into a `docker run` invocation: mount the
    project at /workspace (and VAL_REAL at /valdata), set the workdir, translate
    path args, and use the container's python. Returns (docker_cmd, cwd=ROOT)."""
    inner = []
    for a in cmd:
        if a in (str(TRAIN_PYTHON), "python", "cmd", "/c"):
            continue  # interpreter / shell wrapper handled by the image
        inner.append(_to_container_path(a))
    workdir = "/workspace/" + str(Path(cwd).resolve().relative_to(ROOT)).replace("\\", "/")
    run = ["docker", "run", "--rm", "--gpus", "all",
           "--shm-size=2g",
           "-e", f"SMARTSDG_WIN_ROOT={ROOT}",
           "-v", f"{ROOT}:/workspace"]
    if VAL_REAL.exists():
        run += ["-v", f"{VAL_REAL}:/valdata",
                "-e", f"SMARTSDG_WIN_VAL_REAL={VAL_REAL}"]
    run += ["-w", workdir, DOCKER_IMAGE, "python"] + inner
    return run, ROOT


def _iter_images(folder):
    """All images (png/jpg) under a folder, recursively."""
    for p in folder.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            yield p

# --- Single-job process manager -------------------------------------------------
# Only one job (generation or training) runs at a time — they share the GPU.
_job_lock = threading.Lock()
_job = None  # dict: name, cmd_str, proc, lines, started, returncode

# Live MJPEG stream: only one camera/video stream at a time. Starting a new
# stream bumps the token so the previous loop exits and releases its capture
# before the new one opens — concurrent camera opens hard-crash on Windows.
_stream_lock = threading.Lock()
_stream_token = 0
_stream_active = False


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
                job.setdefault("meta", {})["dataset_rel"] = datasets[0]["rel"]
        except Exception as e:
            job["lines"].append(f"--- could not create dataset AAS: {e} ---")
    # Provenance: record generation params (A) + run outcome (B) into the AAS.
    try:
        msgs = record_to_aas(job)
        for m in msgs:
            job["lines"].append(m)
    except Exception as e:
        job["lines"].append(f"--- AAS provenance write skipped: {e} ---")
    # D) inference/harvest events -> pipeline log
    try:
        if proc.returncode == 0 and job["name"].startswith("AL harvest"):
            man = _newest(HARVEST_DIR, "harvest_manifest.json")
            info = json.loads(man.read_text()) if man else {}
            log_pipeline_event("inference", "harvest_completed",
                               harvest=(man.parent.name if man else None),
                               model=info.get("model"), source=info.get("source"),
                               scored=info.get("scored"), kept=info.get("kept"))
            job["lines"].append("--- pipeline log: harvest event recorded ---")
    except Exception as e:
        job["lines"].append(f"--- pipeline log (harvest) skipped: {e} ---")


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


# --- AAS section read/write (provenance: generation params + run outcomes) -----

def _aas_find_node(node, id_short):
    """Return the first SubmodelElement dict whose idShort matches."""
    if isinstance(node, dict):
        if node.get("idShort") == id_short:
            return node
        for v in node.values():
            r = _aas_find_node(v, id_short)
            if r is not None:
                return r
    elif isinstance(node, list):
        for v in node:
            r = _aas_find_node(v, id_short)
            if r is not None:
                return r
    return None


def _aas_prop(id_short, value):
    return {"idShort": id_short, "valueType": "xs:string",
            "value": "" if value is None else str(value), "modelType": "Property"}


def _aas_parent_collection(data):
    """The container we attach provenance sections to: prefer ODOptimisation,
    else the first submodel's submodelElements, else the root list."""
    od = _aas_find_node(data, "ODOptimisation")
    if od is not None:
        od.setdefault("value", [])
        return od["value"]
    sms = data.get("submodels") if isinstance(data, dict) else None
    if sms and isinstance(sms, list) and sms[0].get("submodelElements") is not None:
        return sms[0]["submodelElements"]
    return None


def _aas_upsert_section(data, section_id, props):
    """Create/update a SubmodelElementList `section_id` of flat properties."""
    container = _aas_parent_collection(data)
    if container is None:
        return False
    sec = next((e for e in container if isinstance(e, dict) and e.get("idShort") == section_id), None)
    if sec is None:
        sec = {"idShort": section_id, "typeValueListElement": "SubmodelElement",
               "modelType": "SubmodelElementList", "value": []}
        container.append(sec)
    sec.setdefault("value", [])
    existing = {e.get("idShort"): e for e in sec["value"] if isinstance(e, dict)}
    for k, v in props.items():
        if k in existing:
            existing[k]["value"] = "" if v is None else str(v)
        else:
            sec["value"].append(_aas_prop(k, v))
    return True


def _aas_append_run(data, entry: dict):
    """Append one run-outcome record (a SubmodelElementCollection) to the
    append-only 'RunHistory' list — every update is kept, not overwritten."""
    container = _aas_parent_collection(data)
    if container is None:
        return False
    hist = next((e for e in container if isinstance(e, dict) and e.get("idShort") == "RunHistory"), None)
    if hist is None:
        hist = {"idShort": "RunHistory", "typeValueListElement": "SubmodelElement",
                "modelType": "SubmodelElementList", "value": []}
        container.append(hist)
    hist.setdefault("value", [])
    rid = "run_" + time.strftime("%Y%m%dT%H%M%S")
    hist["value"].append({
        "idShort": rid, "modelType": "SubmodelElementCollection",
        "value": [_aas_prop(k, v) for k, v in entry.items()],
    })
    return True


def _aas_read_record(aas_path):
    """Read the provenance sections from a dataset AAS into a plain dict for the
    dashboard: GenerationParameters, latest-stage Results, and RunHistory."""
    try:
        data = json.loads(Path(aas_path).read_text(encoding="utf-8-sig"))
    except Exception:
        return {}

    def section(id_short):
        node = _aas_find_node(data, id_short)
        if not node or not node.get("value"):
            return {}
        return {e.get("idShort"): e.get("value") for e in node["value"]
                if isinstance(e, dict) and e.get("modelType") == "Property"}

    hist = _aas_find_node(data, "RunHistory")
    runs = []
    if hist and hist.get("value"):
        for coll in hist["value"]:
            if isinstance(coll, dict) and coll.get("value"):
                runs.append({e.get("idShort"): e.get("value") for e in coll["value"]
                             if isinstance(e, dict)})
    return {
        "generation": section("GenerationParameters"),
        "results": section("Results"),
        "active_learning": section("ActiveLearning"),
        "continual_learning": section("ContinualLearning"),
        "runs": runs,
    }


def _find_dataset_aas_path(rel_or_name):
    """Locate the per-dataset AAS file for a dataset rel/name."""
    if not rel_or_name:
        return None
    name = Path(rel_or_name).name
    base = AAS_DIR / "datasets"
    if base.exists():
        cands = list(base.rglob(f"CA_Modul-OD_dataset_{name}.json"))
        if cands:
            return cands[0]
    return None


def _read_yolo_metrics(csv_path):
    """Final mAP50 / mAP50-95 from a YOLO results.csv (training / CL)."""
    import csv
    try:
        rows = list(csv.DictReader(open(csv_path)))
        if not rows:
            return {}
        last = rows[-1]
        def g(key):
            for c in last:
                if c.strip() == key:
                    try:
                        return round(float(last[c]), 4)
                    except ValueError:
                        return None
            return None
        return {"map50": g("metrics/mAP50(B)"), "map50_95": g("metrics/mAP50-95(B)"),
                "epochs": len(rows)}
    except Exception:
        return {}


def _read_al_metrics(csv_path):
    """Best mAP50 + gain from an al_summary.csv."""
    import csv
    try:
        rows = list(csv.DictReader(open(csv_path)))
        vals = [float(r["mAP50"]) for r in rows if r.get("mAP50") not in (None, "", "nan")]
        if not vals:
            return {"rounds": len(rows)}
        gain = round(max(vals) - vals[0], 4)
        return {"map50": round(max(vals), 4), "gain": gain, "rounds": len(rows) - 1}
    except Exception:
        return {}


def record_to_aas(job):
    """Write provenance into the dataset AAS at job completion:
      A) GenerationParameters for SDG runs,
      B) a Results summary + an append-only RunHistory entry for every stage.
    Returns log messages."""
    meta = job.get("meta") or {}
    stage = meta.get("stage")
    if not stage:
        return []
    msgs = []

    # locate the dataset AAS to write into
    aas_path = None
    if stage == "SDG":
        aas_path = _find_dataset_aas_path(meta.get("dataset_rel"))
    elif stage == "training":
        af = meta.get("aas_file", "")
        cand = (AAS_DIR / af)
        aas_path = cand if cand.exists() else _find_dataset_aas_path(af)
    elif stage == "AL":
        aas_path = _find_dataset_aas_path(meta.get("rel"))
    elif stage == "CL":
        aas_path = _find_dataset_aas_path(meta.get("new_data_rel"))
    if not aas_path or not Path(aas_path).exists():
        return msgs

    data = json.loads(Path(aas_path).read_text(encoding="utf-8-sig"))
    ok = job["returncode"] == 0
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    duration = int(time.time() - job.get("started", time.time()))
    entry = {"stage": stage, "status": "success" if ok else f"failed({job['returncode']})",
             "finished_at": now, "duration_s": duration}

    # A) generation parameters
    if stage == "SDG":
        keys = ["mode", "light", "samples", "env", "width", "height", "min_visibility",
                "spawn_radius", "min_separation", "iou_threshold", "drop_truncated",
                "filter_overlaps", "seed", "renderer", "spp", "max_bounces", "denoiser"]
        gen = {k: meta.get(k) for k in keys if meta.get(k) is not None}
        if meta.get("components"):
            gen["components"] = ",".join(meta["components"])
        gen["generated_at"] = now
        try:
            d = ROOT / meta.get("dataset_rel", "")
            gen["final_images"] = sum(1 for _ in _iter_images(d)) if d.is_dir() else None
        except Exception:
            pass
        _aas_upsert_section(data, "GenerationParameters", gen)
        if gen.get("final_images") is not None:
            entry["final_images"] = gen["final_images"]

    # B) outcomes / metrics
    if ok:
        metrics = {}
        if stage in ("training", "CL"):
            csvp = _newest_results_csv(stage, aas_path)
            if csvp:
                metrics = _read_yolo_metrics(csvp)
                best = csvp.parent / "weights" / "best.pt"
                if best.exists():
                    metrics["model_path"] = str(best)
        elif stage == "AL":
            csvp = _newest(AL_DIR / "output", "al_summary.csv")
            if csvp:
                metrics = _read_al_metrics(csvp)
        metrics = {k: v for k, v in metrics.items() if v is not None}
        if metrics:
            entry.update(metrics)
            _aas_upsert_section(data, "Results", {"stage": stage, "updated_at": now, **metrics})

    _aas_append_run(data, entry)
    Path(aas_path).write_text(json.dumps(data, indent=1), encoding="utf-8")
    msgs.append(f"--- AAS provenance updated: {Path(aas_path).name} ({stage}) ---")

    # C) also append to the factory-wide pipeline event log
    dataset = Path(aas_path).stem.replace("CA_Modul-OD_dataset_", "")
    log_pipeline_event(stage, "run_completed", dataset=dataset, status=entry["status"],
                       duration_s=entry.get("duration_s"), map50=entry.get("map50"),
                       gain=entry.get("gain"), model_path=entry.get("model_path"),
                       aas_file=str(Path(aas_path).relative_to(ROOT)))
    return msgs


def _newest(root, pattern):
    if not Path(root).exists():
        return None
    files = list(Path(root).rglob(pattern))
    return max(files, key=lambda f: f.stat().st_mtime) if files else None


def _newest_results_csv(stage, aas_path):
    if stage == "CL":
        return _newest(CL_DIR / "results" / "cl_runs", "results.csv")
    return _newest(OD_DIR / "results", "results.csv")


# =============================================================================
#  C) Factory-wide pipeline event log  +  E) as a valid IEC 63278 AAS Environment
# =============================================================================
# A single AAS Environment (AAS/pipeline_history.json) with one Submodel,
# 'PipelineHistory', holding an append-only 'Events' list. Every action in the
# pipeline (generation, training, AL, CL, inference, harvest, session start) is
# appended as a SubmodelElementCollection — the modular, trackable factory record
# other modules can consume.

PIPELINE_LOG = AAS_DIR / "pipeline_history.json"
SUBMODEL_ID = "urn:smartsdg:submodel:pipeline-history"
# One session per server start; the user can also start a named session/task.
SESSION = {"id": "session_" + time.strftime("%Y%m%dT%H%M%S"), "name": "", "task": ""}


def _pipeline_log_skeleton():
    """A minimal but valid AAS Environment (parses with aas_core3)."""
    return {
        "assetAdministrationShells": [{
            "idShort": "CA_Modul",
            "id": "urn:smartsdg:aas:ca-modul-pipeline",
            "assetInformation": {"assetKind": "Instance",
                                 "globalAssetId": "urn:smartsdg:asset:ca-modul"},
            "submodels": [{"type": "ModelReference",
                           "keys": [{"type": "Submodel", "value": SUBMODEL_ID}]}],
            "modelType": "AssetAdministrationShell",
        }],
        "submodels": [{
            "id": SUBMODEL_ID,
            "idShort": "PipelineHistory",
            "kind": "Instance",
            "submodelElements": [{
                "idShort": "Events",
                "typeValueListElement": "SubmodelElement",
                "modelType": "SubmodelElementList",
                "value": [],
            }],
            "modelType": "Submodel",
        }],
    }


def _load_pipeline_log():
    if PIPELINE_LOG.exists():
        try:
            return json.loads(PIPELINE_LOG.read_text(encoding="utf-8-sig"))
        except Exception:
            pass
    return _pipeline_log_skeleton()


def log_pipeline_event(stage, action, **props):
    """Append one event to the factory-wide pipeline history AAS."""
    try:
        data = _load_pipeline_log()
        events = _aas_find_node(data, "Events")
        if events is None:
            data = _pipeline_log_skeleton()
            events = _aas_find_node(data, "Events")
        events.setdefault("value", [])
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "session_id": SESSION["id"],
            "session_name": SESSION.get("name", ""),
            "task": SESSION.get("task", ""),
            "stage": stage,
            "action": action,
        }
        entry.update({k: v for k, v in props.items() if v is not None})
        eid = "event_" + time.strftime("%Y%m%dT%H%M%S") + f"_{len(events['value'])}"
        events["value"].append({
            "idShort": eid, "modelType": "SubmodelElementCollection",
            "value": [_aas_prop(k, v) for k, v in entry.items()],
        })
        PIPELINE_LOG.write_text(json.dumps(data, indent=1), encoding="utf-8")
    except Exception as e:
        print(f"[pipeline_log] WARNING: {e}", file=sys.stderr)


def read_pipeline_events():
    data = _load_pipeline_log()
    ev = _aas_find_node(data, "Events")
    out = []
    if ev and ev.get("value"):
        for coll in ev["value"]:
            if isinstance(coll, dict) and coll.get("value"):
                out.append({e.get("idShort"): e.get("value") for e in coll["value"]
                            if isinstance(e, dict)})
    return out


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


CL_DIR = ROOT / "continual_learning"


def _registry_classes():
    import yaml
    cfg = yaml.safe_load(SDG_CONFIG.read_text(encoding="utf-8"))
    return list(cfg["COMPONENT_USD_MAP"].keys())


def _classes_from_dataset_name(name, registry):
    """Parse the component classes encoded in an SDG dataset folder name, e.g.
    'SINGLE_BlackCabin_Medium_plane' -> ['BlackCabin'],
    'MULTI_GlueGun-BLid-BTrailer_Medium_withenv' -> ['GlueGun','BLid','BTrailer'],
    'MULTI_ALL_CLASSES_Medium_plane' -> the full registry. Unknown -> []."""
    LIGHT = {"Low", "Medium", "High"}
    parts = name.split("_")
    if not parts or parts[0] not in ("SINGLE", "MULTI", "REAL"):
        return []
    li = next((i for i, p in enumerate(parts) if p in LIGHT), None)
    if li is None:
        return []
    seg = "_".join(parts[1:li])
    if seg in ("ALL_CLASSES", "ALL"):
        return list(registry)
    return [c for c in seg.split("-") if c in registry]


def _model_trained_classes(model_arg, registry):
    """Best-effort list of classes a model was trained on:
       1. a sidecar trained_classes.json next to the weights (authoritative), else
       2. parse the dataset folder name from the model path, else
       3. [] (nothing grayed -> all selectable, the safe default)."""
    if not model_arg:
        return []
    mpath = (OD_DIR / Path(model_arg).name) if not Path(model_arg).is_absolute() else Path(model_arg)
    if not mpath.exists():
        cands = list(OD_DIR.rglob(Path(model_arg).name))
        mpath = cands[0] if cands else mpath
    # 1. sidecar (written by CL / training when known)
    for cand in [mpath.parent / "trained_classes.json", mpath.parent.parent / "trained_classes.json"]:
        if cand.exists():
            try:
                names = json.loads(cand.read_text())
                return [c for c in names if c in registry]
            except Exception:
                pass
    # 2. dataset name in the path
    for part in Path(model_arg).parts:
        classes = _classes_from_dataset_name(part, registry)
        if classes:
            return classes
    return []


def _stage_cl_split(dataset_rel, keep_global_ids, remap, out_dir, camera=None):
    """Copy a dataset's image+label pairs into out_dir, keeping only labels whose
    GLOBAL class id is in keep_global_ids. If remap is given (dict global->local),
    rewrite the class ids (used to make new-class data 0-based for the CL offset
    logic). Returns the number of images written."""
    d = _safe_path(dataset_rel)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for img in _iter_images(d):
        if camera and camera != "both" and img.parent.name != camera:
            continue
        txt = img.with_suffix(".txt")
        if not txt.exists():
            continue
        kept = []
        for line in txt.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5 and int(parts[0]) in keep_global_ids:
                cid = remap.get(int(parts[0]), int(parts[0])) if remap else int(parts[0])
                kept.append(f"{cid} {' '.join(parts[1:])}")
        if kept:
            shutil.copy2(img, out_dir / img.name)
            (out_dir / (img.stem + ".txt")).write_text("\n".join(kept))
            n += 1
    return n


def build_cl_cmd(p):
    """Stage new-class data (global ids remapped to 0-based) and replay data
    (old classes kept at their global ids), write the params into the dataset
    AAS (record mode), and return the main_cl.py command.

    Class-incremental: current_classes = the model's existing classes
    (registry[0:offset]); new_classes are appended after them. main_cl.py offsets
    the new-data labels by len(current_classes), so new data must be 0-based —
    which is why we remap here."""
    registry = _registry_classes()
    new_classes = p.get("new_classes") or []
    current_classes = p.get("current_classes") or [c for c in registry if c not in new_classes]
    if not new_classes:
        raise ValueError("pick at least one new class to add")
    new_ids_global = [registry.index(c) for c in new_classes if c in registry]
    remap = {g: i for i, g in enumerate(new_ids_global)}        # global -> 0-based new index
    current_ids = [registry.index(c) for c in current_classes if c in registry]

    work = CL_DIR / "data"
    shutil.rmtree(work, ignore_errors=True)
    new_dir, replay_dir = work / "new", work / "replay"

    n_new = _stage_cl_split(p["new_data_rel"], set(new_ids_global), remap, new_dir,
                            camera=p.get("camera", "both"))
    if n_new == 0:
        raise ValueError(f"no images with the new class(es) {new_classes} in the chosen new-data dataset")
    n_replay = 0
    if p.get("replay_data_rel"):
        n_replay = _stage_cl_split(p["replay_data_rel"], set(current_ids), None, replay_dir,
                                   camera=p.get("camera", "both"))

    model = p.get("model_weights")
    mpath = (OD_DIR / Path(model).name) if model and not Path(model).is_absolute() else Path(model)
    if not Path(mpath).exists():
        cands = list(OD_DIR.rglob(Path(model).name)) if model else []
        mpath = cands[0] if cands else mpath
    if not Path(mpath).exists():
        raise ValueError(f"model not found: {model}")

    exp = p.get("exp_name") or f"cl_{'_'.join(new_classes)}"

    # record mode: write the CL params + the extended class list into the
    # new-data dataset's AAS (ContinualLearning section).
    try:
        aas_rel = generate_dataset_aas(p["new_data_rel"])
        aas_path = AAS_DIR / aas_rel
        data = json.loads(aas_path.read_text(encoding="utf-8-sig"))
        _aas_set_properties(data, {
            "cl_strategy": p.get("cl_strategy", "replay"),
            "input_model": str(mpath),
            "current_classes": ", ".join(current_classes),
            "new_classes": ", ".join(new_classes),
            "new_data_dir": str(new_dir),
            "replay_data_dir": str(replay_dir),
            "replay_ratio": p.get("replay_ratio", 0.3),
            "epochs": int(p.get("epochs", 50)),
            "batch_size": int(p.get("batch_size", 8)),
            "img_size": int(p.get("img_size", 640)),
            "exp_name": exp,
            # the extended class list after this CL step (point 7)
            "names": ", ".join(current_classes + new_classes),
        })
        aas_path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    except Exception as e:
        print(f"[CL] WARNING: could not write AAS record: {e}", file=sys.stderr)

    python = TRAIN_PYTHON if Path(TRAIN_PYTHON).exists() else shutil.which("python")
    cmd = [python, "-X", "utf8", "main_cl.py",
           "--current_classes", ", ".join(current_classes),
           "--new_classes", ", ".join(new_classes),
           "--model_weights", str(mpath),
           "--new_data_dir", str(new_dir),
           "--replay_data_dir", str(replay_dir if n_replay else new_dir),
           "--exp_name", exp,
           "--output_dir", str(CL_DIR / "results" / "cl_runs"),
           "--replay_ratio", str(p.get("replay_ratio", 0.3)),
           "--epochs", str(int(p.get("epochs", 50))),
           "--batch_size", str(int(p.get("batch_size", 8))),
           "--img_size", str(int(p.get("img_size", 640)))]
    return cmd, {"new_classes": new_classes, "current_classes": current_classes,
                 "n_new": n_new, "n_replay": n_replay, "exp": exp}


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

    def _mjpeg_stream(self, q):
        """Stream annotated detections as MJPEG (multipart/x-mixed-replace) from
        a webcam index or a video file. The browser drives it via an <img> tag;
        clearing the img src closes the socket and releases the capture.

        Single-stream guarded: starting a new stream bumps _stream_token so any
        previous loop exits and releases its camera before this one opens."""
        global _stream_token, _stream_active
        try:
            import cv2
            from ultralytics import YOLO
        except Exception as e:
            self.send_error(500, f"opencv/ultralytics unavailable: {e}")
            return

        model_arg = q.get("model", [""])[0]
        source = q.get("source", ["0"])[0]
        conf = float(q.get("conf", ["0.25"])[0])

        mpath = (OD_DIR / Path(model_arg).name) if model_arg and not Path(model_arg).is_absolute() else Path(model_arg)
        if not Path(mpath).exists():
            cands = list(OD_DIR.rglob(Path(model_arg).name)) if model_arg else []
            mpath = cands[0] if cands else mpath
        if not Path(mpath).exists():
            self.send_error(400, f"model not found: {model_arg}")
            return

        # become the active stream; signal any previous one to stop, then wait
        # for it to release the device before we open it.
        with _stream_lock:
            _stream_token += 1
            my_token = _stream_token
        for _ in range(20):
            if not _stream_active:
                break
            time.sleep(0.05)

        cap_src = int(source) if source.isdigit() else source
        cap = cv2.VideoCapture(cap_src, getattr(cv2, "CAP_DSHOW", 0) if isinstance(cap_src, int) else 0)
        if not cap or not cap.isOpened():
            self.send_error(400, f"could not open source: {source}")
            return

        model = YOLO(str(mpath))
        is_webcam = isinstance(cap_src, int)
        _stream_active = True
        log_pipeline_event("inference", "live_started",
                           model=Path(mpath).name, source=("webcam:" + str(cap_src)) if is_webcam else str(cap_src))
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            # DirectShow webcams often return empty first frames; warm up.
            warm = 0
            while is_webcam and warm < 30:
                ok0, _ = cap.read()
                if ok0:
                    break
                warm += 1
                time.sleep(0.05)
            misses = 0
            while my_token == _stream_token:   # a newer stream ends this loop
                ok, frame = cap.read()
                if not ok:
                    if is_webcam and misses < 20:
                        misses += 1
                        time.sleep(0.03)
                        continue
                    break  # end of video file, or camera gone
                misses = 0
                res = model.predict(frame, conf=conf, verbose=False)[0]
                annotated = res.plot()  # BGR ndarray with boxes drawn
                ok2, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok2:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                 b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n")
                self.wfile.write(jpg.tobytes())
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # browser stopped the stream
        finally:
            cap.release()
            if my_token == _stream_token:
                _stream_active = False

    # ---- GET ----
    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(url.query)
        route = url.path

        if route in ("/", "/index.html"):
            return self._file(DASH_DIR / "index.html", "text/html; charset=utf-8")

        if route == "/stream":
            return self._mjpeg_stream(q)

        if route == "/api/cameras_available":
            # Probe webcam indices 0..3. Refuse while a stream holds a camera —
            # a second open of the same device hard-crashes on Windows.
            if _stream_active:
                return self._json({"indices": [], "note": "stop the live stream first"})
            found = []
            try:
                import cv2
                for i in range(4):
                    cap = cv2.VideoCapture(i, getattr(cv2, "CAP_DSHOW", 0))
                    if cap is not None and cap.isOpened():
                        found.append(i)
                        cap.release()
            except Exception as e:
                return self._json({"error": str(e), "indices": []})
            return self._json({"indices": found})

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

        if route == "/api/inference_images":
            # Run YOLO on a folder of images; return per-image box predictions +
            # a base64 JPEG thumbnail so this works for any folder, including
            # paths outside the project root (avoids /img security boundary).
            folder_arg = q.get("folder", [""])[0]
            model_arg = q.get("model", [""])[0]
            conf = float(q.get("conf", ["0.25"])[0])
            limit = min(int(q.get("limit", ["30"])[0]), 200)
            try:
                import base64
                import cv2
                from ultralytics import YOLO
                folder = Path(folder_arg)
                if not folder.is_dir():
                    return self._json({"error": f"folder not found: {folder_arg}"}, 400)
                if model_arg and Path(model_arg).is_absolute():
                    mpath = Path(model_arg)
                elif model_arg and (ROOT / model_arg).exists():
                    mpath = ROOT / model_arg      # e.g. OD_training/results/.../best.pt
                elif model_arg:
                    mpath = OD_DIR / Path(model_arg).name  # e.g. yolov12l.pt
                    if not mpath.exists():
                        cands = list(OD_DIR.rglob(Path(model_arg).name))
                        mpath = cands[0] if cands else mpath
                else:
                    mpath = Path("")
                if not Path(mpath).exists():
                    return self._json({"error": f"model not found: {model_arg}"}, 400)
                imgs = sorted(_iter_images(folder))[:limit]
                if not imgs:
                    return self._json([])
                model = YOLO(str(mpath))
                out = []
                for img_path in imgs:
                    try:
                        res = model.predict(str(img_path), conf=conf, verbose=False)[0]
                        boxes = []
                        if res.boxes is not None:
                            for box in res.boxes:
                                cls = int(box.cls[0])
                                xywhn = box.xywhn[0].tolist()  # [cx,cy,w,h] normalised
                                boxes.append([cls] + xywhn)
                        # thumbnail: resize to 220 px wide, encode as base64 JPEG
                        frame = cv2.imread(str(img_path))
                        data_uri = ""
                        if frame is not None:
                            h, w = frame.shape[:2]
                            tw = 220; th = max(1, int(h * tw / w))
                            thumb = cv2.resize(frame, (tw, th))
                            ok2, jpg = cv2.imencode(".jpg", thumb,
                                                    [cv2.IMWRITE_JPEG_QUALITY, 80])
                            if ok2:
                                data_uri = ("data:image/jpeg;base64,"
                                            + base64.b64encode(jpg).decode())
                        out.append({"name": img_path.name,
                                    "dataUri": data_uri,
                                    "boxes": boxes})
                    except Exception:
                        continue
                return self._json(out)
            except ImportError as ie:
                return self._json({"error": f"missing dependency: {ie}"}, 500)
            except Exception as e:
                return self._json({"error": str(e)}, 500)

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
                dirs, files, n_imgs = [], [], 0
                vid_exts = (".mp4", ".avi", ".mov", ".mkv", ".webm")
                for entry in sorted(p.iterdir(), key=lambda x: x.name.lower()):
                    try:
                        if entry.is_dir():
                            dirs.append(str(entry))
                        else:
                            if entry.suffix.lower() in IMG_EXTS:
                                n_imgs += 1
                            if entry.suffix.lower() in vid_exts:
                                files.append(str(entry))
                    except OSError:
                        continue
                return self._json({"path": str(p), "parent": str(p.parent) if p.parent != p else None,
                                   "dirs": dirs, "files": files, "images": n_imgs})
            except Exception as e:
                return self._json({"error": str(e)}, 400)

        if route == "/api/cl_results":
            # Per-class mAP of every CL run (from results.csv + the run's args),
            # for the CL overview and CL-vs-baseline comparison.
            import csv
            out = []
            base = CL_DIR / "results" / "cl_runs"
            if base.exists():
                for run in sorted(base.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
                    csvf = run / "results.csv"
                    if not (run.is_dir() and csvf.exists()):
                        continue
                    try:
                        rows = list(csv.DictReader(open(csvf)))
                        last = rows[-1] if rows else {}
                        def g(*keys):
                            for k in keys:
                                for c in last:
                                    if c.strip() == k:
                                        return float(last[c])
                            return None
                        out.append({
                            "name": run.name,
                            "epochs": len(rows),
                            "map50": g("metrics/mAP50(B)"),
                            "map5095": g("metrics/mAP50-95(B)"),
                            "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(run.stat().st_mtime)),
                            "rel": str(run.relative_to(ROOT)),
                            "weights": (run / "weights" / "best.pt").exists(),
                        })
                    except Exception:
                        continue
            return self._json(out)

        if route == "/api/defaults":
            # Master cross-stage defaults (pipeline.yaml) for the dashboard forms.
            return self._json(load_defaults())

        if route == "/api/docker_status":
            return self._json(docker_status())

        if route == "/api/pipeline_log":
            # Factory-wide event log (AAS PipelineHistory) — newest first.
            events = read_pipeline_events()
            return self._json({"session": SESSION, "events": list(reversed(events))})

        if route == "/api/session":
            return self._json(SESSION)

        if route == "/api/dataset_aas":
            # Full AAS provenance record for a dataset: generation params, latest
            # results, AL/CL params, and the append-only run history. The
            # dashboard fetches this so all info is sourced from the AAS.
            rel = q.get("rel", [""])[0]
            aas = _find_dataset_aas_path(rel)
            if not aas:
                return self._json({"error": "no AAS record for this dataset yet", "found": False})
            rec = _aas_read_record(aas)
            rec["found"] = True
            rec["aas_file"] = str(aas.relative_to(ROOT))
            return self._json(rec)

        if route == "/api/cl_classes":
            # Registry split into "known by a typical model" vs available to add.
            return self._json({"registry": _registry_classes()})

        if route == "/api/model_classes":
            # Classes a trained model actually saw (so the CL tab can gray them
            # out — only NEW classes should be selectable). The checkpoint's
            # model.names is unreliable (Ultralytics stores the full registry),
            # so we derive the classes from the dataset folder name encoded in
            # the model's path (SINGLE_<comp>_..., MULTI_<a-b-c>_..., REAL_...),
            # falling back to a sidecar trained_classes.json if present.
            model = q.get("model", [""])[0]
            registry = _registry_classes()
            known = _model_trained_classes(model, registry)
            return self._json({"names": known})

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
            ok, err = start_job("SDG generation", cmd, SMARTSDG, meta={**payload, "stage": "SDG"})
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
            cwd = OD_DIR
            if docker_enabled():
                cmd, cwd = dockerize(cmd, OD_DIR)
            ok, err = start_job("OD training", cmd, cwd, meta={**payload, "stage": "training"})
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
                    log_pipeline_event("inference", "real_image_accepted",
                                       harvest=d.name, image=img_name, dataset=f"REAL_{d.name}")
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
            cwd = AL_DIR
            if docker_enabled():
                cmd, cwd = dockerize(cmd, AL_DIR)
            ok, err = start_job(f"Active Learning ({payload.get('strategy', 'ppal_feat')})", cmd, cwd,
                                meta={**payload, "stage": "AL"})
            return self._json({"ok": ok, "error": err, "pool": n_pool, "val": n_val})

        if route == "/api/docker_toggle":
            _set_machine_cfg("docker_enabled", bool(payload.get("enabled")))
            return self._json({"ok": True, "enabled": docker_enabled()})

        if route == "/api/build_docker":
            st = docker_status()
            if not st["installed"]:
                return self._json({"ok": False, "error": "Docker is not installed / not on PATH"}, 400)
            if not st["daemon"]:
                return self._json({"ok": False, "error": "Docker daemon not running — start Docker Desktop first"}, 400)
            cmd = ["docker", "build", "-t", DOCKER_IMAGE, "-f", str(ROOT / "docker" / "Dockerfile"), str(ROOT)]
            ok, err = start_job("Docker image build", cmd, ROOT)
            return self._json({"ok": ok, "error": err})

        if route == "/api/session":
            # Start / name a session or task — tags all subsequent log events.
            if payload.get("name") is not None:
                SESSION["name"] = str(payload["name"])
            if payload.get("task") is not None:
                SESSION["task"] = str(payload["task"])
            if payload.get("new"):
                SESSION["id"] = "session_" + time.strftime("%Y%m%dT%H%M%S")
            log_pipeline_event("session", "session_set",
                               name=SESSION["name"], task=SESSION["task"])
            return self._json({"ok": True, "session": SESSION})

        if route == "/api/stop_stream":
            # Explicitly end any live stream: bumping the token makes the
            # capture loop exit on its next iteration and release the camera
            # (clearing the <img> alone doesn't reliably do this on Windows).
            global _stream_token, _stream_active
            with _stream_lock:
                _stream_token += 1
            _stream_active = False
            return self._json({"ok": True})

        if route == "/api/run_cl":
            try:
                cmd, meta = build_cl_cmd(payload)
            except Exception as e:
                return self._json({"ok": False, "error": str(e)}, 400)
            cwd = CL_DIR
            if docker_enabled():
                cmd, cwd = dockerize(cmd, CL_DIR)
            ok, err = start_job(f"Continual Learning (+{','.join(meta['new_classes'])})", cmd, cwd,
                                meta={**payload, "stage": "CL", **meta})
            return self._json({"ok": ok, "error": err, **meta})

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
    log_pipeline_event("session", "session_started", session=SESSION["id"])
    print(f"SmartSDG dashboard running at {url}  (Ctrl+C to stop)")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")


if __name__ == "__main__":
    main()
