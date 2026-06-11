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
ISAAC_PYTHON = r"C:\Users\hota02\Documents\omniverse\python.bat"
# Interpreter for training jobs (OD/AL/CL). Falls back to 'python' on PATH if
# this conda env doesn't exist on the machine.
TRAIN_PYTHON = r"C:\Users\hota02\AppData\Local\anaconda3\envs\yolov12\python.exe"

PORT = 8765

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
    Class names stay the full global 9-class list — label files use global IDs."""
    d = _safe_path(dataset_rel)
    if not d.is_dir():
        raise ValueError(f"dataset not found: {d}")
    data = json.loads(TEMPLATE_AAS.read_text(encoding="utf-8-sig"))
    cameras = sorted({p.parent.name if p.parent != d else "" for p in d.rglob("*.png")} - {""}) \
              or ["Replicator", "Replicator_01"]
    _aas_set_properties(data, {
        "base_data_dir": str(d) + "\\",
        "cameras_to_train": ", ".join(cameras),
        "exp_names_to_train": d.name,
    })
    out = AAS_DIR / f"CA_Modul-OD_dataset_{d.name}.json"
    # NOTE: must be written WITHOUT a UTF-8 BOM — aas_core3 rejects BOMs.
    out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return out.name


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
    safe = (AAS_DIR / Path(aas).name)
    if not safe.exists():
        raise ValueError(f"AAS file not found: {safe}")
    # Optional camera subset: train on camera 1, camera 2, or both. The chosen
    # AAS is copied to a temp file with cameras_to_train overridden, so the
    # original config stays untouched.
    cameras = p.get("cameras")
    if cameras:
        data = json.loads(safe.read_text(encoding="utf-8-sig"))
        _aas_set_properties(data, {"cameras_to_train": ", ".join(cameras)})
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
                pngs = list(d.rglob("*.png"))
                out.append({
                    "name": d.name,
                    "images": len(pngs),
                    "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(d.stat().st_mtime)),
                    "rel": str(d.relative_to(ROOT)),
                })
    return out


def list_results():
    out = []
    for base in [OD_DIR / "results", OD_DIR / "runs", OD_DIR / "yolo_runs_multi_mix",
                 ROOT / "active_learning" / "output", ROOT / "continual_learning" / "results"]:
        if base.exists():
            for d in sorted(base.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
                if d.is_dir():
                    out.append({
                        "name": f"{base.parent.name}/{base.name}/{d.name}",
                        "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(d.stat().st_mtime)),
                        "rel": str(d.relative_to(ROOT)),
                    })
    return out


def dataset_preview(rel, limit=12):
    d = _safe_path(rel)
    return [str(p.relative_to(ROOT)) for p in sorted(d.rglob("*.png"))[:limit]]


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
            files = sorted(f.name for f in AAS_DIR.glob("*.json"))
            return self._json(files)

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
                aas = AAS_DIR / f"CA_Modul-OD_dataset_{p.name}.json"
                if aas.exists():
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
