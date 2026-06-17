"""
AAS passport — one IEC 63278 record per dataset, shared by every pipeline stage.

A "passport" is the per-dataset AAS file
``AAS/datasets/<date>/CA_Modul-OD_dataset_<name>.json``. It is cloned once from
the template (``AAS/CA_Modul-OD_pipeline_v2.0.json``) and then every stage
(SDG -> Training -> Active Learning -> Continual Learning -> Inference) reads its
own configuration from it and writes its own parameters + measurements back into
the SAME file, round after round. The result is a single, git-committed,
reproducible record of exactly how each dataset was produced and used.

This module is the ONE place that knows the AAS layout. It is dependency-free
(stdlib only) so it can be imported from any stage regardless of which Python
interpreter runs it (Isaac Sim for SDG, the conda env for OD/AL/CL).

Layout written into the passport (under the ``ODOptimisation`` container):
  - GenerationParameters  SubmodelElementList  -- SDG config (per round, latest)
  - <training properties>  Properties           -- OD training config (read by
                                                    OD_training generate_config)
  - ActiveLearning        SubmodelElementList   -- AL config
  - ContinualLearning     SubmodelElementList   -- CL config
  - Results               SubmodelElementList   -- latest stage measurements
  - RunHistory            SubmodelElementList    -- append-only, one collection
                                                    per completed run (with a
                                                    config snapshot for reproducibility)

HARD INVARIANT: AAS files are written WITHOUT a UTF-8 BOM (aas_core3 + Isaac yaml
reject BOMs). Always read with utf-8-sig (tolerant) and write with plain utf-8.
"""

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEMPLATE_AAS = ROOT / "AAS" / "CA_Modul-OD_pipeline_v3.0.json"
DATASETS_DIR = ROOT / "AAS" / "datasets"

# Stages whose config lives in a dedicated SubmodelElementList section.
STAGE_CONFIG_SECTION = {
    "SDG": "GenerationParameters",
    "AL": "ActiveLearning",
    "CL": "ContinualLearning",
}
# Training config is stored as top-level Properties (read by OD_training's
# generate_config from OD_Module/ObjectDetection), not a section.
TRAINING_KEYS = ["base_data_dir", "cameras_to_train", "exp_names_to_train",
                 "epochs", "img_size", "batch_size", "weights", "save_period",
                 "val_data_path"]


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# =============================================================================
#  Low-level AAS JSON helpers (single source of truth — the dashboard imports
#  these too, so the on-disk layout stays consistent everywhere).
# =============================================================================

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
    """Recursively set existing AAS Property values by idShort (no creation)."""
    if isinstance(node, dict):
        if node.get("modelType") == "Property" and node.get("idShort") in targets:
            node["value"] = str(targets[node["idShort"]])
        for v in node.values():
            _aas_set_properties(v, targets)
    elif isinstance(node, list):
        for v in node:
            _aas_set_properties(v, targets)


def _aas_prop(id_short, value):
    return {"idShort": id_short, "valueType": "xs:string",
            "value": "" if value is None else str(value), "modelType": "Property"}


def _aas_parent_collection(data):
    """The container we attach provenance sections to: prefer ODOptimisation,
    else the first submodel's submodelElements, else None."""
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


PROVENANCE_SECTIONS = ("GenerationParameters", "ActiveLearning", "ContinualLearning",
                       "Results", "RunHistory")


def _strip_provenance(data):
    """Remove per-dataset provenance sections so a freshly cloned passport starts
    clean. The template is a schema, not a record — any run residue it picked up
    must not be inherited by a new dataset's passport."""
    container = _aas_parent_collection(data)
    if not container:
        return
    container[:] = [e for e in container
                    if not (isinstance(e, dict) and e.get("idShort") in PROVENANCE_SECTIONS)]


def _aas_section_props(data, id_short):
    """Read a section's flat properties into a plain dict (idShort -> value)."""
    node = _aas_find_node(data, id_short)
    if not node or not node.get("value"):
        return {}
    return {e.get("idShort"): e.get("value") for e in node["value"]
            if isinstance(e, dict) and e.get("modelType") == "Property"}


def read_record(aas_path):
    """Read all per-stage sections from a dataset passport into a plain dict for
    surveying (dashboard hyperparameter view, external consumers).

    Returns: generation, training, active_learning, continual_learning, results,
    and the append-only run list."""
    try:
        data = json.loads(Path(aas_path).read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    training = {k: _aas_get_property(data, k) for k in TRAINING_KEYS}
    training = {k: v for k, v in training.items() if v is not None}
    hist = _aas_find_node(data, "RunHistory")
    runs = []
    if hist and hist.get("value"):
        for coll in hist["value"]:
            if isinstance(coll, dict) and coll.get("value"):
                runs.append({e.get("idShort"): e.get("value") for e in coll["value"]
                             if isinstance(e, dict)})
    return {
        "generation": _aas_section_props(data, "GenerationParameters"),
        "training": training,
        "active_learning": _aas_section_props(data, "ActiveLearning"),
        "continual_learning": _aas_section_props(data, "ContinualLearning"),
        "results": _aas_section_props(data, "Results"),
        "runs": runs,
    }


# Back-compat alias (the dashboard historically called this name).
_aas_read_record = read_record


# =============================================================================
#  Passport — the clean per-stage API every pipeline part uses.
# =============================================================================

class Passport:
    """Open (or create) a dataset's AAS passport and read/write per-stage config
    and measurements. Read-modify-write a single file; call save() to persist.

    Typical use inside a stage:
        pp = Passport.for_dataset("MULTI_Cabin-GlueGun_Medium_plane")
        cfg = pp.read_config("Training")            # inputs (with what's recorded)
        ... run the stage ...
        pp.write_measurements("Training", {"map50": 0.81})
        pp.append_run({"stage": "training", "status": "success"})
        pp.save()
    """

    def __init__(self, path, data):
        self.path = Path(path)
        self.data = data

    # -- construction ---------------------------------------------------------
    @classmethod
    def from_file(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return cls(path, data)

    @classmethod
    def resolve_path(cls, dataset_name, datasets_dir=DATASETS_DIR):
        """Find the existing passport for a dataset name (any date), or None."""
        filename = f"CA_Modul-OD_dataset_{dataset_name}.json"
        if datasets_dir.exists():
            hits = list(datasets_dir.rglob(filename))
            if hits:
                return hits[0]
        return None

    @classmethod
    def for_dataset(cls, dataset_name, base_data_dir=None, cameras=None,
                    template_path=TEMPLATE_AAS, datasets_dir=DATASETS_DIR, create=True):
        """Open the passport for a dataset, cloning it from the template on first
        use (the 'every pipeline start creates the passport' behaviour). If a
        record already exists (any date) it is opened in place, never duplicated."""
        existing = cls.resolve_path(dataset_name, datasets_dir)
        if existing:
            return cls.from_file(existing)
        if not create:
            raise FileNotFoundError(f"no passport for dataset '{dataset_name}'")
        data = json.loads(Path(template_path).read_text(encoding="utf-8-sig"))
        _strip_provenance(data)   # never inherit template run residue into a new passport
        props = {"exp_names_to_train": dataset_name}
        if base_data_dir:
            props["base_data_dir"] = str(base_data_dir).rstrip("\\/") + "\\"
        if cameras:
            props["cameras_to_train"] = ", ".join(cameras)
        _aas_set_properties(data, props)
        day_dir = datasets_dir / time.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        out = day_dir / f"CA_Modul-OD_dataset_{dataset_name}.json"
        pp = cls(out, data)
        pp.save()
        return pp

    # -- reads ----------------------------------------------------------------
    def get(self, id_short):
        return _aas_get_property(self.data, id_short)

    def read_section(self, section_id):
        return _aas_section_props(self.data, section_id)

    def read_config(self, stage):
        """Read a stage's recorded configuration as a plain dict."""
        if stage == "Training":
            return {k: self.get(k) for k in TRAINING_KEYS if self.get(k) is not None}
        return self.read_section(STAGE_CONFIG_SECTION.get(stage, stage))

    def survey(self):
        """All per-stage config + measurements + run history (for the dashboard)."""
        return read_record(self.path) if self.path.exists() else {}

    # -- writes (mutate in memory; call save() to persist) --------------------
    def set_properties(self, mapping):
        _aas_set_properties(self.data, mapping)
        return self

    def write_config(self, stage, props):
        """Record a stage's hyperparameters/configuration (latest wins)."""
        props = {k: v for k, v in props.items() if v is not None}
        if stage == "Training":
            self.set_properties(props)
        else:
            _aas_upsert_section(self.data, STAGE_CONFIG_SECTION.get(stage, stage), props)
        return self

    def write_measurements(self, stage, props):
        """Record a stage's outcome metrics into the latest Results section."""
        props = {k: v for k, v in props.items() if v is not None}
        _aas_upsert_section(self.data, "Results", {"stage": stage, "updated_at": _now(), **props})
        return self

    def upsert_section(self, section_id, props):
        _aas_upsert_section(self.data, section_id, props)
        return self

    def append_run(self, entry):
        """Append an immutable run record (kept forever, never overwritten)."""
        _aas_append_run(self.data, dict(entry))
        return self

    def save(self):
        """Persist the passport WITHOUT a UTF-8 BOM (hard invariant)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        return self.path


def record_run(dataset_name, stage, config=None, measurements=None,
               status="success", run_label=None, base_data_dir=None, cameras=None,
               datasets_dir=DATASETS_DIR, template_path=TEMPLATE_AAS):
    """One-call helper for a stage's CLI to self-document into the dataset
    passport: open (or create) it, write this stage's config + measurements,
    append an immutable run record, and save.

    NON-RAISING by design — returns the passport path (str) on success or None on
    failure. Call it at the tail of any stage so a passport-write hiccup can never
    break the actual pipeline run.
    """
    try:
        if not dataset_name:
            return None
        # When the dashboard launches a stage it records provenance itself, so it
        # sets SMARTSDG_SKIP_PASSPORT=1 to suppress the stage's own write (avoids
        # duplicate run-history entries). Direct CLI runs don't set it.
        import os
        if os.environ.get("SMARTSDG_SKIP_PASSPORT") == "1":
            return None
        pp = Passport.for_dataset(dataset_name, base_data_dir=base_data_dir, cameras=cameras,
                                  datasets_dir=datasets_dir, template_path=template_path)
        if config:
            pp.write_config(stage, config)
        if measurements:
            pp.write_measurements(stage, measurements)
        entry = {"stage": run_label or stage, "status": status, "finished_at": _now()}
        if measurements:
            entry.update(measurements)
        pp.append_run(entry)
        return str(pp.save())
    except Exception:
        return None
