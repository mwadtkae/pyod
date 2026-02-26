# app.py
import os
import pickle
import sys
import subprocess
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import FastAPI, HTTPException, Body
from openpyxl import Workbook, load_workbook

# ------------------------------------------------------------
# Unpickle shim (legacy pickle references "__main__.GroupRule")
# ------------------------------------------------------------
class GroupRule:
    """Placeholder used ONLY to unpickle legacy models."""
    pass


try:
    sys.modules["__main__"].GroupRule = GroupRule  # type: ignore[attr-defined]
except Exception:
    pass
# ------------------------------------------------------------

app = FastAPI(title="Timesheet Scoring Service", version="2.0")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MODEL_PATHS = [
    os.path.join(BASE_DIR, "models", "halfthreshold_model.pkl"),
    os.path.join(BASE_DIR, "halfthreshold_model.pkl"),
]

MODEL_PATH = os.environ.get("MODEL_PATH", "")
MODEL: Optional[dict] = None

# Excel + script runner settings
SCORES_XLSX_PATH = os.environ.get("SCORES_XLSX_PATH", os.path.join(BASE_DIR, "scores_log.xlsx"))
RUN_SCRIPT_NAME = os.environ.get("RUN_SCRIPT_NAME", "run_me.py")
RUN_SCRIPT_PATH = os.path.join(BASE_DIR, RUN_SCRIPT_NAME)

_excel_lock = Lock()
_script_lock = Lock()

# Your caps
DAILY_HOURS_CAP = float(os.environ.get("DAILY_HOURS_CAP", "8"))
SINGLE_ENTRY_CAP = float(os.environ.get("SINGLE_ENTRY_CAP", "8"))

# Payload key aliases
EMPLOYEE_KEYS = ["Name", "Employee", "employee", "name"]
PROJECT_KEYS = ["Project Name", "Project_Combined", "project", "project_name", "Project"]
TASK_KEYS = ["Project Task", "Task", "project_task", "task"]
DATE_KEYS = ["Timesheet Date", "TimesheetDate", "date", "Date"]
HOURS_KEYS = ["Time in hours", "Hours", "hours", "time_in_hours"]
CONFIRM_KEYS = ["Confirm", "confirm", "Confirmed", "confirmed"]

# Required Excel columns (database columns)
REQUIRED_HEADERS = ["Name", "Project Name", "Project Task", "Timesheet Date", "Time in hours"]


# -----------------------------
# Helpers (basic parsing)
# -----------------------------
def _first_present(d: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in d:
            return d[k]
    return None


def _normalize_str(x: Any) -> str:
    return str(x).strip()


def _norm_key(x: Any) -> str:
    return str(x).strip().lower()


def _parse_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in ("true", "1", "yes", "y", "t", "on")


def _coerce_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        raise ValueError(f"Invalid hours value: {x!r}")


def _parse_payload(payload: Union[List[Dict[str, Any]], Dict[str, Any]]) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if "entries" in payload and isinstance(payload["entries"], list):
            return payload["entries"]
        return [payload]
    raise HTTPException(status_code=400, detail="Payload must be a JSON object or a list of objects.")


def _entry_key(name: str, project: str, task: str, date: str) -> Tuple[str, str, str, str]:
    # uniqueness inside Excel is per (Name, Project Name, Project Task, Timesheet Date)
    return (_norm_key(name), _norm_key(project), _norm_key(task), _norm_key(date))


def _consolidate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Consolidate duplicates for SAME:
      (Name, Project Name, Project Task, Timesheet Date)
    by summing "Time in hours".
    """
    agg: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}

    for r in rows:
        name = str(r["Name"]).strip()
        proj = str(r["Project Name"]).strip()
        task = str(r["Project Task"]).strip()
        datev = str(r["Timesheet Date"]).strip()
        hours = float(r["Time in hours"])

        k = _entry_key(name, proj, task, datev)
        if k not in agg:
            agg[k] = {
                "Name": name,
                "Project Name": proj,
                "Project Task": task,
                "Timesheet Date": datev,
                "Time in hours": 0.0,
                "source_indices": [],
            }

        agg[k]["Time in hours"] += hours

        si = r.get("source_indices")
        if isinstance(si, list):
            agg[k]["source_indices"].extend(si)
        elif "index" in r:
            agg[k]["source_indices"].append(r["index"])

    out = list(agg.values())
    out.sort(
        key=lambda x: (
            _norm_key(x["Name"]),
            _norm_key(x["Timesheet Date"]),
            _norm_key(x["Project Name"]),
            _norm_key(x["Project Task"]),
        )
    )
    return out


# -----------------------------
# Model loading + rule scoring
# -----------------------------
def _normalize_rule_obj(rule_obj: Any) -> Dict[str, Any]:
    """
    Convert legacy rule objects into:
      { "threshold": float, "direction": "below|above", "rule": str }
    """
    if isinstance(rule_obj, dict):
        return rule_obj

    if hasattr(rule_obj, "__dict__"):
        d = dict(rule_obj.__dict__)
        out = {
            "threshold": d.get("threshold", d.get("thr", d.get("t"))),
            "direction": d.get("direction", d.get("dir")),
            "rule": d.get("rule", d.get("name", "RULE")),
        }
        if out["threshold"] is None or out["direction"] is None:
            raise ValueError(f"Rule object missing threshold/direction: {d}")
        return out

    raise ValueError(f"Unsupported rule type in pickle: {type(rule_obj)}")


def _load_model() -> dict:
    candidates = [MODEL_PATH] if MODEL_PATH else DEFAULT_MODEL_PATHS
    model_file = next((p for p in candidates if p and os.path.exists(p)), None)
    if not model_file:
        raise FileNotFoundError(
            "Could not find model pickle. Set MODEL_PATH env var or place it in one of:\n"
            + "\n".join(DEFAULT_MODEL_PATHS)
        )

    with open(model_file, "rb") as f:
        model = pickle.load(f)

    if not isinstance(model, dict):
        raise ValueError(f"Model pickle must be a dict payload, got: {type(model)}")

    model["_model_file"] = model_file

    if "rules" not in model:
        raise ValueError("Model payload missing 'rules'")

    # dict form
    if isinstance(model["rules"], dict):
        converted_rules: Dict[Any, Dict[str, Any]] = {}
        for k, v in model["rules"].items():
            converted_rules[k] = _normalize_rule_obj(v)
        model["rules"] = converted_rules

    # list form
    elif isinstance(model["rules"], list):
        rules_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for r in model["rules"]:
            if not isinstance(r, dict):
                raise ValueError(f"Rule in rules list is not a dict: {type(r)}")
            emp = str(r.get("employee", "")).strip()
            proj = str(r.get("project", "")).strip()
            if not emp or not proj:
                raise ValueError(f"Rule missing employee/project: {r}")
            rr = _normalize_rule_obj(r)
            rules_map[(emp, proj)] = rr
        model["rules"] = rules_map
    else:
        raise ValueError("Model payload 'rules' must be dict or list")

    # Optional: write a clean pickle
    try:
        clean_path = os.path.join(os.path.dirname(model_file), "halfthreshold_model_clean.pkl")
        with open(clean_path, "wb") as wf:
            pickle.dump(model, wf)
        model["_clean_model_file"] = clean_path
    except Exception:
        pass

    return model


def _get_rule_for(emp: str, proj: str) -> Optional[Dict[str, Any]]:
    rules = MODEL.get("rules", {}) if MODEL else {}
    key = (emp, proj)
    if key in rules:
        return rules[key]
    key2 = (_normalize_str(emp), _normalize_str(proj))
    if key2 in rules:
        return rules[key2]
    return None


def _rule_eval(rule_dict: Dict[str, Any], hours: float) -> Dict[str, Any]:
    direction = str(rule_dict.get("direction", "")).lower().strip()
    if direction not in ("below", "above"):
        raise ValueError(f"Rule direction missing/invalid: {direction!r}")

    if "threshold" not in rule_dict:
        raise ValueError("Rule missing required key: 'threshold'")
    threshold = float(rule_dict["threshold"])
    rule_name = str(rule_dict.get("rule", "RULE"))

    below_normal = (direction == "below" and hours < threshold)
    above_normal = (direction == "above" and hours > threshold)

    return {
        "rule": rule_name,
        "threshold": threshold,
        "direction": direction,
        "below_normal": bool(below_normal),
        "above_normal": bool(above_normal),
    }


# -----------------------------
# Excel "database" helpers
# -----------------------------
def _ensure_workbook(path: str):
    """
    Ensures workbook exists AND required headers exist on row 1 somewhere.
    Appends missing headers to the end.
    Returns (wb, ws).
    """
    if os.path.exists(path):
        wb = load_workbook(path)
        ws = wb.active
    else:
        wb = Workbook()
        ws = wb.active
        ws.title = "scores"
        ws.append([])  # row 1

    header_map: Dict[str, int] = {}
    max_col = ws.max_column if ws.max_column else 0

    for col in range(1, max_col + 1):
        v = ws.cell(row=1, column=col).value
        if v is None:
            continue
        header_map[str(v).strip()] = col

    for h in REQUIRED_HEADERS:
        if h not in header_map:
            max_col += 1
            ws.cell(row=1, column=max_col, value=h)
            header_map[h] = max_col

    wb.save(path)
    return wb, ws


def _get_header_map(ws) -> Dict[str, int]:
    header_map: Dict[str, int] = {}
    for col in range(1, ws.max_column + 1):
        v = ws.cell(row=1, column=col).value
        if v is None:
            continue
        header_map[str(v).strip()] = col
    return header_map


def _read_existing_rows(ws, header_map: Dict[str, int]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if ws.max_row < 2:
        return out

    c_name = header_map.get("Name")
    c_proj = header_map.get("Project Name")
    c_task = header_map.get("Project Task")
    c_date = header_map.get("Timesheet Date")
    c_hrs = header_map.get("Time in hours")

    if not (c_name and c_proj and c_task and c_date and c_hrs):
        return out

    for r in range(2, ws.max_row + 1):
        name = ws.cell(row=r, column=c_name).value
        proj = ws.cell(row=r, column=c_proj).value
        task = ws.cell(row=r, column=c_task).value
        datev = ws.cell(row=r, column=c_date).value
        hrs = ws.cell(row=r, column=c_hrs).value
        if name is None and proj is None and task is None and datev is None and hrs is None:
            continue
        out.append(
            {
                "_row": r,
                "Name": name,
                "Project Name": proj,
                "Project Task": task,
                "Timesheet Date": datev,
                "Time in hours": hrs,
            }
        )
    return out


def _employee_exists(existing: List[Dict[str, Any]], name: str) -> bool:
    nn = _norm_key(name)
    return any(_norm_key(r.get("Name", "")) == nn for r in existing)


def _project_exists_for_employee(existing: List[Dict[str, Any]], name: str, project: str) -> bool:
    nn = _norm_key(name)
    pp = _norm_key(project)
    return any(_norm_key(r.get("Name", "")) == nn and _norm_key(r.get("Project Name", "")) == pp for r in existing)


def _task_exists_for_employee_project(existing: List[Dict[str, Any]], name: str, project: str, task: str) -> bool:
    nn = _norm_key(name)
    pp = _norm_key(project)
    tt = _norm_key(task)
    return any(
        _norm_key(r.get("Name", "")) == nn
        and _norm_key(r.get("Project Name", "")) == pp
        and _norm_key(r.get("Project Task", "")) == tt
        for r in existing
    )


def _has_existing_entries_for_day(existing: List[Dict[str, Any]], name: str, date: str) -> bool:
    nn = _norm_key(name)
    dd = _norm_key(date)
    return any(_norm_key(r.get("Name", "")) == nn and _norm_key(r.get("Timesheet Date", "")) == dd for r in existing)


def _daily_total_for(existing: List[Dict[str, Any]], name: str, date: str) -> float:
    nn = _norm_key(name)
    dd = _norm_key(date)
    total = 0.0
    for r in existing:
        if _norm_key(r.get("Name", "")) == nn and _norm_key(r.get("Timesheet Date", "")) == dd:
            try:
                total += float(r.get("Time in hours") or 0)
            except Exception:
                total += 0.0
    return total


def _consolidate_duplicates_in_sheet(ws, header_map: Dict[str, int]) -> Dict[str, Any]:
    """
    Enforce 1 row per (Name, Project Name, Project Task, Timesheet Date) inside the sheet.
    If duplicates exist: SUM hours into first row and DELETE extra rows.
    """
    c_name = header_map.get("Name")
    c_proj = header_map.get("Project Name")
    c_task = header_map.get("Project Task")
    c_date = header_map.get("Timesheet Date")
    c_hrs = header_map.get("Time in hours")
    if not (c_name and c_proj and c_task and c_date and c_hrs):
        return {"consolidated": False, "reason": "missing_headers"}

    key_to_first_row: Dict[Tuple[str, str, str, str], int] = {}
    key_to_sum: Dict[Tuple[str, str, str, str], float] = {}
    to_delete: List[int] = []

    for r in range(2, ws.max_row + 1):
        name = ws.cell(row=r, column=c_name).value
        proj = ws.cell(row=r, column=c_proj).value
        task = ws.cell(row=r, column=c_task).value
        datev = ws.cell(row=r, column=c_date).value
        hrs = ws.cell(row=r, column=c_hrs).value

        if name is None and proj is None and task is None and datev is None and hrs is None:
            continue

        k = (_norm_key(name), _norm_key(proj), _norm_key(task), _norm_key(datev))
        try:
            h = float(hrs or 0)
        except Exception:
            h = 0.0

        if k not in key_to_first_row:
            key_to_first_row[k] = r
            key_to_sum[k] = h
        else:
            key_to_sum[k] = float(key_to_sum.get(k, 0.0)) + h
            to_delete.append(r)

    if not to_delete:
        return {"consolidated": True, "duplicates_found": 0, "rows_deleted": 0}

    for k, first_row in key_to_first_row.items():
        ws.cell(row=first_row, column=c_hrs, value=float(key_to_sum.get(k, 0.0)))

    deleted = 0
    for r in sorted(to_delete, reverse=True):
        ws.delete_rows(r, 1)
        deleted += 1

    return {"consolidated": True, "duplicates_found": len(to_delete), "rows_deleted": deleted}


def _upsert_rows(ws, header_map: Dict[str, int], existing: List[Dict[str, Any]], rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    UPSERT on (Name, Project Name, Project Task, Timesheet Date):
      - if exists: hours += incoming
      - else: append new row
    """
    # index existing keys -> row number
    existing_index: Dict[Tuple[str, str, str, str], int] = {}
    for r in existing:
        k = (
            _norm_key(r.get("Name", "")),
            _norm_key(r.get("Project Name", "")),
            _norm_key(r.get("Project Task", "")),
            _norm_key(r.get("Timesheet Date", "")),
        )
        if "_row" in r and k not in existing_index:
            existing_index[k] = int(r["_row"])

    c_name = header_map["Name"]
    c_proj = header_map["Project Name"]
    c_task = header_map["Project Task"]
    c_date = header_map["Timesheet Date"]
    c_hrs = header_map["Time in hours"]

    updated = 0
    appended = 0
    next_row = (ws.max_row + 1) if ws.max_row else 2

    for nr in rows:
        k = _entry_key(nr["Name"], nr["Project Name"], nr["Project Task"], nr["Timesheet Date"])
        incoming_h = float(nr["Time in hours"])

        if k in existing_index:
            rnum = existing_index[k]
            cur = ws.cell(row=rnum, column=c_hrs).value
            try:
                cur_h = float(cur or 0.0)
            except Exception:
                cur_h = 0.0
            ws.cell(row=rnum, column=c_hrs, value=cur_h + incoming_h)
            updated += 1
        else:
            ws.cell(row=next_row, column=c_name, value=str(nr["Name"]))
            ws.cell(row=next_row, column=c_proj, value=str(nr["Project Name"]))
            ws.cell(row=next_row, column=c_task, value=str(nr["Project Task"]))
            ws.cell(row=next_row, column=c_date, value=str(nr["Timesheet Date"]))
            ws.cell(row=next_row, column=c_hrs, value=float(nr["Time in hours"]))
            appended += 1
            next_row += 1

    return {"updated": updated, "appended": appended}


# -----------------------------
# Startup
# -----------------------------
@app.on_event("startup")
def startup():
    global MODEL
    MODEL = _load_model()
    print(f"Loaded model: {MODEL.get('_model_file')}")
    if MODEL.get("_clean_model_file"):
        print(f"Wrote clean model: {MODEL.get('_clean_model_file')}")
    print(f"Rules loaded: {len(MODEL.get('rules', {}))}")


# -----------------------------
# The ONLY external endpoint
# -----------------------------
@app.post("/score_entries")
def score_entries(payload: Union[List[Dict[str, Any]], Dict[str, Any]] = Body(...)):
    global MODEL
    """
    Input example:
      {
        "Name": "Sonal Andge",
        "Project Name": "Fisher and Paykel Service",
        "Timesheet Date": "2026-02-09",
        "Time in hours": 5,
        "Project Task": "Development",
        "Confirm": "True"
      }

    Behaviour:
      - Always scores + returns flags + reasons.
      - If Confirm=true (any entry OR top-level Confirm), then after scoring:
          1) append/upsert into the Excel "database"
          2) run training script
          3) reload model on success
        and returns append + training results.
    """
    if not MODEL:
        raise HTTPException(status_code=500, detail="Model not loaded")

    raw_entries = _parse_payload(payload)

    # Allow global Confirm on wrapper object too
    global_confirm = False
    if isinstance(payload, dict):
        global_confirm = _parse_bool(_first_present(payload, CONFIRM_KEYS))

    # Normalise raw payload -> canonical rows (includes Project Task)
    canonical: List[Dict[str, Any]] = []
    per_entry_confirm: List[bool] = []

    for i, row in enumerate(raw_entries):
        if not isinstance(row, dict):
            raise HTTPException(status_code=400, detail=f"Entry index {i} must be a JSON object")

        emp = _first_present(row, EMPLOYEE_KEYS)
        proj = _first_present(row, PROJECT_KEYS)
        task = _first_present(row, TASK_KEYS)
        dt = _first_present(row, DATE_KEYS)
        hrs = _first_present(row, HOURS_KEYS)
        conf = _first_present(row, CONFIRM_KEYS)

        if emp is None or proj is None or task is None or dt is None or hrs is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Entry index {i} missing required fields. "
                    f"Need Name, Project Name, Project Task, Timesheet Date, Time in hours. "
                    f"Got keys: {list(row.keys())}"
                ),
            )

        try:
            hours_f = _coerce_float(hrs)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Entry index {i}: {e}")

        per_entry_confirm.append(_parse_bool(conf))

        canonical.append(
            {
                "index": i,
                "Name": _normalize_str(emp),
                "Project Name": _normalize_str(proj),
                "Project Task": _normalize_str(task),
                "Timesheet Date": _normalize_str(dt),
                "Time in hours": hours_f,
                "source_indices": [i],
            }
        )

    confirm = bool(global_confirm or any(per_entry_confirm))

    # Consolidate duplicates within the request
    consolidated = _consolidate_rows(canonical)

    # Read Excel "database" once for flags (and later for upsert if confirm=true)
    with _excel_lock:
        wb, ws = _ensure_workbook(SCORES_XLSX_PATH)
        header_map = _get_header_map(ws)

        # Keep sheet tidy (optional, but avoids "database" drifting)
        sheet_consolidation = _consolidate_duplicates_in_sheet(ws, header_map)
        existing = _read_existing_rows(ws, header_map)
        wb.save(SCORES_XLSX_PATH)

    # Score + flagging
    results: List[Dict[str, Any]] = []
    incremental_daily: Dict[Tuple[str, str], float] = {}

    for out_i, r in enumerate(consolidated):
        name = r["Name"]
        proj = r["Project Name"]
        task = r["Project Task"]
        datev = r["Timesheet Date"]
        hours = float(r["Time in hours"])
        src = r.get("source_indices", [])

        flags: List[str] = []
        reasons: List[str] = []

        # --- Database existence flags ---
        emp_exists = _employee_exists(existing, name)
        if not emp_exists:
            flags.append("NO_EMPLOYEE_IN_DATABASE")
            reasons.append("No employee with that name in the database.")
        else:
            proj_exists = _project_exists_for_employee(existing, name, proj)
            if not proj_exists:
                flags.append("NO_PROJECT_FOR_EMPLOYEE")
                reasons.append("No project name with that name for that employee.")
            else:
                task_exists = _task_exists_for_employee_project(existing, name, proj, task)
                if not task_exists:
                    flags.append("NO_TASK_FOR_EMPLOYEE_PROJECT")
                    reasons.append("No project task with that name for that employee (for that project).")

        if _has_existing_entries_for_day(existing, name, datev):
            flags.append("EXISTING_ENTRIES_FOR_DAY")
            reasons.append("Existing entries for that day for that employee in database.")

        # --- Hours bounds flags ---
        if hours < 0:
            flags.append("HOURS_BELOW_0")
            reasons.append("Time in hours below 0.")
        if hours > SINGLE_ENTRY_CAP:
            flags.append("HOURS_ABOVE_8")
            reasons.append("Time in hours above 8.")

        # --- Normality flags (model threshold) ---
        rule = _get_rule_for(name, proj)
        rule_meta = {"rule": None, "threshold": None, "direction": None}
        if rule is None:
            flags.append("NO_RULE")
            reasons.append("No normal-range rule found for that employee/project.")
        else:
            try:
                ev = _rule_eval(rule, hours)
                rule_meta = {"rule": ev["rule"], "threshold": ev["threshold"], "direction": ev["direction"]}

                if ev["below_normal"]:
                    flags.append("BELOW_NORMAL_FOR_EMPLOYEE_PROJECT")
                    reasons.append("Time in hours below normal for that project & employee.")
                if ev["above_normal"]:
                    flags.append("ABOVE_NORMAL_FOR_EMPLOYEE_PROJECT")
                    reasons.append("Time in hours above normal for that project & employee.")
            except Exception as e:
                flags.append("RULE_ERROR")
                reasons.append(f"Rule evaluation failed: {e}")

        # Optional: daily cap (useful when Confirm=true, but still informative when Confirm=false)
        key = (_norm_key(name), _norm_key(datev))
        base_total = _daily_total_for(existing, name, datev)
        added_so_far = incremental_daily.get(key, 0.0)
        would_be_total = base_total + added_so_far + hours
        incremental_daily[key] = added_so_far + hours

        if would_be_total > DAILY_HOURS_CAP:
            flags.append("DAILY_TOTAL_ABOVE_8")
            reasons.append("Total hours for that employee/day would exceed 8 (including this request).")

        flagged = len(flags) > 0

        PRIORITY_ORDER = [
            "NO_EMPLOYEE_IN_DATABASE",
            "NO_PROJECT_FOR_EMPLOYEE",
            "NO_TASK_FOR_EMPLOYEE_PROJECT",
            "HOURS_BELOW_0",
            "HOURS_ABOVE_8",
            "DAILY_TOTAL_ABOVE_8",
            "BELOW_NORMAL_FOR_EMPLOYEE_PROJECT",
            "ABOVE_NORMAL_FOR_EMPLOYEE_PROJECT",
            "NO_RULE",
        ]

        FLAG_MESSAGES = {
            "NO_EMPLOYEE_IN_DATABASE": "No employee with that name in the database.",
            "NO_PROJECT_FOR_EMPLOYEE": "No project with that name for that employee.",
            "NO_TASK_FOR_EMPLOYEE_PROJECT": "No task with that name for that employee and project.",
            "HOURS_BELOW_0": "Time in hours is below 0.",
            "HOURS_ABOVE_8": "Time in hours exceeds 8.",
            "DAILY_TOTAL_ABOVE_8": "Total hours for that day exceed 8.",
            "BELOW_NORMAL_FOR_EMPLOYEE_PROJECT": "Hours are below normal for that employee and project.",
            "ABOVE_NORMAL_FOR_EMPLOYEE_PROJECT": "Hours are above normal for that employee and project.",
            "NO_RULE": "No normal range rule exists for this employee and project.",
        }

        # Determine primary reason
        primary_flag = None
        for p in PRIORITY_ORDER:
            if p in flags:
                primary_flag = p
                break

        primary_reason = FLAG_MESSAGES.get(primary_flag) if primary_flag else None

        results.append({
            "Name": name,
            "Project": proj,
            "Task": task,
            "Date": datev,
            "Hours": hours,
            "Flagged": bool(flags),
            "Flag": primary_flag,
            "Reason": primary_reason,
        })

    # Build minimal clean response
    response = {
        "Confirm": confirm,
        "Entries": results
    }

    if not confirm:
        return response

    # If confirm=true: append/upsert then train, then reload model if training succeeded
    append_summary: Dict[str, Any] = {"status": "skipped"}
    train_summary: Dict[str, Any] = {"status": "skipped"}

    # Append/upsert (uses consolidated request rows)
    with _excel_lock:
        wb, ws = _ensure_workbook(SCORES_XLSX_PATH)
        header_map = _get_header_map(ws)

        # consolidate in-sheet first, then re-read for accurate upsert
        sheet_consolidation2 = _consolidate_duplicates_in_sheet(ws, header_map)
        existing2 = _read_existing_rows(ws, header_map)

        up = _upsert_rows(ws, header_map, existing2, consolidated)
        wb.save(SCORES_XLSX_PATH)

        append_summary = {
            "status": "ok",
            "excel_path": SCORES_XLSX_PATH,
            "consolidated_in_request": len(consolidated),
            "updated": int(up["updated"]),
            "appended": int(up["appended"]),
            "sheet_consolidation": sheet_consolidation2,
        }

    # Train (run script) + reload model on success
    if not os.path.exists(RUN_SCRIPT_PATH):
        train_summary = {
            "status": "error",
            "detail": f"Training script not found at {RUN_SCRIPT_PATH}.",
            "script": RUN_SCRIPT_NAME,
            "model_reloaded": False,
        }
    else:
        if not _script_lock.acquire(blocking=False):
            train_summary = {"status": "error", "detail": "Training script is already running", "script": RUN_SCRIPT_NAME}
        else:
            try:
                proc = subprocess.run(
                    [sys.executable, RUN_SCRIPT_PATH],
                    cwd=BASE_DIR,
                    capture_output=True,
                    text=True,
                )

                model_reloaded = False
                if proc.returncode == 0:
                    MODEL = _load_model()
                    model_reloaded = True

                train_summary = {
                    "status": "ok" if proc.returncode == 0 else "error",
                    "returncode": proc.returncode,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "script": RUN_SCRIPT_NAME,
                    "model_reloaded": model_reloaded,
                    "model_file": MODEL.get("_model_file") if MODEL else None,
                    "clean_model_file": MODEL.get("_clean_model_file") if MODEL else None,
                }
            finally:
                _script_lock.release()
    response["Append"] = {
        "Updated": append_summary.get("updated", 0),
        "Appended": append_summary.get("appended", 0)
    }

    response["Training"] = {
        "Success": train_summary.get("status") == "ok",
        "ModelReloaded": train_summary.get("model_reloaded", False)
    }

    return response
