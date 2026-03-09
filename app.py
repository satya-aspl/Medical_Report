import os
import sqlite3
import pandas as pd
import plotly.graph_objects as go
import json
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
from flask_cors import CORS
from openai import OpenAI
from pydantic import BaseModel
import dotenv
import logging
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import uuid

dotenv.load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['LOG_FILE'] = os.path.join(os.path.dirname(__file__), "logs", "processing.log")
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.dirname(app.config['LOG_FILE']), exist_ok=True)

CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://localhost:5000", "http://localhost:5173"]}})

def get_current_user():
    """Get the current user from session."""
    return session.get('user_id')

def require_login(f):
    """Decorator to require user login."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not get_current_user():
            flash('Please log in to continue', 'warning')
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function

_processing_log = logging.getLogger("report_processing")
_processing_log.setLevel(logging.INFO)
_fh = logging.FileHandler(app.config['LOG_FILE'], encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
_processing_log.addHandler(_fh)

# --- AI SETUP ---
# GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
# gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def _load_prompt(name: str) -> str:
    """Load prompt from prompts/<name>.txt"""
    path = os.path.join(_PROMPTS_DIR, f"{name}.txt")
    with open(path, encoding="utf-8") as f:
        return f.read().strip()

class LabResult(BaseModel):
    name_original: str = ""  # As on report (optional from LLM)
    name: str  # English / canonical
    value: float | None
    unit: str
    ref_min: float | None
    ref_max: float | None

class LabReport(BaseModel):
    date: str  # YYYY-MM-DD (report date only, not DOB)
    results: list[LabResult]


# def extract_lab_report_gemini(path: str) -> LabReport:
#     """Extract lab results using Gemini. Requires GEMINI_API_KEY."""
#     if not gemini_client:
#         raise ValueError("GEMINI_API_KEY not set")
#     uploaded_file = gemini_client.files.upload(file=path)
#     prompt = "Extract lab results. Translate names to standard English (e.g. 'Hb' -> 'Hemoglobin')."
#     response = gemini_client.models.generate_content(
#         model="gemini-2.0-flash",
#         contents=[uploaded_file, prompt],
#         config={'response_mime_type': 'application/json', 'response_schema': LabReport}
#     )
#     return response.parsed


def analyze_document(
    file_path: str,
    input_prompt: str,
    model_name: str = "gpt-4o",
    client: OpenAI | None = None,
    ) -> str:
    """
    Analyze a PDF or image using OpenAI multimodal models.
    Works for: PDF, PNG, JPG, JPEG, WebP.
    """
    c = client or openai_client
    if not c:
        raise ValueError("OpenAI client not set (OPENAI_API_KEY required)")
    with open(file_path, "rb") as f:
        file_obj = c.files.create(file=f, purpose="user_data")
    input_payload = [
        {
            "role": "user",
            "content": [
                {"type": "input_file", "file_id": file_obj.id},
                {"type": "input_text", "text": input_prompt},
            ],
        }
    ]
    response = c.responses.create(model=model_name, input=input_payload)
    return response.output_text


def _strip_json_markdown(text: str) -> str:
    """Strip markdown code blocks (```json ... ```) from model output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting for chat display."""
    import re
    text = text.strip()
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Remove inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Remove bold/italic
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    # Remove links but keep text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Remove headers
    text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)
    # Clean up extra whitespace
    text = re.sub(r"\n\n+", "\n\n", text)
    return text.strip()


def extract_lab_report_openai(path: str, model_name: str = "gpt-4o") -> LabReport:
    """Extract lab results using OpenAI VLM (PDF + images via file upload + Responses API)."""
    prompt = _load_prompt("lab_extraction")
    output = analyze_document(path, prompt, model_name=model_name)
    return LabReport.model_validate_json(_strip_json_markdown(output))


def init_db():
    conn = sqlite3.connect('health_data.db')
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS parameters (
            user_id TEXT NOT NULL,
            canonical_name TEXT NOT NULL,
            default_unit TEXT NOT NULL,
            PRIMARY KEY (user_id, canonical_name),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS parameter_aliases (
            user_id TEXT NOT NULL,
            alias TEXT NOT NULL,
            canonical_name TEXT NOT NULL,
            PRIMARY KEY (user_id, alias),
            FOREIGN KEY (user_id, canonical_name) REFERENCES parameters(user_id, canonical_name)
        );
        CREATE TABLE IF NOT EXISTS labs (
            user_id TEXT NOT NULL,
            date TEXT,
            name_original TEXT,
            name TEXT NOT NULL,
            value REAL,
            unit TEXT,
            ref_min REAL,
            ref_max REAL,
            FOREIGN KEY (user_id, name) REFERENCES parameters(user_id, canonical_name)
        );
        CREATE TABLE IF NOT EXISTS assessment_cache (
            user_id TEXT NOT NULL,
            name TEXT NOT NULL,
            latest_value REAL NOT NULL,
            latest_date TEXT,
            latest_ref_min REAL,
            latest_ref_max REAL,
            assessment_json TEXT NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, name),
            FOREIGN KEY (user_id, name) REFERENCES parameters(user_id, canonical_name)
        );
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS chat_messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    conn.close()


def _resolve_canonical_name(conn: sqlite3.Connection, user_id: str, name_original: str, name_english: str) -> str:
    """If this parameter exists in DB (by canonical or alias), return canonical name; else return name_english."""
    for candidate in (name_english.strip(), name_original.strip()):
        if not candidate:
            continue
        row = conn.execute(
            "SELECT canonical_name FROM parameters WHERE user_id = ? AND canonical_name = ?", (user_id, candidate,)
        ).fetchone()
        if row:
            return row[0]
        row = conn.execute(
            "SELECT canonical_name FROM parameter_aliases WHERE user_id = ? AND LOWER(alias) = LOWER(?)", (user_id, candidate,)
        ).fetchone()
        if row:
            return row[0]
    return name_english.strip()


def _ensure_parameter(conn: sqlite3.Connection, user_id: str, canonical_name: str, default_unit: str, aliases: list[str]):
    """Ensure parameter and aliases exist for user."""
    conn.execute(
        "INSERT OR IGNORE INTO parameters (user_id, canonical_name, default_unit) VALUES (?, ?, ?)",
        (user_id, canonical_name, default_unit),
    )
    for a in aliases:
        if a and a.strip():
            conn.execute(
                "INSERT OR IGNORE INTO parameter_aliases (user_id, alias, canonical_name) VALUES (?, ?, ?)",
                (user_id, a.strip(), canonical_name),
            )


def _get_existing_ref(conn: sqlite3.Connection, user_id: str, canonical_name: str) -> tuple[float | None, float | None]:
    """Get latest ref_min, ref_max for this parameter from user's labs."""
    row = conn.execute(
        "SELECT ref_min, ref_max FROM labs WHERE user_id = ? AND name = ? ORDER BY date DESC LIMIT 1",
        (user_id, canonical_name,),
    ).fetchone()
    if row:
        return (row[0], row[1])
    return (None, None)


def _llm_convert_units(
    param_name: str,
    value: float,
    ref_min: float | None,
    ref_max: float | None,
    from_unit: str,
    to_unit: str,
) -> tuple[float, float | None, float | None]:
    """Ask LLM to convert value and ref_min, ref_max from from_unit to to_unit. Returns (value, ref_min, ref_max)."""
    if not openai_client:
        return value, ref_min, ref_max
    template = _load_prompt("unit_conversion")
    prompt = template.format(
        param_name=param_name,
        from_unit=from_unit,
        to_unit=to_unit,
        value=value,
        ref_min=ref_min if ref_min is not None else "null",
        ref_max=ref_max if ref_max is not None else "null",
    )
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        out = json.loads(_strip_json_markdown(raw))
        return (
            float(out.get("value", value)),
            float(out["ref_min"]) if out.get("ref_min") is not None else None,
            float(out["ref_max"]) if out.get("ref_max") is not None else None,
        )
    except Exception as e:
        _processing_log.warning("LLM unit conversion failed: %s", e)
        return value, ref_min, ref_max


def _classify_status(value: float, ref_min: float | None, ref_max: float | None) -> str:
    """Classify latest lab value against reference range."""
    if ref_min is None and ref_max is None:
        return "unknown"

    # Borderline threshold: within 10% of the reference interval near either bound.
    if ref_min is not None and ref_max is not None and ref_max > ref_min:
        margin = (ref_max - ref_min) * 0.10
        if value < ref_min or value > ref_max:
            return "outside"
        if value <= (ref_min + margin) or value >= (ref_max - margin):
            return "borderline"
        return "normal"

    if ref_min is not None:
        if value < ref_min:
            return "outside"
        if value <= ref_min * 1.10:
            return "borderline"
        return "normal"

    # ref_max only
    if value > ref_max:
        return "outside"
    if value >= ref_max * 0.90:
        return "borderline"
    return "normal"


def _fallback_assessment(name: str, value: float, unit: str, ref_min: float | None, ref_max: float | None, status: str) -> dict:
    """Fallback text when OpenAI is unavailable or fails."""
    summary = (
        f"Latest {name}: {value} {unit}. "
        f"Reference range: {ref_min if ref_min is not None else '-'} to {ref_max if ref_max is not None else '-'} ({status})."
    )
    if status in {"outside", "borderline"}:
        tips = [
            "Prioritize whole foods: vegetables, lean protein, and fiber; reduce added sugar and ultra-processed snacks.",
            "Do regular activity: 30 minutes brisk walking 5 days/week plus 2 days of light strength training.",
        ]
    else:
        tips = [
            "Maintain your current routine with balanced meals and hydration.",
            "Keep a consistent weekly exercise schedule and recheck labs as advised.",
        ]
    return {"summary": summary, "tips": tips}


def _get_deeper_assessment(name: str, unit: str, trend_rows: list[dict]) -> dict:
    """Generate deeper lab assessment using OpenAI based on trend and reference range."""
    latest = trend_rows[-1]
    value = float(latest["value"])
    ref_min = latest.get("ref_min")
    ref_max = latest.get("ref_max")
    status = _classify_status(value, ref_min, ref_max)

    if not openai_client:
        return _fallback_assessment(name, value, unit, ref_min, ref_max, status)

    prompt = (
        "You are a clinical-lab explainer for patients. Provide concise, non-diagnostic education. "
        "Do not prescribe medication. If status is outside or borderline, include exactly 2 practical lifestyle tips "
        "(food/exercise) and keep each tip to one line."
    )
    user_data = {
        "investigation": name,
        "unit": unit,
        "latest_value": value,
        "latest_ref_min": ref_min,
        "latest_ref_max": ref_max,
        "status": status,
        "trend": trend_rows,
    }
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        "Analyze this lab trend data and return JSON only with keys: "
                        "summary (string, max 2 sentences) and tips (array of exactly 2 strings). "
                        "If status is normal, tips should be maintenance tips. Data: " + json.dumps(user_data)
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        out = json.loads(_strip_json_markdown(raw))
        summary = str(out.get("summary", "")).strip()
        tips = out.get("tips", [])
        if not isinstance(tips, list):
            tips = []
        tips = [str(t).strip() for t in tips if str(t).strip()]
        if len(tips) < 2:
            fallback = _fallback_assessment(name, value, unit, ref_min, ref_max, status)
            while len(tips) < 2:
                tips.append(fallback["tips"][len(tips)])
        return {
            "summary": summary or _fallback_assessment(name, value, unit, ref_min, ref_max, status)["summary"],
            "tips": tips[:2],
        }
    except Exception as e:
        _processing_log.warning("Deeper assessment failed: %s", e)
        return _fallback_assessment(name, value, unit, ref_min, ref_max, status)


def _float_equal(a: float | None, b: float | None, tol: float = 1e-9) -> bool:
    """Compare floats safely, handling None values."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol


def _ensure_assessment_cache_table(conn: sqlite3.Connection) -> None:
    """Ensure assessment cache table exists for older DB files."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS assessment_cache (
            user_id TEXT NOT NULL,
            name TEXT NOT NULL,
            latest_value REAL NOT NULL,
            latest_date TEXT,
            latest_ref_min REAL,
            latest_ref_max REAL,
            assessment_json TEXT NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, name),
            FOREIGN KEY (user_id, name) REFERENCES parameters(user_id, canonical_name)
        )
        """
    )


def _load_cached_assessment(conn: sqlite3.Connection, user_id: str, name: str, latest: dict) -> dict | None:
    """Return cached assessment when latest value is unchanged for investigation."""
    _ensure_assessment_cache_table(conn)
    row = conn.execute(
        """
        SELECT latest_value, latest_date, latest_ref_min, latest_ref_max, assessment_json
        FROM assessment_cache
        WHERE user_id = ? AND name = ?
        """,
        (user_id, name,),
    ).fetchone()
    if not row:
        return None

    cached_value, _, _, _, cached_json = row
    latest_value = float(latest["value"])
    if not _float_equal(cached_value, latest_value):
        return None

    try:
        parsed = json.loads(cached_json)
        if isinstance(parsed, dict) and "summary" in parsed and "tips" in parsed:
            return parsed
    except Exception:
        return None
    return None


def _save_cached_assessment(conn: sqlite3.Connection, user_id: str, name: str, latest: dict, assessment: dict) -> None:
    """Upsert latest assessment cache for an investigation."""
    _ensure_assessment_cache_table(conn)
    conn.execute(
        """
        INSERT INTO assessment_cache
            (user_id, name, latest_value, latest_date, latest_ref_min, latest_ref_max, assessment_json, updated_at)
        VALUES
            (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(user_id, name) DO UPDATE SET
            latest_value = excluded.latest_value,
            latest_date = excluded.latest_date,
            latest_ref_min = excluded.latest_ref_min,
            latest_ref_max = excluded.latest_ref_max,
            assessment_json = excluded.assessment_json,
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            user_id,
            name,
            float(latest["value"]),
            latest.get("date"),
            latest.get("ref_min"),
            latest.get("ref_max"),
            json.dumps(assessment),
        ),
    )


def _ensure_chat_tables(conn: sqlite3.Connection) -> None:
    """Ensure chat tables exist for older DB files."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS chat_messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        """
    )


def _resolve_investigation_from_query(
    conn: sqlite3.Connection,
    user_id: str,
    query: str,
) -> tuple[str | None, str]:
    """Resolve free-text query to one known investigation for the user."""
    rows = conn.execute(
        "SELECT DISTINCT name FROM labs WHERE user_id = ? ORDER BY name",
        (user_id,),
    ).fetchall()
    names = [r[0] for r in rows if r and r[0]]
    if not names:
        return None, "no-investigations"

    q = query.strip().lower()
    if not q:
        return None, "empty-query"

    by_lower = {n.lower(): n for n in names}
    if q in by_lower:
        return by_lower[q], "exact"

    contains = [n for n in names if n.lower() in q or q in n.lower()]
    if len(contains) == 1:
        return contains[0], "substring"

    if openai_client:
        prompt = (
            "Map this user query to exactly one investigation name from candidates. "
            "Return JSON only with keys: name (string or null), reason (string). "
            "Never invent names outside candidates."
        )
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": json.dumps({"query": query, "candidates": names}),
                    },
                ],
                response_format={"type": "json_object"},
            )
            out = json.loads(_strip_json_markdown(response.choices[0].message.content or "{}"))
            chosen = str(out.get("name") or "").strip()
            if chosen in names:
                return chosen, "llm"
        except Exception as e:
            _processing_log.warning("Query-to-investigation mapping failed: %s", e)

    q_tokens = set(q.replace("_", " ").split())
    best_name = None
    best_score = 0
    for name in names:
        n_tokens = set(name.lower().replace("_", " ").split())
        score = len(q_tokens & n_tokens)
        if score > best_score:
            best_name = name
            best_score = score

    if best_name and best_score > 0:
        return best_name, "token-overlap"
    return None, "no-match"


def _build_lab_context(conn: sqlite3.Connection, user_id: str) -> tuple[pd.DataFrame, str]:
    """Load user labs and build concise context for chat prompts."""
    df = pd.read_sql(
        "SELECT date, name, value, unit, ref_min, ref_max FROM labs WHERE user_id = ? ORDER BY date",
        conn,
        params=(user_id,),
    )
    if df.empty:
        return df, "No lab records available for this user."

    latest_date = str(df['date'].iloc[-1])[:10]
    prev_dates = sorted({str(d)[:10] for d in df['date'].tolist()})
    prev_date = prev_dates[-2] if len(prev_dates) > 1 else None
    names = sorted(df['name'].dropna().unique().tolist())
    context = {
        "latest_report_date": latest_date,
        "previous_report_date": prev_date,
        "investigations": names,
        "total_records": int(len(df)),
    }
    return df, json.dumps(context)


def _generate_chat_reply(message: str, history: list[dict], df: pd.DataFrame, context_json: str) -> str:
    """Generate conversational response using OpenAI with lab context."""
    if df.empty:
        return "I could not find any lab reports yet. Please upload a report first, then ask your question again."

    # Keep prompt payload bounded for responsiveness.
    compact_df = df.tail(120).copy()
    compact_records = compact_df.to_dict(orient='records')

    if not openai_client:
        return (
            "I can compare your latest and previous reports, summarize trends, and explain changes. "
            "OpenAI API is not configured, so detailed AI analysis is currently unavailable."
        )

    system_prompt = (
        "You are a concise lab-report assistant. Use only provided data. "
        "Do not diagnose or prescribe medication. "
        "If user asks to compare current vs last report, provide a short comparison summary. "
        "When uncertain, say what data is missing."
    )

    messages = [{"role": "system", "content": system_prompt}]
    for item in history[-8:]:
        role = item.get('role')
        content = item.get('content')
        if role in {'user', 'assistant'} and content:
            messages.append({"role": role, "content": content})

    messages.append(
        {
            "role": "user",
            "content": (
                f"Lab context: {context_json}\n"
                f"Recent lab rows (JSON): {json.dumps(compact_records)}\n"
                f"User question: {message}\n"
                "Respond in plain text with key findings first, then brief explanation."
            ),
        }
    )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.2,
        )
        raw_reply = (response.choices[0].message.content or "").strip()
        cleaned_reply = _strip_markdown(raw_reply)
        return cleaned_reply or "I could not generate a response."
    except Exception as e:
        _processing_log.warning("Chat response generation failed: %s", e)
        return "I ran into an issue while analyzing your data. Please try again."


def process_and_store_report(conn: sqlite3.Connection, user_id: str, data: LabReport) -> None:
    """Resolve canonical names, convert units (LLM when name+unit exist in DB), apply ref sanity check, insert."""
    for res in data.results:
        # Some OCR/LLM outputs include rows without numeric values; skip instead of failing the full upload.
        if res.value is None:
            _processing_log.warning(
                "Skipping lab row with missing value | user=%s | report_date=%s | name=%s | original=%s",
                user_id,
                data.date,
                res.name,
                res.name_original,
            )
            continue

        canonical = _resolve_canonical_name(conn, user_id, res.name_original, res.name)
        aliases = list(dict.fromkeys([x for x in (res.name.strip(), res.name_original.strip()) if x]))
        default_unit = res.unit
        existing = conn.execute(
            "SELECT default_unit FROM parameters WHERE user_id = ? AND canonical_name = ?", (user_id, canonical,)
        ).fetchone()
        if existing:
            default_unit = existing[0]
        else:
            _ensure_parameter(conn, user_id, canonical, res.unit, aliases)

        from_u = res.unit.strip().lower()
        to_u = default_unit.strip().lower()
        if from_u != to_u:
            value_stored, ref_min_stored, ref_max_stored = _llm_convert_units(
                canonical, res.value, res.ref_min, res.ref_max, res.unit, default_unit
            )
        else:
            value_stored, ref_min_stored, ref_max_stored = res.value, res.ref_min, res.ref_max

        ref_min_use, ref_max_use = _get_existing_ref(conn, user_id, canonical)
        if ref_min_use is not None or ref_max_use is not None:
            if (ref_min_use != ref_min_stored) or (ref_max_use != ref_max_stored):
                _processing_log.info(
                    "Ref discrepancy | user=%s | param=%s | report_date=%s | report ref=%s–%s | stored ref=%s–%s",
                    user_id, canonical, data.date, ref_min_stored, ref_max_stored, ref_min_use, ref_max_use,
                )
            ref_min_use = ref_min_use if ref_min_use is not None else ref_min_stored
            ref_max_use = ref_max_use if ref_max_use is not None else ref_max_stored
        else:
            ref_min_use, ref_max_use = ref_min_stored, ref_max_stored

        conn.execute(
            "INSERT INTO labs (user_id, date, name_original, name, value, unit, ref_min, ref_max) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, data.date, res.name_original, canonical, value_stored, default_unit, ref_min_use, ref_max_use),
        )


# --- ROUTES ---

@app.route('/')
def index():
    """Home page - redirect to dashboard if logged in, else to login"""
    if get_current_user():
        return redirect(url_for('dashboard'))
    return redirect(url_for('login_page'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Register page"""
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        password_confirm = request.form.get('password_confirm', '').strip()
        
        # Validation
        if not email or not password:
            flash('Email and password are required', 'danger')
            return redirect(url_for('register'))
        
        if password != password_confirm:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('register'))
        
        if len(password) < 6:
            flash('Password must be at least 6 characters', 'danger')
            return redirect(url_for('register'))
        
        try:
            conn = sqlite3.connect('health_data.db')
            cursor = conn.cursor()
            
            # Check if user exists
            cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
            if cursor.fetchone():
                flash('Email already registered', 'warning')
                conn.close()
                return redirect(url_for('register'))
            
            # Create user
            import uuid
            user_id = str(uuid.uuid4())
            password_hash = generate_password_hash(password)
            cursor.execute(
                'INSERT INTO users (id, email, password_hash) VALUES (?, ?, ?)',
                (user_id, email, password_hash)
            )
            conn.commit()
            conn.close()
            
            flash('Account created! Please log in.', 'success')
            return redirect(url_for('login_page'))
        except Exception as e:
            flash(f'Registration error: {str(e)}', 'danger')
            return redirect(url_for('register'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    """Login page"""
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        
        if not email or not password:
            flash('Email and password are required', 'danger')
            return redirect(url_for('login_page'))
        
        try:
            conn = sqlite3.connect('health_data.db')
            cursor = conn.cursor()
            cursor.execute('SELECT id, password_hash FROM users WHERE email = ?', (email,))
            user = cursor.fetchone()
            conn.close()
            
            if not user or not check_password_hash(user[1], password):
                flash('Invalid email or password', 'danger')
                return redirect(url_for('login_page'))
            
            # Login successful
            session['user_id'] = user[0]
            session['email'] = email
            flash(f'Welcome back, {email}!', 'success')
            return redirect(url_for('dashboard'))
        except Exception as e:
            flash(f'Login error: {str(e)}', 'danger')
            return redirect(url_for('login_page'))
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logout"""
    session.clear()
    flash('Logged out successfully', 'info')
    return redirect(url_for('login_page'))

@app.route('/dashboard')
@require_login
def dashboard():
    """Dashboard - accessible only to authenticated users"""
    user_id = get_current_user()
    return render_template(
        'index.html',
        names=[],
        graph_json=None,
        selected=None,
        processed_date=None,
        processed_results=None,
    )

@app.route('/api/investigations', methods=['GET'])
@require_login
def get_investigations():
    """Get list of investigations for the current user."""
    user_id = get_current_user()
    try:
        conn = sqlite3.connect('health_data.db')
        names = pd.read_sql("SELECT DISTINCT name FROM labs WHERE user_id = ?", conn, params=(user_id,))['name'].tolist()
        conn.close()
        return jsonify({"investigations": sorted(names)}), 200
    except Exception as e:
        _processing_log.error(f"Error fetching investigations for user {user_id}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/investigation/<investigation_name>', methods=['GET'])
@require_login
def get_investigation_data(investigation_name):
    """Get chart data for a specific investigation."""
    user_id = get_current_user()
    try:
        conn = sqlite3.connect('health_data.db')
        df = pd.read_sql(f"SELECT * FROM labs WHERE user_id = ? AND name = ? ORDER BY date", conn, params=(user_id, investigation_name,))
        conn.close()
        
        if df.empty:
            return jsonify({"error": "No data found"}), 404
        
        fig = go.Figure()
        latest = df.iloc[-1]
        if pd.notnull(latest['ref_min']) and pd.notnull(latest['ref_max']):
            fig.add_hrect(y0=latest['ref_min'], y1=latest['ref_max'], 
                          fillcolor="rgba(0, 255, 0, 0.1)", line_width=0, layer="below")
        
        dates = df['date'].astype(str).str[:10]
        fig.add_trace(go.Scatter(x=dates.tolist(), y=df['value'].tolist(), mode='lines+markers', name=investigation_name))
        fig.update_layout(
            title=f"Trend: {investigation_name}",
            xaxis_title="Date",
            yaxis_title=latest['unit'],
            xaxis_type="category",
        )
        graph_json = json.loads(fig.to_json())
        return jsonify({"graph": graph_json}), 200
    except Exception as e:
        _processing_log.error(f"Error fetching investigation data for user {user_id}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/chat/sessions', methods=['GET'])
@require_login
def get_chat_sessions():
    """Return recent chat sessions for current user."""
    user_id = get_current_user()
    try:
        conn = sqlite3.connect('health_data.db')
        _ensure_chat_tables(conn)
        rows = conn.execute(
            """
            SELECT id, COALESCE(title, 'Untitled chat') AS title, created_at, updated_at
            FROM chat_sessions
            WHERE user_id = ?
            ORDER BY updated_at DESC
            LIMIT 20
            """,
            (user_id,),
        ).fetchall()
        conn.close()
        sessions = [
            {
                "id": r[0],
                "title": r[1],
                "created_at": r[2],
                "updated_at": r[3],
            }
            for r in rows
        ]
        return jsonify({"sessions": sessions}), 200
    except Exception as e:
        _processing_log.error(f"Error loading chat sessions for user {user_id}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/chat/messages/<session_id>', methods=['GET'])
@require_login
def get_chat_messages(session_id):
    """Return full message history for one chat session."""
    user_id = get_current_user()
    try:
        conn = sqlite3.connect('health_data.db')
        _ensure_chat_tables(conn)

        session_row = conn.execute(
            "SELECT id FROM chat_sessions WHERE id = ? AND user_id = ?",
            (session_id, user_id),
        ).fetchone()
        if not session_row:
            conn.close()
            return jsonify({"error": "Session not found."}), 404

        rows = conn.execute(
            """
            SELECT id, role, content, created_at
            FROM chat_messages
            WHERE session_id = ? AND user_id = ?
            ORDER BY created_at ASC
            """,
            (session_id, user_id),
        ).fetchall()
        conn.close()

        messages = [
            {
                "id": r[0],
                "role": r[1],
                "content": r[2],
                "created_at": r[3],
            }
            for r in rows
        ]
        return jsonify({"messages": messages}), 200
    except Exception as e:
        _processing_log.error(f"Error loading chat messages for user {user_id}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/chat/query', methods=['POST'])
@require_login
def chat_query():
    """Chat endpoint with per-user memory and optional graph response."""
    user_id = get_current_user()
    payload = request.get_json(silent=True) or {}
    message = (payload.get('message') or '').strip()
    session_id = (payload.get('session_id') or '').strip()

    if not message:
        return jsonify({"error": "Message is required."}), 400

    try:
        conn = sqlite3.connect('health_data.db')
        _ensure_chat_tables(conn)

        if session_id:
            exists = conn.execute(
                "SELECT id FROM chat_sessions WHERE id = ? AND user_id = ?",
                (session_id, user_id),
            ).fetchone()
            if not exists:
                conn.close()
                return jsonify({"error": "Session not found."}), 404
        else:
            session_id = str(uuid.uuid4())
            title = message[:80]
            conn.execute(
                "INSERT INTO chat_sessions (id, user_id, title) VALUES (?, ?, ?)",
                (session_id, user_id, title),
            )

        user_message_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO chat_messages (id, session_id, user_id, role, content) VALUES (?, ?, ?, ?, ?)",
            (user_message_id, session_id, user_id, 'user', message),
        )

        history_rows = conn.execute(
            """
            SELECT role, content
            FROM chat_messages
            WHERE session_id = ? AND user_id = ?
            ORDER BY created_at ASC
            """,
            (session_id, user_id),
        ).fetchall()
        history = [{"role": r[0], "content": r[1]} for r in history_rows]

        df, context_json = _build_lab_context(conn, user_id)
        reply = _generate_chat_reply(message, history, df, context_json)

        matched_name, match_method = _resolve_investigation_from_query(conn, user_id, message)
        graph = None
        if matched_name:
            inv_df = pd.read_sql(
                "SELECT * FROM labs WHERE user_id = ? AND name = ? ORDER BY date",
                conn,
                params=(user_id, matched_name),
            )
            if not inv_df.empty:
                fig = go.Figure()
                latest = inv_df.iloc[-1]
                if pd.notnull(latest['ref_min']) and pd.notnull(latest['ref_max']):
                    fig.add_hrect(
                        y0=latest['ref_min'],
                        y1=latest['ref_max'],
                        fillcolor="rgba(0, 255, 0, 0.1)",
                        line_width=0,
                        layer="below",
                    )

                dates = inv_df['date'].astype(str).str[:10]
                fig.add_trace(
                    go.Scatter(
                        x=dates.tolist(),
                        y=inv_df['value'].tolist(),
                        mode='lines+markers',
                        name=matched_name,
                    )
                )
                fig.update_layout(
                    title=f"Trend: {matched_name}",
                    xaxis_title="Date",
                    yaxis_title=latest['unit'],
                    xaxis_type="category",
                )
                graph = json.loads(fig.to_json())

        assistant_message_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO chat_messages (id, session_id, user_id, role, content) VALUES (?, ?, ?, ?, ?)",
            (assistant_message_id, session_id, user_id, 'assistant', reply),
        )
        conn.execute(
            "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ? AND user_id = ?",
            (session_id, user_id),
        )
        conn.commit()
        conn.close()

        return jsonify(
            {
                "session_id": session_id,
                "reply": reply,
                "matched_investigation": matched_name,
                "match_method": match_method if matched_name else None,
                "graph": graph,
            }
        ), 200
    except Exception as e:
        _processing_log.error(f"Chat query error for user {user_id}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['POST'])
@require_login
def upload_file():
    user_id = get_current_user()
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    try:
        path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(path)

        # 1. Extract lab results (backend only)
        data = extract_lab_report_openai(path)

        # 2. Process and store: resolve canonical names, unit conversion, ref sanity check
        conn = sqlite3.connect('health_data.db')
        process_and_store_report(conn, user_id, data)
        conn.commit()
        conn.close()

        return jsonify({"success": True, "processed_date": data.date, "message": "File processed successfully"}), 200
    except Exception as e:
        _processing_log.error(f"Upload error for user {user_id}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/assessment', methods=['POST'])
@require_login
def assessment():
    user_id = get_current_user()
    payload = request.get_json(silent=True) or {}
    selected_name = (payload.get('investigation') or '').strip()
    if not selected_name:
        return jsonify({"error": "Missing investigation name."}), 400

    conn = sqlite3.connect('health_data.db')
    df = pd.read_sql(
        "SELECT date, value, unit, ref_min, ref_max FROM labs WHERE user_id = ? AND name = ? ORDER BY date",
        conn,
        params=(user_id, selected_name,),
    )

    if df.empty:
        conn.close()
        return jsonify({"error": "No data found for selected investigation."}), 404

    trend_rows = []
    for _, row in df.tail(8).iterrows():
        trend_rows.append(
            {
                "date": str(row["date"])[:10],
                "value": float(row["value"]),
                "ref_min": float(row["ref_min"]) if pd.notnull(row["ref_min"]) else None,
                "ref_max": float(row["ref_max"]) if pd.notnull(row["ref_max"]) else None,
            }
        )

    latest = trend_rows[-1]
    cached = _load_cached_assessment(conn, user_id, selected_name, latest)
    if cached:
        conn.close()
        return jsonify(cached)

    latest_unit = str(df.iloc[-1]["unit"])
    out = _get_deeper_assessment(selected_name, latest_unit, trend_rows)
    _save_cached_assessment(conn, user_id, selected_name, latest, out)
    conn.commit()
    conn.close()
    return jsonify(out)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)