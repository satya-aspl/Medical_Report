import os
import sqlite3
import pandas as pd
import plotly.graph_objects as go
import json
from flask import Flask, render_template, request, redirect, url_for
# from google import genai
from openai import OpenAI
from pydantic import BaseModel
import dotenv
import logging
dotenv.load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['LOG_FILE'] = os.path.join(os.path.dirname(__file__), "logs", "processing.log")
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.dirname(app.config['LOG_FILE']), exist_ok=True)

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
    value: float
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


def extract_lab_report_openai(path: str, model_name: str = "gpt-4o") -> LabReport:
    """Extract lab results using OpenAI VLM (PDF + images via file upload + Responses API)."""
    prompt = _load_prompt("lab_extraction")
    output = analyze_document(path, prompt, model_name=model_name)
    return LabReport.model_validate_json(_strip_json_markdown(output))


def init_db():
    conn = sqlite3.connect('health_data.db')
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS parameters (
            canonical_name TEXT PRIMARY KEY,
            default_unit TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS parameter_aliases (
            alias TEXT PRIMARY KEY,
            canonical_name TEXT NOT NULL,
            FOREIGN KEY (canonical_name) REFERENCES parameters(canonical_name)
        );
        CREATE TABLE IF NOT EXISTS labs (
            date TEXT,
            name_original TEXT,
            name TEXT NOT NULL,
            value REAL,
            unit TEXT,
            ref_min REAL,
            ref_max REAL
        );
    """)
    conn.close()


def _resolve_canonical_name(conn: sqlite3.Connection, name_original: str, name_english: str) -> str:
    """If this parameter exists in DB (by canonical or alias), return canonical name; else return name_english."""
    for candidate in (name_english.strip(), name_original.strip()):
        if not candidate:
            continue
        row = conn.execute(
            "SELECT canonical_name FROM parameters WHERE canonical_name = ?", (candidate,)
        ).fetchone()
        if row:
            return row[0]
        row = conn.execute(
            "SELECT canonical_name FROM parameter_aliases WHERE LOWER(alias) = LOWER(?)", (candidate,)
        ).fetchone()
        if row:
            return row[0]
    return name_english.strip()


def _ensure_parameter(conn: sqlite3.Connection, canonical_name: str, default_unit: str, aliases: list[str]):
    """Ensure parameter and aliases exist."""
    conn.execute(
        "INSERT OR IGNORE INTO parameters (canonical_name, default_unit) VALUES (?, ?)",
        (canonical_name, default_unit),
    )
    for a in aliases:
        if a and a.strip():
            conn.execute(
                "INSERT OR IGNORE INTO parameter_aliases (alias, canonical_name) VALUES (?, ?)",
                (a.strip(), canonical_name),
            )


def _get_existing_ref(conn: sqlite3.Connection, canonical_name: str) -> tuple[float | None, float | None]:
    """Get latest ref_min, ref_max for this parameter from labs."""
    row = conn.execute(
        "SELECT ref_min, ref_max FROM labs WHERE name = ? ORDER BY date DESC LIMIT 1",
        (canonical_name,),
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


def process_and_store_report(conn: sqlite3.Connection, data: LabReport) -> None:
    """Resolve canonical names, convert units (LLM when name+unit exist in DB), apply ref sanity check, insert."""
    for res in data.results:
        canonical = _resolve_canonical_name(conn, res.name_original, res.name)
        aliases = list(dict.fromkeys([x for x in (res.name.strip(), res.name_original.strip()) if x]))
        default_unit = res.unit
        existing = conn.execute(
            "SELECT default_unit FROM parameters WHERE canonical_name = ?", (canonical,)
        ).fetchone()
        if existing:
            default_unit = existing[0]
        else:
            _ensure_parameter(conn, canonical, res.unit, aliases)

        from_u = res.unit.strip().lower()
        to_u = default_unit.strip().lower()
        if from_u != to_u:
            value_stored, ref_min_stored, ref_max_stored = _llm_convert_units(
                canonical, res.value, res.ref_min, res.ref_max, res.unit, default_unit
            )
        else:
            value_stored, ref_min_stored, ref_max_stored = res.value, res.ref_min, res.ref_max

        ref_min_use, ref_max_use = _get_existing_ref(conn, canonical)
        if ref_min_use is not None or ref_max_use is not None:
            if (ref_min_use != ref_min_stored) or (ref_max_use != ref_max_stored):
                _processing_log.info(
                    "Ref discrepancy | param=%s | report_date=%s | report ref=%s–%s | stored ref=%s–%s",
                    canonical, data.date, ref_min_stored, ref_max_stored, ref_min_use, ref_max_use,
                )
            ref_min_use = ref_min_use if ref_min_use is not None else ref_min_stored
            ref_max_use = ref_max_use if ref_max_use is not None else ref_max_stored
        else:
            ref_min_use, ref_max_use = ref_min_stored, ref_max_stored

        conn.execute(
            "INSERT INTO labs (date, name_original, name, value, unit, ref_min, ref_max) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (data.date, res.name_original, canonical, value_stored, default_unit, ref_min_use, ref_max_use),
        )


# --- ROUTES ---
@app.route('/')
def index():
    conn = sqlite3.connect('health_data.db')
    # Get unique investigations for the dropdown
    names = pd.read_sql("SELECT DISTINCT name FROM labs", conn)['name'].tolist()
    
    selected_name = request.args.get('investigation')
    graph_json = None
    processed_date = request.args.get('processed')
    processed_results = None
    if processed_date:
        df = pd.read_sql(
            "SELECT name, value, unit, ref_min, ref_max FROM labs WHERE date = ? ORDER BY name",
            conn,
            params=(processed_date,),
        )
        processed_results = df.to_dict("records") if not df.empty else None

    if selected_name:
        df = pd.read_sql(f"SELECT * FROM labs WHERE name = '{selected_name}' ORDER BY date", conn)
        if not df.empty:
            fig = go.Figure()
            # 1. Add Shaded Reference Range (from latest record)
            latest = df.iloc[-1]
            if pd.notnull(latest['ref_min']) and pd.notnull(latest['ref_max']):
                fig.add_hrect(y0=latest['ref_min'], y1=latest['ref_max'], 
                              fillcolor="rgba(0, 255, 0, 0.1)", line_width=0, layer="below")
            
            # 2. Add Trend Line (date-only, no time)
            dates = df['date'].astype(str).str[:10]  # ensure YYYY-MM-DD
            fig.add_trace(go.Scatter(x=dates.tolist(), y=df['value'].tolist(), mode='lines+markers', name=selected_name))
            fig.update_layout(
                title=f"Trend: {selected_name}",
                xaxis_title="Date",
                yaxis_title=latest['unit'],
                xaxis_type="category",
            )
            graph_json = json.loads(fig.to_json())
            
    conn.close()
    return render_template(
        'index.html',
        names=names,
        graph_json=graph_json,
        selected=selected_name,
        processed_date=processed_date,
        processed_results=processed_results,
    )

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files: return redirect(request.url)
    file = request.files['file']
    if file.filename == '': return redirect(request.url)
    
    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)

    # 1. Extract lab results (backend only)
    data = extract_lab_report_openai(path)

    # 2. Process and store: resolve canonical names, unit conversion, ref sanity check
    conn = sqlite3.connect('health_data.db')
    process_and_store_report(conn, data)
    conn.commit()
    conn.close()

    return redirect(url_for('index', processed=data.date))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)