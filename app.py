import os
import csv
import uuid
import json
import io
import urllib.parse
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
from collections import Counter

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError
from fpdf import FPDF


# ------------------ CONFIG ------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=GEMINI_API_KEY)

app = Flask(__name__)
CORS(app)


# ------------------ FILES ------------------
CSV_FILE = "records.csv"
REMINDER_FILE = "reminders.csv"

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "id", "name", "department", "timestamp",
            "transcript", "summary", "sentiment", "ai_output", "reminder"
        ])

if not os.path.exists(REMINDER_FILE):
    with open(REMINDER_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "id", "detected_text", "minutes", "remind_time", "created_at"
        ])


# ------------------ MASTER PROMPT ------------------
MASTER_PROMPT = """
You are an advanced AI Voice Intelligence System.


Your tasks:
- Detect how many speakers are talking.
- Identify speakers separately (Speaker 1, Speaker 2...).
- Understand voice of each speaker and store it in your memory
- If a speaker reveals a name or job role, assign it.
- Translate ALL speech to Hinglish (no matter original language).
- Create structured transcripts WITH speaker labels.
- Detect sentiment per speaker.
- Detect overall emotion (happy, sad, stressed, angry, confident).
- Extract summary in 3 levels: short, medium, detailed.
- Extract action items (tasks, deadlines, follow-ups).
- Create an email if someone asks to send them an email.
- Extract keywords/topics.
- Extract important info: emails, phone numbers, URLs, social media profiles.
- Detect meeting type or conversation topic automatically.
- Generate clean JSON only.


OUTPUT STRICT JSON ONLY:


{
  "languages": ["en"],
  "topic": "<detected_topic>",
  "speakers": {
      "count": 2,
      "details": [
          {"id": 1, "name": "Vinit", "role": "Developer", "sentiment": "positive", "emotion": "confident"},
          {"id": 2, "name": "Rahul", "role": "Manager", "sentiment": "neutral", "emotion": "calm"}
      ]
  },
  "transcription": "Speaker 1: <text>\\nSpeaker 2: <text>",
  "summary": {
      "short": "1–2 line summary",
      "medium": ["point1","point2","point3"],
      "detailed": "<long summary paragraph>"
  },
  "action_items": [
      "Submit the report by evening",
      "Call the client tomorrow"
  ],
  "keywords": ["Infosys","Project Update","Meeting"],
  "important_info": {
      "emails": [],
      "phones": [],
      "urls": [],
      "social": {
          "linkedin": "",
          "instagram": "",
          "facebook": ""
      }
  },
  "smart_reminders_detected": [],
  "sentiment_overall": "neutral",
  "emotion_overall": "calm"
  "email": "Hello, Mr. ,
  It was pleasure talking to you about..."
}
"""


# ------------------ AI PROCESSING ------------------
def process_audio_through_ai(audio_bytes: bytes, mime_type: str) -> str:
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {"text": MASTER_PROMPT},
                            {
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": audio_bytes
                                }
                            }
                        ]
                    }
                ]
            )
            return response.text or ""

        except ServerError as e:
            if e.code == 503 and attempt < 2:
                print(f"Gemini overloaded, retrying... ({attempt+1})")
                time.sleep(2 ** attempt)
            else:
                raise

    return ""


# ------------------ JSON SAFE PARSER ------------------
def safe_json_extract(text: str) -> Dict[str, Any]:
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end <= start:
            return {}
        return json.loads(text[start:end])
    except Exception:
        return {}


# ------------------ REMINDER EXTRACT ------------------
def extract_minutes(ai_json: Dict[str, Any]):
    reminders = ai_json.get("smart_reminders_detected", []) or []
    for r in reminders:
        if not isinstance(r, str):
            continue
        digits = "".join(c for c in r if c.isdigit())
        if not digits:
            continue
        n = int(digits)
        if "hour" in r.lower():
            return n * 60, r
        if "minute" in r.lower():
            return n, r
    return None, None


# ------------------ CSV HELPERS ------------------
def save_record(record: Dict[str, Any]):
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            record["id"],
            record["name"],
            record["department"],
            record["timestamp"],
            record["transcript"],
            json.dumps(record["summary"]),
            json.dumps(record["sentiment"]),
            json.dumps(record["ai_output"]),
            json.dumps(record["reminder"])
        ])


JSON_COLUMNS = ["summary", "sentiment", "ai_output", "reminder"]


def load_records() -> List[Dict[str, Any]]:
    if not os.path.exists(CSV_FILE):
        return []
    with open(CSV_FILE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for col in JSON_COLUMNS:
            try:
                row[col] = json.loads(row.get(col) or "null")
            except json.JSONDecodeError:
                pass
        row["created_at"] = row.get("timestamp", "")
    return rows


def save_reminder(reminder: Dict[str, Any]):
    with open(REMINDER_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            reminder["id"],
            reminder["detected_text"],
            reminder["minutes"],
            reminder["remind_time"],
            reminder["created_at"]
        ])


def load_reminders() -> List[Dict[str, Any]]:
    if not os.path.exists(REMINDER_FILE):
        return []
    with open(REMINDER_FILE, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def remove_rows(path: str, ids: set) -> int:
    """Rewrite a CSV without the rows whose id is in ids; returns rows removed."""
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    kept = [r for r in rows if r and r[0] not in ids]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(kept)
    return len(rows) - len(kept)


def record_text(record: Dict[str, Any]) -> str:
    ai = record.get("ai_output") or {}
    summary = record.get("summary") or {}
    parts = [
        record.get("name", ""),
        record.get("department", ""),
        record.get("transcript", ""),
        ai.get("topic", "") if isinstance(ai, dict) else "",
        " ".join(ai.get("keywords", []) or []) if isinstance(ai, dict) else "",
        json.dumps(summary, ensure_ascii=False),
    ]
    return " ".join(str(p) for p in parts).lower()


def pdf_safe(text: Any) -> str:
    # Built-in PDF fonts only cover Latin-1; replace anything else.
    return str(text or "").encode("latin-1", "replace").decode("latin-1")


def write_line(pdf: FPDF, height: float, text: str):
    pdf.multi_cell(0, height, text, new_x="LMARGIN", new_y="NEXT")


# ============================================================
#                       ROUTES
# ============================================================

@app.route("/upload", methods=["POST"])
def upload():
    try:
        audio = request.files.get("audio")
        if not audio:
            return jsonify({"error": "No audio uploaded"}), 400

        audio_bytes = audio.read()
        mime_type = audio.content_type or "audio/wav"

        ai_text = process_audio_through_ai(audio_bytes, mime_type)
        if not ai_text:
            return jsonify({"error": "Empty response from Gemini"}), 500

        ai_json = safe_json_extract(ai_text)

        mins, detected = extract_minutes(ai_json)
        reminder = None

        if mins:
            now = datetime.now()
            reminder = {
                "id": str(uuid.uuid4()),
                "minutes": mins,
                "detected_text": detected,
                "remind_time": (now + timedelta(minutes=mins)).strftime("%Y-%m-%d %H:%M:%S"),
                "created_at": now.strftime("%Y-%m-%d %H:%M:%S")
            }
            save_reminder(reminder)

        record = {
            "id": str(uuid.uuid4()),
            "name": request.form.get("name", "Unknown"),
            "department": request.form.get("department", "General"),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "transcript": ai_json.get("transcription", ""),
            "summary": ai_json.get("summary", {}),
            "sentiment": {"overall": ai_json.get("sentiment_overall", "neutral")},
            "ai_output": ai_json,
            "reminder": reminder
        }

        save_record(record)
        return jsonify(record)

    except Exception as e:
        import traceback
        print("UPLOAD ERROR >>>", repr(e))
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/records")
def records():
    return jsonify(load_records())


@app.route("/records/<record_id>", methods=["DELETE"])
def delete_record(record_id):
    record = next((r for r in load_records() if r["id"] == record_id), None)
    if not record:
        return jsonify({"error": "Record not found"}), 404

    remove_rows(CSV_FILE, {record_id})
    reminder = record.get("reminder")
    if isinstance(reminder, dict) and reminder.get("id"):
        remove_rows(REMINDER_FILE, {reminder["id"]})
    return jsonify({"deleted": record_id})


@app.route("/reminders")
def reminders():
    return jsonify(load_reminders())


@app.route("/semantic-search")
def semantic_search():
    # Keyword search ranked by how many query words each record contains.
    terms = [t for t in request.args.get("q", "").lower().split() if t]
    if not terms:
        return jsonify([])
    scored = []
    for record in load_records():
        text = record_text(record)
        score = sum(text.count(t) for t in terms)
        if score:
            scored.append((score, record))
    scored.sort(key=lambda x: x[0], reverse=True)
    return jsonify([r for _, r in scored])


@app.route("/export/all")
def export_all():
    if not os.path.exists(CSV_FILE):
        return jsonify({"error": "No records"}), 404
    return send_file(
        os.path.abspath(CSV_FILE),
        mimetype="text/csv",
        as_attachment=True,
        download_name="all_records.csv"
    )


@app.route("/export/pdf/<record_id>")
def export_pdf(record_id):
    record = next((r for r in load_records() if r["id"] == record_id), None)
    if not record:
        return jsonify({"error": "Record not found"}), 404

    ai = record.get("ai_output") if isinstance(record.get("ai_output"), dict) else {}
    summary = record.get("summary") if isinstance(record.get("summary"), dict) else {}

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    write_line(pdf, 10, pdf_safe(f"{record.get('name', '')} - Conversation"))
    pdf.set_font("Helvetica", size=10)
    write_line(pdf, 6, pdf_safe(
        f"Department: {record.get('department', '')}\n"
        f"Date: {record.get('timestamp', '')}\n"
        f"Topic: {ai.get('topic', '')}\n"
        f"Sentiment: {ai.get('sentiment_overall', '')}"
    ))

    sections = [
        ("Summary", summary.get("detailed") or summary.get("short", "")),
        ("Action items", "\n".join(f"- {a}" for a in ai.get("action_items", []) or [])),
        ("Transcript", record.get("transcript", "")),
    ]
    for title, body in sections:
        if not body:
            continue
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 12)
        write_line(pdf, 8, title)
        pdf.set_font("Helvetica", size=10)
        write_line(pdf, 6, pdf_safe(body))

    return send_file(
        io.BytesIO(bytes(pdf.output())),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"record_{record_id}.pdf"
    )


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/")
def home_page():
    return render_template("index.html")


@app.route("/ui/upload")
def upload_ui():
    return render_template("upload.html")


@app.route("/ui/records")
def records_ui():
    return render_template("records.html")


@app.route("/ui/reminders")
def reminders_ui():
    return render_template("reminders.html")


# ------------------ RUN ------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)