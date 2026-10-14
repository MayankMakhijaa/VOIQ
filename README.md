# voiq

voiq is a small Flask app that turns recorded conversations into structured notes. Upload an audio file and Gemini (`gemini-2.5-flash`) returns a speaker-labelled transcript translated to Hinglish, along with summaries, sentiment, action items, keywords, contact details it heard, and a draft follow-up email. Results are saved to a local CSV file.

## Requirements

- Python 3.10 or newer
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

## Setup

```bash
python3 -m venv env
source env/bin/activate          # Windows: env\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then set GEMINI_API_KEY in .env
```

## Run

```bash
python app.py
```

The app serves on http://localhost:5000 in debug mode.

## Pages and API

| Route | Method | What it does |
| --- | --- | --- |
| `/` | GET | Home page (`templates/index.html`) |
| `/ui/upload` | GET | Upload form (`templates/upload.html`) |
| `/ui/records` | GET | Saved transcripts (`templates/records.html`) |
| `/ui/reminders` | GET | Reminders (`templates/reminders.html`) |
| `/upload` | POST | Multipart form with an `audio` file and optional `name` and `department` fields. Sends the audio to Gemini, saves the result and returns it as JSON. |
| `/records` | GET | All saved records as JSON |
| `/records/<id>` | DELETE | Deletes a record and its reminder |
| `/reminders` | GET | All detected reminders as JSON |
| `/semantic-search?q=` | GET | Records containing the query words, best matches first (keyword match, not embeddings) |
| `/export/all` | GET | Downloads `records.csv` |
| `/export/pdf/<id>` | GET | Downloads one record as a PDF (characters outside Latin-1, such as Devanagari, show as `?`) |
| `/health` | GET | Returns `{"status": "ok"}` |

Example upload:

```bash
curl -F audio=@call.wav -F name=Rahul -F department=Sales http://localhost:5000/upload
```

## Data files

On first start the app creates two CSV files in the working directory:

- `records.csv` holds every processed call: transcript, summary, sentiment, the full Gemini output and any reminder detected.
- `reminders.csv` holds each reminder detected in a call ("remind me in 30 minutes"), linked to its record.

These contain real conversation content, so they are gitignored along with `.env` and `uploads/`. Keep them out of commits.
