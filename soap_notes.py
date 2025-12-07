import json
from openrouter_client import client


# ---------------------------------------------------------
# Build SOAP prompt
# ---------------------------------------------------------
def build_soap_prompt(transcript: str):
    template = """
You are a clinical documentation assistant.

Convert the transcript below into a structured SOAP note.

SOAP Format:
- Subjective: Patient's symptoms, history, reported issues.
- Objective: Physical exam findings, observable facts.
- Assessment: Diagnosis + clinical interpretation.
- Plan: Treatment, recommendations, follow-up.

Rules:
1. Output ONLY valid JSON.
2. Keep statements concise and clinically accurate.
3. Extract information ONLY from the transcript.

Transcript:
<<<TRANSCRIPT>>>

Return JSON in this format:

{
  "Subjective": {
    "Chief_Complaint": "",
    "History_of_Present_Illness": "",
    "Functional_Impact": ""
  },
  "Objective": {
    "Physical_Exam": "",
    "Observations": ""
  },
  "Assessment": {
    "Diagnosis": "",
    "Severity": "",
    "Clinical_Impression": ""
  },
  "Plan": {
    "Treatment": "",
    "Follow_Up": "",
    "Prognosis": ""
  }
}
"""
    return template.replace("<<<TRANSCRIPT>>>", transcript)


# ---------------------------------------------------------
# Clean JSON fences
# ---------------------------------------------------------
def clean_json_output(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return text


# ---------------------------------------------------------
# Generate SOAP note
# ---------------------------------------------------------
def generate_soap_note(transcript: str):
    prompt = build_soap_prompt(transcript)

    response = client.chat.completions.create(
        model="google/gemma-3-4b-it",
        messages=[
            {
                "role": "system",
                "content": "You convert medical transcripts into SOAP notes.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    raw_output = response.choices[0].message.content
    cleaned = clean_json_output(raw_output)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print("❌ JSON Parse Error (SOAP output):")
        print(cleaned)
        raise
    return cleaned
