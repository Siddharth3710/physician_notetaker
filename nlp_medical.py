import json
from openrouter_client import client


# ---------------------------------------------------------
# Build prompt for Gemma (medical JSON extraction)
# ---------------------------------------------------------
def build_medical_prompt(transcript: str) -> str:
    template = """
You are a medical NLP extraction system.

Extract the following structured medical information from the patient–physician transcript below:

- Patient_Name (if mentioned)
- Accident_Details
- Symptoms
- Diagnosis
- Treatment
- Current_Status
- Physical_Examination
- Prognosis
- Follow_Up_Advice

Rules:
1. Output ONLY valid JSON. No explanations.
2. If any field is missing, return null or an empty list.
3. Symptoms only from patient statements.
4. Prognosis only from physician statements.
5. Keep medical wording concise.

Transcript:
<<<TRANSCRIPT_HERE>>>

Return JSON only in this format:

{
  "Patient_Name": "",
  "Accident_Details": {
    "Date": "",
    "Time": "",
    "Location": "",
    "Mechanism": ""
  },
  "Symptoms": [],
  "Diagnosis": "",
  "Treatment": [],
  "Current_Status": "",
  "Physical_Examination": "",
  "Prognosis": "",
  "Follow_Up_Advice": ""
}
"""
    return template.replace("<<<TRANSCRIPT_HERE>>>", transcript)


# ---------------------------------------------------------
# Clean LLM output (remove ```json fences)
# ---------------------------------------------------------
def clean_json_output(text: str) -> str:
    text = text.strip()

    # Remove markdown code fences
    if text.startswith("```"):
        # Remove leading/trailing ```
        text = text.strip("`").strip()

        # Remove "json" label
        if text.lower().startswith("json"):
            text = text[4:].strip()

    return text


# ---------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------
def extract_medical_details(transcript: str):
    prompt = build_medical_prompt(transcript)

    # Call Gemma 3n 4B model via OpenRouter
    response = client.chat.completions.create(
        model="google/gemma-3-4b-it",  # <-- correct model
        messages=[
            {
                "role": "system",
                "content": "You are a strict medical JSON extraction system.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    # Extract text
    raw_output = response.choices[0].message.content

    # Clean markdown code blocks
    cleaned = clean_json_output(raw_output)

    # Debug print if needed
    # print("CLEANED OUTPUT:", cleaned)

    # Attempt JSON parsing
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print("❌ JSON Parse Error. Raw cleaned output:")
        print(cleaned)
        raise
