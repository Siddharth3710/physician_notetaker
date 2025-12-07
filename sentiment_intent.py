import json
from openrouter_client import client


# ---------------------------------------------------------
# Extract only patient utterances from transcript
# ---------------------------------------------------------
def get_patient_utterances(transcript: str):
    lines = transcript.split("\n")
    patient_lines = []

    for line in lines:
        line = line.strip()
        if line.lower().startswith("patient:"):
            # Remove the "Patient:" prefix
            utterance = line.split(":", 1)[1].strip()
            if utterance:
                patient_lines.append(utterance)

    return patient_lines


# ---------------------------------------------------------
# Build prompt for Gemma (sentiment + intent)
# ---------------------------------------------------------
def build_sentiment_intent_prompt(patient_utterances):
    template = """
You are an NLP system specialized in clinical communication analysis.

Your task is to analyze ONLY the patient's utterances and classify each one with:

1. Sentiment:
   - Anxious
   - Neutral
   - Reassured

2. Intent:
   - Reporting symptoms
   - Expressing concern
   - Seeking reassurance
   - Answering a question
   - General conversation

Rules:
1. Output ONLY valid JSON.
2. Output must be a JSON array of objects.
3. Each object must contain:
   - "utterance"
   - "Sentiment"
   - "Intent"
4. Do NOT analyze physician statements.
5. Be concise and medically accurate.

Patient Utterances:
<<<UTTERANCES>>>

Return only JSON in this format:

[
  {
    "utterance": "",
    "Sentiment": "",
    "Intent": ""
  }
]
"""

    # Join all utterances as one block of text
    block = "\n".join([f"- {u}" for u in patient_utterances])
    return template.replace("<<<UTTERANCES>>>", block)


# ---------------------------------------------------------
# Clean JSON output from LLM (remove ```json fences)
# ---------------------------------------------------------
def clean_json_output(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    return text


# ---------------------------------------------------------
# Main function: analyze sentiment + intent
# ---------------------------------------------------------
def analyze_sentiment_and_intent(transcript: str):
    patient_utterances = get_patient_utterances(transcript)

    prompt = build_sentiment_intent_prompt(patient_utterances)

    response = client.chat.completions.create(
        model="google/gemma-3-4b-it",
        messages=[
            {
                "role": "system",
                "content": "You classify sentiment and intent of clinical utterances.",
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
        print("❌ JSON Parse Error. Raw cleaned output:")
        print(cleaned)
        raise
