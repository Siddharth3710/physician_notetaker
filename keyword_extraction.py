import json
from openrouter_client import client


# ---------------------------------------------------------
# Build prompt for LLM-based keyword extraction
# ---------------------------------------------------------
def build_keyword_prompt(transcript: str):
    template = """
You are a clinical NLP system.

Extract the most important MEDICAL and EVENT-RELATED KEY PHRASES from the transcript below.

Include keywords related to:
- Symptoms
- Diagnosis
- Treatment
- Accident details
- Clinical procedures
- Prognosis
- Medical events

Rules:
1. Output ONLY valid JSON.
2. Only return a JSON array of keyword strings.
3. Each keyword should be concise (1–5 words).
4. Do NOT include duplicate phrases.
5. Focus on medically meaningful terminology.

Transcript:
<<<TRANSCRIPT>>>

Return JSON in this format:

[
  "keyword1",
  "keyword2",
  "keyword3"
]
"""
    return template.replace("<<<TRANSCRIPT>>>", transcript)


# ---------------------------------------------------------
# Clean code fences from LLM output
# ---------------------------------------------------------
def clean_json_output(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    return text


# ---------------------------------------------------------
# Main function: extract keywords
# ---------------------------------------------------------
def extract_keywords(transcript: str):
    prompt = build_keyword_prompt(transcript)

    response = client.chat.completions.create(
        model="google/gemma-3-4b-it",
        messages=[
            {
                "role": "system",
                "content": "You extract clinical keywords from medical transcripts.",
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
        print("❌ JSON Parse Error (keyword output):")
        print(cleaned)
        raise
