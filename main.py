import json
from utils import load_transcript
from nlp_medical import extract_medical_details
from sentiment_intent import analyze_sentiment_and_intent
from keyword_extraction import extract_keywords
from openrouter_client import client
from soap_notes import generate_soap_note

# ---------------------------------------------------------


def main():
    transcript = load_transcript("data/sample_transcript1.txt")

    print("\nExtracting medical details...\n")
    medical = extract_medical_details(transcript)
    print(json.dumps(medical, indent=2))

    print("\nAnalyzing sentiment & intent...\n")
    sentiment_results = analyze_sentiment_and_intent(transcript)
    print(json.dumps(sentiment_results, indent=2))

    print("\nExtracting clinical keywords...\n")
    keywords = extract_keywords(transcript)
    print(json.dumps(keywords, indent=2))

    print("\nGenerating SOAP Note...\n")
    soap = generate_soap_note(transcript)
    print(json.dumps(soap, indent=2))


if __name__ == "__main__":
    main()
