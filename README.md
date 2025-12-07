Physician Notetaker AI
========================

An AI-powered medical NLP system that processes doctor–patient conversations and automatically extracts clinical information, analyzes patient sentiment, identifies key medical keywords, and generates structured SOAP notes using Google’s Gemma 3n 4B model via the OpenRouter API.

Project Overview
----------------
Healthcare professionals spend significant time writing clinical notes.  
This project automates that process using an end-to-end NLP pipeline that:

- Extracts symptoms, diagnosis, treatments, and prognostic details  
- Performs sentiment & intent analysis on patient utterances  
- Identifies key medical terms from the transcript  
- Generates structured SOAP notes ready for documentation  

The system ensures clinical safety, structured output, and zero hallucination by using strict JSON formatting and rule-based constraints.

Core Features
-------------
1. Medical Information Extraction  
2. Sentiment & Intent Analysis  
3. Medical Keyword Extraction  
4. Automatic SOAP Note Generation

Model Used
----------
Google: Gemma 3n 4B (Instruct model) via OpenRouter.

Project Structure
-----------------
Physician-Notetaker-AI/
│
├── data/
│   ├── sample_transcript.txt
│   ├── sample_transcript2.txt
│
├── main.py
├── utils.py
├── nlp_medical.py
├── sentiment_intent.py
├── keyword_extraction.py
├── soap_notes.py
│
├── requirements.txt
├── README.md
├── .gitignore
└── .env.example

Setup
-----
1. Clone:
   git clone https://github.com/Siddharth3710/physician_notetaker
   cd Physician-Notetaker-AI

2. Install:
   pip install -r requirements.txt

3. Add API key:
   Copy .env.example → .env  
   OPENROUTER_API_KEY=your_key_here

Run
---
python main.py

Example SOAP Output
-------------------
{
  "Subjective": {
    "Chief_Complaint": "Occasional backaches",
    "History_of_Present_Illness": "Patient reports discomfort since car accident..."
  },
  "Objective": {
    "Physical_Exam": "Full range of motion with no tenderness"
  },
  "Assessment": {
    "Diagnosis": "Whiplash injury"
  },
  "Plan": {
    "Prognosis": "Expected full recovery in six months"
  }
}

Author
------
Siddharth Jha
GitHub: https://github.com/Siddharth3710
