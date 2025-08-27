# python -m streamlit run app.py
import os
import re
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from dotenv import load_dotenv
load_dotenv()

apikey = os.getenv("GOOGLE_API_KEY")
print(os.getenv("GOOGLE_API_KEY"))
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",
    temperature=0.3,
    api_key=apikey
)

template = """
You are a helpful assistant. Given the transcript of a YouTube video,
generate detailed structured notes covering the main points, explanations,
and important insights.

Transcript:
{text}

Notes:
"""
prompt = PromptTemplate(template=template, input_variables=["text"])
chain = LLMChain(llm=llm, prompt=prompt)

def get_youtube_video_id(video_url: str) -> str:
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", video_url)
    if not match:
        return None
    return match.group(1)

def fetch_transcript_in_english(video_url: str) -> str:
    try:
        video_id = get_youtube_video_id(video_url)
        if not video_id:
            return "Error: Could not extract video ID."

        transcript_api = YouTubeTranscriptApi()
        transcript_list = transcript_api.list(video_id)
        
        try:
            transcript = transcript_list.find_transcript(['en'])
        except Exception:
            first_transcript = list(transcript_list)[0]
            if first_transcript.is_translatable:
                transcript = first_transcript.translate('en')
            else:
                transcript = first_transcript
        
        fetched = transcript.fetch()
        transcript_text = " ".join([snippet.text for snippet in fetched])
        return transcript_text
        
    except TranscriptsDisabled:
        return "Error: Transcripts are disabled for this video."
    except Exception as e:
        return f"Error: {e}"

def generate_pdf_from_text(text_content: str, file_name: str):
    pdf_doc = SimpleDocTemplate(file_name, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    for line in text_content.split("\n"):
        story.append(Paragraph(line, styles["Normal"]))
        story.append(Spacer(1, 8))

    pdf_doc.build(story)
    return file_name

st.set_page_config(page_title="YouTube Notes Generator", page_icon="🎥", layout="wide")

st.title("YouTube Notes Generator")
st.write("Paste a YouTube link and get detailed notes .")

video_link = st.text_input("Enter YouTube Video Link:", "")

if st.button("Generate Notes") and video_link.strip():
    with st.spinner("Fetching transcript..."):
        transcript_text = fetch_transcript_in_english(video_link)

    if transcript_text.startswith("Error:"):
        st.error(transcript_text)
    else:
        with st.spinner("Generating notes with Gemini..."):
            notes = chain.run(text=transcript_text)

        st.subheader("📝 Detailed Notes")
        st.write(notes)

        pdf_file = "notes.pdf"
        generate_pdf_from_text(notes, pdf_file)
        with open(pdf_file, "rb") as f:
            st.download_button(
                label="📥 Download Notes as PDF",
                data=f,
                file_name="youtube_notes.pdf",
                mime="application/pdf",
            )
