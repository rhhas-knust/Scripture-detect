import gradio as gr
import pandas as pd
import numpy as np
import csv
import os
import io
import scipy.io.wavfile as wavfile
from datetime import datetime
from groq import Groq
from src.embed import load_model
from src.detector import ScriptureDetector

groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set.")
groq_client = Groq(api_key=groq_api_key)

print("Loading Scripture Detector models...")
verses = pd.read_parquet("data/processed/verses.parquet")
emb = np.load("data/processed/embeddings.npy")
model = load_model()
detector = ScriptureDetector(verses, emb, model, threshold=0.55)
print("Ready!")

def format_results(results):
    if not results:
        return "<p>No scriptures detected yet...</p>"
    html = ""
    for i, r in enumerate(results):
        border_color = "#2e7d32" if i == 0 else "#bdbdbd"
        bg_color = "#e8f5e9" if i == 0 else "#fafafa"
        verse_ref = f"{r.book} {r.chapter}:{r.verse}" if r.verse else f"{r.book} {r.chapter}"
        html += f"""
        <div style="padding: 15px; border-radius: 10px; border-left: 6px solid {border_color}; background-color: {bg_color}; margin-bottom: 10px;">
            <h3 style="margin-top: 0; color: {border_color}; font-family: sans-serif;">
                {verse_ref} <span style="font-size: 14px; font-weight: normal; color: gray;">({r.method} | Score: {r.score:.3f})</span>
            </h3>
            <p style="font-style: italic; font-size: 18px; font-family: sans-serif;">"{r.text}"</p>
        </div>
        """
    return html

def process_audio_chunk(audio, threshold):
    # If audio is a tuple from the mic
    if isinstance(audio, tuple):
        sample_rate, data = audio
    else:
        return "Waiting for audio...", "<p>Ready.</p>"

    # Write to in-memory WAV buffer
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, data)
    buffer.name = "audio.wav"
    buffer.seek(0)

    try:
        transcription = groq_client.audio.transcriptions.create(
            file=(buffer.name, buffer.read()),
            model="whisper-large-v3",
            response_format="text"
        )
        transcript = transcription.strip()
    except Exception as e:
        return f"Error: {e}", "<p>Transcription failed.</p>"

    if not transcript:
        return "Listening...", "<p>Waiting for speech...</p>"

    detector.threshold = threshold
    results = detector.detect(transcript, top_k=3)
    return transcript, format_results(results)

def process_audio_file(audio_path, threshold):
    if audio_path is None:
        return "No audio provided.", "<p style='color:red;'>Please record or upload audio.</p>"
    try:
        with open(audio_path, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(audio_path, file.read()), model="whisper-large-v3", response_format="text"
            )
        transcript = transcription.strip()
    except Exception as e:
        return f"Error: {e}", "<p>Transcription failed.</p>"

    detector.threshold = threshold
    results = detector.detect(transcript, top_k=3)
    return transcript, format_results(results)

def process_text(transcript, threshold):
    if not transcript.strip():
        return "<p style='color:red;'>Please enter some text.</p>"
    detector.threshold = threshold
    results = detector.detect(transcript, top_k=3)
    return format_results(results)

def log_correction(transcript, correct_verse):
    if not transcript.strip() or not correct_verse.strip():
        return "⚠️ Please enter both the transcript and the correct verse."
    with open("data/corrections.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), transcript, correct_verse, "CORRECTED"])
    return f"✅ Saved! The AI will learn that '{transcript[:30]}...' is {correct_verse}."

with gr.Blocks(title="Scripture Detector") as demo:
    gr.Markdown("# 📖 AI Scripture Detector")

    threshold_slider = gr.Slider(minimum=0.30, maximum=0.90, value=0.55, step=0.05, label="Semantic Threshold")

    with gr.Tabs():
        with gr.TabItem("🔴 Live Preacher Mode"):
            gr.Markdown("### How to use Live Mode on Mobile:\n1. Tap the microphone icon below and allow permissions.\n2. **Leave it recording.** As you speak, the app will send audio chunks to the AI.\n3. Tap stop only when you are finished.")
            with gr.Row():
                with gr.Column():
                    # Setting streaming=True tells Gradio to send chunks
                    live_audio = gr.Audio(sources=["microphone"], streaming=True, type="numpy", label="Live Microphone")
                with gr.Column():
                    live_transcript = gr.Textbox(label="📝 Live Transcript", interactive=False, lines=4)
                    live_html = gr.HTML(label="Live Results")

            # Trigger on stream
            live_audio.stream(fn=process_audio_chunk, inputs=[live_audio, threshold_slider], outputs=[live_transcript, live_html])

        with gr.TabItem("🎙️ Upload Audio"):
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(sources=["upload"], type="filepath", label="Upload Sermon Audio")
                    audio_btn = gr.Button("Transcribe & Detect", variant="primary")
                with gr.Column():
                    transcript_audio_out = gr.Textbox(label="📝 Transcribed Text", interactive=False, lines=4)
                    output_audio_html = gr.HTML(label="Results")

        with gr.TabItem("⌨️ Text Input"):
            with gr.Row():
                with gr.Column():
                    transcript_input = gr.Textbox(label="Paste Transcript", lines=4)
                    text_btn = gr.Button("Detect Scriptures", variant="primary")
                with gr.Column():
                    output_text_html = gr.HTML(label="Results")

    gr.Markdown("---")
    gr.Markdown("### 🧠 Teach the AI (Correction)")
    gr.Markdown("Paste the transcript, type the correct verse, and save it. The AI will learn from this later.")
    with gr.Row():
        correct_verse_input = gr.Textbox(label="Correct Verse (e.g., John 3:16)")
        log_btn = gr.Button("Log Correction")
    log_output = gr.Textbox(label="Status", interactive=False)

    audio_btn.click(fn=process_audio_file, inputs=[audio_input, threshold_slider], outputs=[transcript_audio_out, output_audio_html])
    text_btn.click(fn=process_text, inputs=[transcript_input, threshold_slider], outputs=output_text_html)
    log_btn.click(fn=log_correction, inputs=[live_transcript, correct_verse_input], outputs=log_output)

demo.launch(share=True)