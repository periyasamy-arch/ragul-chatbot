from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from google import genai
import shutil
import os
from fpdf import FPDF
import pdfplumber
from docx import Document

# -------------------------------
# 🔑 CONFIG
# -------------------------------
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI()

# Allow frontend (important)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# 🧠 MEMORY SYSTEM
# -------------------------------
user_memory = {}

def get_memory(user_id):
    return user_memory.get(user_id, [])

def update_memory(user_id, message):
    if user_id not in user_memory:
        user_memory[user_id] = []
    user_memory[user_id].append(message)

# -------------------------------
# 📂 UTIL FUNCTIONS
# -------------------------------
def extract_text(file_path):
    text = ""

    if file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""

    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"

    return text

def create_pdf(content, filename="generated_resume.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=11)

    for line in content.split("\n"):
        pdf.multi_cell(0, 8, line)

    pdf.output(filename)
    return filename

# -------------------------------
# 🤖 CHATBOT (WITH MEMORY)
# -------------------------------
@app.post("/chat")
async def chat(data: dict = Body(...)):
    user_id = data.get("user_id", "default")
    message = data["message"]

    history = get_memory(user_id)
    history_text = "\n".join(history[-5:])

    prompt = f"""
    You are an AI Career Assistant.

    Help with:
    - Resume building
    - Interview preparation
    - Soft skills (communication, confidence)

    Previous conversation:
    {history_text}

    User: {message}
    """

    response = client.models.generate_content(
        model="models/gemini-flash-latest",
        contents=prompt
    )

    update_memory(user_id, f"User: {message}")
    update_memory(user_id, f"AI: {response.text}")

    return {"reply": response.text}

# -------------------------------
# 📄 RESUME GENERATOR
# -------------------------------
@app.post("/generate-resume")
async def generate_resume(data: dict):
    prompt = f"""
    Create a professional ATS-friendly resume.

    Name: {data['name']}
    Skills: {data['skills']}
    Experience: {data['experience']}
    Education: {data['education']}

    Format properly with sections.
    """

    response = client.models.generate_content(
        model="models/gemini-flash-latest",
        contents=prompt
    )
    content = response.text

    file_path = create_pdf(content)

    return {"message": "Resume generated", "file": file_path}

# -------------------------------
# 📤 UPLOAD + ANALYZE RESUME
# -------------------------------
@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)

    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = extract_text(file_path)

    prompt = f"""
    Analyze this resume and return STRICT JSON:

    {{
      "score": number,
      "skills": number,
      "experience": number,
      "formatting": number,
      "weak_points": ["..."],
      "suggestions": ["..."],
      "improved_version": "..."
    }}

    Resume:
    {text}
    """

    response = client.models.generate_content(
        model="models/gemini-flash-latest",
        contents=prompt
    )

    return {"analysis": response.text}

# -------------------------------
# 📥 DOWNLOAD RESUME
# -------------------------------
@app.get("/download")
def download():
    return FileResponse(
        "generated_resume.pdf",
        media_type="application/pdf",
        filename="resume.pdf"
    )

# -------------------------------
# 🎤 INTERVIEW PRACTICE
# -------------------------------
@app.post("/interview")
async def interview(data: dict):
    role = data["role"]

    prompt = f"""
    Act as an interviewer for {role}.

    Ask one question.
    Wait for answer.
    Then evaluate:
    - Score /10
    - Feedback
    - Better answer
    """

    response = client.models.generate_content(
        model="models/gemini-flash-latest",
        contents=prompt
    )

    return {"response": response.text}

# -------------------------------
# 🎙 VOICE ANSWER EVALUATION
# -------------------------------
@app.post("/voice-interview")
async def voice_interview(data: dict):
    answer = data["answer"]

    prompt = f"""
    Evaluate this interview answer:

    {answer}

    Give:
    - Score out of 10
    - Feedback
    - Improved answer
    """

    response = client.models.generate_content(
        model="models/gemini-flash-latest",
        contents=prompt

)

    return {"result": response.text}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)