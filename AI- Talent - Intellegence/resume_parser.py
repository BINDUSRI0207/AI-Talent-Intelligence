import os
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path):
    text = ""

    try:
        reader = PdfReader(pdf_path)

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")

    return text


def parse_resumes(folder_path):
    resumes = {}

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            path = os.path.join(folder_path, file)
            resumes[file] = extract_text_from_pdf(path)

    return resumes