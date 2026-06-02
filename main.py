import os
import re
import json
import numpy as np

import pdfplumber
import faiss

from sentence_transformers import SentenceTransformer


# =========================
# CONFIG
# =========================
DATA_FOLDER = "dataset"
OUTPUT_FILE = "output.json"


# =========================
# SAFE PDF READER
# =========================
def is_valid_pdf(path):
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"%PDF"
    except:
        return False


def load_documents(folder):
    docs = {}

    if not os.path.exists(folder):
        print(f"[ERROR] Folder not found: {folder}")
        return docs

    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        text = ""

        try:
            if file.lower().endswith(".pdf"):

                if not is_valid_pdf(path):
                    print(f"[SKIP INVALID PDF] {file}")
                    continue

                with pdfplumber.open(path) as pdf:
                    for page in pdf.pages:
                        t = page.extract_text()
                        if t:
                            text += t + "\n"

            elif file.lower().endswith(".txt"):
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()

            else:
                continue

            text = " ".join(text.split())

            if text.strip():
                docs[file] = text
            else:
                print(f"[SKIP EMPTY] {file}")

        except Exception as e:
            print(f"[ERROR SKIPPING {file}] {e}")

    return docs


# =========================
# CLASSIFIER (embedding-based)
# =========================
class Classifier:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

        self.labels = {
            "Invoice": "invoice bill payment total amount invoice number company",
            "Resume": "resume CV experience education skills email phone",
            "Utility Bill": "electricity bill gas usage kwh account number bill amount",
            "Other": "general document text information"
        }

        self.label_vecs = self.model.encode(list(self.labels.values()))

    def classify(self, text):

        vec = self.model.encode([text])

        sims = np.dot(vec, self.label_vecs.T)[0]

        idx = int(np.argmax(sims))

        label = list(self.labels.keys())[idx]

        return label


# =========================
# EXTRACTION (regex + spaCy optional fallback)
# =========================
def extract_invoice(text):
    return {
        "invoice_number": re.search(r"INV[- ]?\d+", text).group(0)
        if re.search(r"INV[- ]?\d+", text) else None,

        "date": re.search(r"\d{4}-\d{2}-\d{2}", text).group(0)
        if re.search(r"\d{4}-\d{2}-\d{2}", text) else None,

        "total_amount": re.search(r"(\d+(\.\d{1,2})?)", text).group(1)
        if re.search(r"(\d+(\.\d{1,2})?)", text) else None,
    }


def extract_resume(text):
    return {
        "email": re.search(r"[\w\.-]+@[\w\.-]+", text).group(0)
        if re.search(r"[\w\.-]+@[\w\.-]+", text) else None,

        "phone": re.search(r"(\+?\d[\d\s-]{8,})", text).group(0)
        if re.search(r"(\+?\d[\d\s-]{8,})", text) else None,

        "experience_years": re.search(r"(\d+)\+?\s+years", text).group(1)
        if re.search(r"(\d+)\+?\s+years", text) else None,
    }


def extract_utility(text):
    return {
        "account_number": re.search(r"Account[: ]+(\w+)", text).group(1)
        if re.search(r"Account[: ]+(\w+)", text) else None,

        "usage_kwh": re.search(r"(\d+)\s*kWh", text).group(1)
        if re.search(r"(\d+)\s*kWh", text) else None,

        "amount_due": re.search(r"Amount Due[: ]+(\d+(\.\d+)?)", text).group(1)
        if re.search(r"Amount Due[: ]+(\d+(\.\d+)?)", text) else None,
    }


# =========================
# SEMANTIC SEARCH (FAISS)
# =========================
class SearchEngine:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        self.index = None
        self.docs = []
        self.embeddings = None

    def build(self, docs):

        if not docs:
            print("[WARN] No documents to index")
            return

        self.docs = list(docs.keys())

        vectors = self.model.encode(list(docs.values()))

        vectors = np.array(vectors).astype("float32")

        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)

        self.embeddings = vectors

        dim = vectors.shape[1]

        self.index = faiss.IndexFlatL2(dim)
        self.index.add(vectors)

    def search(self, query, k=3):

        if self.index is None:
            return []

        q = self.model.encode([query]).astype("float32")

        _, idx = self.index.search(q, k)

        return [self.docs[i] for i in idx[0] if i < len(self.docs)]


# =========================
# MAIN PIPELINE
# =========================
def main():

    docs = load_documents(DATA_FOLDER)

    classifier = Classifier()

    results = {}

    for file, text in docs.items():

        doc_class = classifier.classify(text)

        output = {
            "class": doc_class
        }

        if doc_class == "Invoice":
            output.update(extract_invoice(text))

        elif doc_class == "Resume":
            output.update(extract_resume(text))

        elif doc_class == "Utility Bill":
            output.update(extract_utility(text))

        results[file] = output

    os.makedirs("output", exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=4)

    print("\n[✔] Output saved to output.json")

    # SEARCH DEMO
    searcher = SearchEngine()
    searcher.build(docs)

    print("\n[SEARCH RESULT]")
    print(searcher.search("payments due in January"))


if __name__ == "__main__":
    main()