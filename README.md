# 📊 Multimodal Financial Investment & Insurance RAG Chatbot (Multimodal + FAISS)

This project is a **Retrieval-Augmented Generation (RAG)** chatbot for **Bajaj Finserv Mutual Fund Factsheets**.  
It extracts data from monthly Factsheetss PDFs (text, tables, and charts via OCR), embeds them using **OpenAI Embeddings + CLIP**, and stores them in a **FAISS vector database**.

The chatbot answers user queries about fund performance, CAGR, risk metrics, and asset allocation — **only using facts from the uploaded PDFs**.  
It can visualize results as tables or charts and provides confidence scores for each answer.

---

## 🚀 Features

✅ **RAG Pipeline**
- Extracts data from Bajaj Finserv factsheets (`pdfplumber`, `PyPDF2` or `pdf2image + OCR`)  
- Generates text and image embeddings  
- Stores vectors in **FAISS**  
- Retrieves context for queries and answers via **OpenAI LLM (gpt-4o-mini)**  

✅ **Multimodal Understanding**
- Handles text + table + chart data  
- Uses CLIP embeddings for images (chart regions)

✅ **Computation Layer**
- Calculates CAGR, Sharpe ratio, averages, and asset allocations using extracted table data

✅ **Chat UI**
- Built in **Streamlit**
- Upload PDFs and chat interactively  
- Displays **tables or charts** in the response  
- Shows **retrieval confidence scores**

✅ **Answer Grounding**
- Cites the source (file name, page number, chunk ID)

✅ **Context-Aware Chat**
- Handles follow-up questions from the same session

---

## 🧠 Tech Stack

| Component | Library / Service |
|------------|------------------|
| **Backend / Orchestration** | LangChain |
| **LLM** | OpenAI GPT-4o-mini |
| **Embeddings** | OpenAI Text Embedding 3 Small + HuggingFace CLIP |
| **Vector DB** | FAISS |
| **Frontend** | Streamlit |
| **OCR** | pytesseract |
| **PDF Parsing** | PyPDF2 / pdfplumber |

---

## 🗂️ Project Structure

bajaj_finserv_rag/
│
├── app.py # Streamlit UI
├── kb/bajaj_finserv_factsheet_Oct.pdf
├── main.ipynb # Data ingestion + FAISS creation
├── faiss_index/ # Vector DB storage
├── .env # API keys & config
├── requirements.txt
└── README.md

yaml
Copy code

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository
```bash
git clone https://github.com/omkarbhosale1623/bajaj-finserv-rag.git
cd bajaj-finserv-rag
2️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
3️⃣ Create .env
bash
Copy code
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
FAISS_INDEX_DIR=faiss_index
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_LLM_MODEL=gpt-4o-mini
4️⃣ Run the App
bash
Copy code
streamlit run streamlit_app.py
🧮 Example Queries
“Calculate 3-year CAGR for Bajaj Growth Fund.”

“Compare fund performance between October and September.”

“Show asset allocation of Balanced Advantage Fund as a pie chart.”

“Explain the Sharpe ratio of Bajaj Conservative Fund.”

📊 Visualization Examples
The app can return:

📈 Line chart of NAV or return trends

🥧 Pie chart of asset allocation

📋 Table of fund comparison metrics

🔒 Important Notes
The chatbot only answers from uploaded PDFs.

If a query is outside the document, it politely replies:

“I cannot find that information in the uploaded factsheets.”

👨‍💻 Credits

Built by Omkar Bhosale

