import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.tools import tool

# ── Step 1: Fetch Apple's latest 10-K from EDGAR ──────────────────────────────
def fetch_edgar_filing(ticker: str = "AAPL", form_type: str = "10-K") -> str:
    """Fetch the latest SEC filing text for a given ticker."""
    
    # EDGAR requires a user agent header
    headers = {"User-Agent": "financial-agent-demo aman@example.com"}
    
    # get company CIK number
    search_url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt=2024-01-01&enddt=2024-12-31&forms={form_type}"
    
    # simpler approach — use the submissions endpoint
    ticker_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={ticker}&type={form_type}&dateb=&owner=include&count=1&search_text=&output=atom"
    
    # most reliable: hit the EDGAR full text search API
    url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&forms={form_type}&dateRange=custom&startdt=2024-01-01&enddt=2025-01-01"
    
    print(f"Fetching {form_type} for {ticker} from EDGAR...")
    
    # use the company facts API to get filing info
    cik_url = f"https://www.sec.gov/cgi-bin/browse-edgar?company={ticker}&CIK=&type={form_type}&dateb=&owner=include&count=5&search_text=&action=getcompany"
    
    # hardcode Apple's CIK for now — we'll make this dynamic later
    # Apple CIK: 0000320193
    cik = "0000320193"
    
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(submissions_url, headers=headers)
    data = response.json()
    
    # find the latest 10-K
    filings = data["filings"]["recent"]
    forms = filings["form"]
    accession_numbers = filings["accessionNumber"]
    
    latest_10k_idx = None
    for i, form in enumerate(forms):
        if form == form_type:
            latest_10k_idx = i
            break
    
    if latest_10k_idx is None:
        return "No 10-K found"
    
    accession = accession_numbers[latest_10k_idx].replace("-", "")
    accession_formatted = accession_numbers[latest_10k_idx]
    
    print(f"Found filing: {accession_formatted}")
    
    # fetch the filing index
    index_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession}/{accession_formatted}-index.htm"
    
    # get the actual text document
    doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession}/{accession_formatted}.txt"
    
    doc_response = requests.get(doc_url, headers=headers)
    
    # return first 50000 chars — enough for meaningful RAG without being too slow
    text = doc_response.text[:50000]
    print(f"Fetched {len(text)} characters")
    return text


# ── Step 2: Chunk the document ─────────────────────────────────────────────────
def build_vector_store(text: str):
    """Chunk text and build FAISS index."""
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,        # ~500 chars per chunk
        chunk_overlap=50,      # 50 char overlap between chunks
        separators=["\n\n", "\n", ".", " "]  # split on paragraphs first, then sentences
    )
    
    chunks = splitter.split_text(text)
    print(f"Created {len(chunks)} chunks from document")
    
    # ── Step 3: Embed chunks ───────────────────────────────────────────────────
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    print("Embedding chunks...")
    vector_store = FAISS.from_texts(chunks, embeddings)
    print("Vector store built")
    
    return vector_store, embeddings


# ── Step 4: Build retriever tool ───────────────────────────────────────────────
def create_retriever_tool(vector_store, k: int = 3):
    """Wrap FAISS retriever as a LangChain tool."""
    
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    
    @tool
    def search_sec_filing(query: str) -> str:
        """
        Search Apple's SEC 10-K filing for information about 
        business operations, risk factors, revenue, and financial results.
        Use this for qualitative information about the company.
        """
        docs = retriever.invoke(query)
        results = []
        for i, doc in enumerate(docs):
            results.append(f"Chunk {i+1}:\n{doc.page_content}")
        return "\n\n".join(results)
    
    return search_sec_filing

from bs4 import BeautifulSoup

def clean_edgar_text(raw_text: str) -> str:
    """Strip XML/HTML tags from EDGAR filing text."""
    soup = BeautifulSoup(raw_text, "html.parser")
    # remove script and style elements
    for tag in soup(["script", "style", "table"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # collapse excessive whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

# ── Step 5: Run everything ─────────────────────────────────────────────────────
print("Building RAG pipeline...")
filing_text = fetch_edgar_filing("AAPL", "10-K")
clean_text = clean_edgar_text(filing_text) 
vector_store, embeddings = build_vector_store(clean_text)
search_sec_filing = create_retriever_tool(vector_store)

print("\nRAG pipeline ready")
print(f"Test retrieval:")
test_result = search_sec_filing.invoke("What are Apple's main revenue sources?")
print(test_result)