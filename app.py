    # app.py
# ==================================================================================================
# Akıllı Tez Rehberi — RAG Chatbot (LangChain + Chroma + Gemini + Gradio)

import os
import re
import json
from typing import List, Tuple

import google.generativeai as genai
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd

import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, sizes


# ==================================================================================================
# 0) KONFIGURASYON
# ==================================================================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-2.0-flash")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", ".chroma")

PDF_PAGE_START = int(os.getenv("PDF_PAGE_START", "13"))
PDF_PAGE_END = int(os.getenv("PDF_PAGE_END", "104"))
PDF_TO_THESIS_OFFSET = int(os.getenv("PDF_TO_THESIS_OFFSET", "0"))

genai.configure(api_key=GOOGLE_API_KEY)

GENERATION_CFG = dict(
    temperature=0.25,
    top_p=0.95,
    top_k=40,
    max_output_tokens=1024
)

RESPONSE_LENGTH_TO_TOKENS = {
    "Kısa": 400,
    "Orta": 800,
    "Uzun": 1500
}
DEFAULT_RESPONSE_LENGTH = os.getenv("DEFAULT_RESPONSE_LENGTH", "Orta")

CURRENT_TOP_K = int(os.getenv("CURRENT_TOP_K", "5"))


# ==================================================================================================
# 1) METIN YARDIMCILARI
# ==================================================================================================
def clean_text(s: str) -> str:
    """Metin temizliği"""
    s = (s or "").replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def split_into_chunks(text: str, size: int = 800, overlap: int = 120) -> List[str]:
    """Metni kelime bazlı parçalara böler"""
    words = (text or "").split()
    if not words:
        return []
    chunks: List[str] = []
    i = 0
    while i < len(words):
        piece = " ".join(words[i:i + size]).strip()
        if piece:
            chunks.append(piece)
        nxt = i + size - overlap
        i = nxt if nxt > i else i + size
    return chunks


def is_valid_page(page_num: int) -> bool:
    """Sayfa filtreleme: PDF sayfa 13-104 arası"""
    return PDF_PAGE_START <= page_num <= PDF_PAGE_END


def pdf_to_thesis_page(pdf_page: int) -> int:
    """PDF sayfa numarasını tez sayfa numarasına çevirir"""
    return pdf_page + PDF_TO_THESIS_OFFSET


# ==================================================================================================
# 2) EMBEDDING VE CHROMA
# ==================================================================================================
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
vectorstore = Chroma(
    client=chroma_client,
    collection_name="docs",
    embedding_function=embeddings
)


# ==================================================================================================
# 3) INGEST FONKSIYONLARI
# ==================================================================================================
def ingest_jsonl(file_obj) -> str:
    """JSONL dosyasını ingest eder"""
    try:
        lines = file_obj.read().decode("utf-8").splitlines()
        texts, metas = [], []
        print(f"📄 {len(lines)} satır okundu")
        
        for ln in lines:
            row = json.loads(ln)
            content = clean_text(row.get("content", ""))
            if not content:
                continue
            meta = row.get("meta", {}) or {}
            
            page_num = meta.get("page_start")
            if page_num and not is_valid_page(int(page_num)):
                continue
                
            texts.append(content)
            metas.append(meta)
            
        if not texts:
            return "JSONL boş ya da geçerli kayıt bulunamadı."
            
        print(f"📊 {len(texts)} parça eklenecek")
        vectorstore.add_texts(texts=texts, metadatas=metas)
        return f"JSONL ingest tamamlandı: {len(texts)} parça eklendi."
    except Exception as e:
        return f"JSONL ingest hatası: {e}"


def ingest_parquet(file_obj) -> str:
    """Parquet dosyasını ingest eder"""
    try:
        df = pd.read_parquet(file_obj)
        if "content" not in df.columns:
            return "Parquet: 'content' sütunu bulunamadı."
        texts, metas = [], []
        for _, r in df.iterrows():
            content = clean_text(str(r["content"]))
            if not content:
                continue
            meta = {}
            if "meta" in df.columns and isinstance(r.get("meta"), dict):
                meta = r.get("meta") or {}
            for key in ["title", "source", "page", "page_start", "page_end"]:
                if key in df.columns:
                    val = r.get(key)
                    if pd.notna(val) and val != "":
                        meta.setdefault(key, val)
            
            page_num = meta.get("page_start")
            if page_num and not is_valid_page(int(page_num)):
                continue
                
            texts.append(content)
            metas.append(meta)
        if not texts:
            return "Parquet: geçerli satır bulunamadı."
        vectorstore.add_texts(texts=texts, metadatas=metas)
        return f"Parquet ingest tamamlandı: {len(texts)} parça eklendi."
    except Exception as e:
        return f"Parquet ingest hatası: {e}"


# ==================================================================================================
# 4) RETRIEVAL - 3 FARKLI ÇÖZÜM
# ==================================================================================================

SYSTEM_MSG = (
    "Aşağıdaki bağlam parçalarını kullanarak yanıt ver. "
    "Bağlamda verilen TÜM ilgili bilgileri, tarihleri, isimleri ve detayları MUTLAKA dahil et. "
    "Eğer bağlamda yeterli bilgi yoksa: 'Bu konu tezde yeterli detayla bulunamadı.' de. "
    "Yanıtı tam cümlelerle bitir; maddeleri yarım bırakma."
)


def retrieve_solution_1(query: str, k: int):
    """
    ÇÖZÜM 1: Top-K: 3 + Sayfa Aralığı Filtresi (13-30)
    Dartmouth/ELIZA bölgesindeki belgeler tercih edilir
    """
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    
    print(f"🔍 Toplam {len(results)} belge bulundu")
    for i, (doc, score) in enumerate(results[:3], 1):
        meta = doc.metadata or {}
        page = page_label(meta)
        print(f"   📄 Belge {i}: Sayfa {page}, skor: {score:.3f}")
    
    CRITICAL_SECTION_START = 13
    CRITICAL_SECTION_END = 30
    
    filtered = []
    for doc, score in results:
        meta = doc.metadata or {}
        page = meta.get("page_start")
        if page and CRITICAL_SECTION_START <= page <= CRITICAL_SECTION_END:
            filtered.append((doc, score))
    
    if not filtered:
        filtered = results[:k]
    else:
        filtered = filtered[:k]
    
    docs = [doc for doc, _score in filtered]
    return docs


def retrieve_solution_2(query: str, k: int):
    """
    ÇÖZÜM 2: Distance → Similarity Dönüşümü
    Negatif distance'ları similarity'ye dönüştür: similarity = 1 / (1 + abs(distance))
    Threshold: 0.005
    """
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    
    print(f"🔍 Toplam {len(results)} belge bulundu")
    
    converted = []
    for doc, distance in results:
        similarity = 1 / (1 + abs(float(distance)))
        converted.append((doc, similarity))
        if len(converted) <= 3:
            meta = doc.metadata or {}
            page = page_label(meta)
            print(f"   📄 Belge {len(converted)}: Sayfa {page}, dönüştürülmüş skor: {similarity:.4f}")
    
    SIMILARITY_THRESHOLD = 0.005
    filtered = [(doc, sim) for doc, sim in converted if sim >= SIMILARITY_THRESHOLD]
    
    if not filtered:
        filtered = converted[:k]
    else:
        filtered = filtered[:k]
    
    docs = [doc for doc, _score in filtered]
    return docs


def retrieve_solution_3(query: str, k: int):
    """
    ÇÖZÜM 3: Yeni Embedding Modeli (paraphrase-multilingual-mpnet-base-v2)
    Bu model daha iyi embedding üretiyor, skorlar genellikle 0-1 arasında ve tutarlı
    Threshold: 0.5
    """
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    
    print(f"🔍 Toplam {len(results)} belge bulundu")
    for i, (doc, score) in enumerate(results[:3], 1):
        meta = doc.metadata or {}
        page = page_label(meta)
        print(f"   📄 Belge {i}: Sayfa {page}, skor: {score:.3f}")
    
    filtered = [(doc, score) for doc, score in results if score >= 0.5]
    
    if not filtered:
        filtered = results[:k]
    else:
        filtered = filtered[:k]
    
    docs = [doc for doc, _score in filtered]
    return docs


# Kullanılacak retrieve fonksiyonu seç (hangisini test etmek istersen)
# SEÇENEK:
# retrieve = retrieve_solution_1  # Top-K + Sayfa Filtresi
# retrieve = retrieve_solution_2  # Distance → Similarity
retrieve = retrieve_solution_3  # Yeni Embedding Model

def page_label(meta: dict) -> str:
    """Metadata'dan sayfa etiketini çıkarır"""
    if meta.get("page_label"): 
        return str(meta["page_label"])
    if meta.get("logical_page"):
        return str(meta["logical_page"])
    
    if meta.get("page_start") and meta.get("page_end"):
        start = str(meta["page_start"])
        end = str(meta["page_end"])
        if start == end:
            return start
        return f"{start}-{end}"
    if meta.get("page_start"):
        return str(meta["page_start"])
    
    if meta.get("page") is not None:
        try:
            return str(int(meta["page"]) + PDF_TO_THESIS_OFFSET)
        except Exception:
            return str(meta["page"])
    
    return "?"


def build_prompt(query: str, docs, length_choice: str) -> str:
    """Gemini'ye verilecek istem oluştur"""
    length_instructions = {
        "Kısa": "Kısa ve öz bir yanıt ver. Sadece temel bilgileri belirt.",
        "Orta": "Detaylı ama özlü bir yanıt ver. Önemli noktaları açıkla.",
        "Uzun": "Kapsamlı ve detaylı bir yanıt ver. Tüm ilgili bilgileri, örnekleri ve açıklamaları dahil et."
    }
    
    ctx_lines = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        page_display = page_label(meta)
        ctx_lines.append(f"[{i}] ({src} s.{page_display}) {d.page_content}")
    
    context = "\n\n".join(ctx_lines) if ctx_lines else "(bağlam yok)"
    length_instruction = length_instructions.get(length_choice, length_instructions["Orta"])
    
    return f"{SYSTEM_MSG}\n\n{length_instruction}\n\nBağlam:\n{context}\n\nSoru: {query}\nYanıt:"


def generate_with_gemini(prompt: str, max_tokens: int | None = None) -> str:
    """Gemini'den yanıt üretir"""
    cfg = dict(GENERATION_CFG)
    if max_tokens is not None:
        cfg["max_output_tokens"] = int(max_tokens)
    model = genai.GenerativeModel(GENERATION_MODEL, generation_config=cfg)
    resp = model.generate_content(prompt)
    return (resp.text or "").strip()


def answer_fn(message: str, history: List[Tuple[str, str]], length_choice: str) -> str:
    """Soru-cevap fonksiyonu"""
    try:
        simple_greetings = ["merhaba", "selam", "hello", "hi", "nasılsın", "iyi misin"]
        if message.lower().strip() in simple_greetings:
            return "Merhaba! Yapay Zekâ Dil Modelleri tezi hakkında sorularınızı sorabilirsiniz."
        
        docs = retrieve(message, k=CURRENT_TOP_K)
        print(f"🔍 DEBUG: {len(docs)} belge bulundu")
        
        if not docs:
            return "Bu konu tezde bulunamadı veya sorunuzla yeterince ilgili değil."

        prompt = build_prompt(message, docs, length_choice)
        max_tokens = RESPONSE_LENGTH_TO_TOKENS.get(length_choice, RESPONSE_LENGTH_TO_TOKENS["Orta"])
        answer = generate_with_gemini(prompt, max_tokens=max_tokens)

        if not answer:
            return "Yanıt üretilemedi."

        pages_by_source = {}
        for d in docs:
            m = d.metadata or {}
            display_name = "Yapay Zekâ Dil Modelleri"
            page_display = page_label(m)
            
            if page_display != "?":
                pages_by_source.setdefault(display_name, set()).add(page_display)

        if pages_by_source:
            def sort_key(p: str):
                head = str(p).split("-")[0]
                return int(head) if head.isdigit() else 10**9

            items = []
            for src, pages in pages_by_source.items():
                ordered = ", ".join(sorted(pages, key=sort_key))
                items.append(f"- {src} s. {ordered}")
            
            sources_block = "Kaynak: " + items[0][2:] if len(items) == 1 else "Kaynaklar:\n" + "\n".join(items)
        else:
            sources_block = ""

        if sources_block:
            final_answer = (answer or "Yanıt üretilemedi.").rstrip() + "\n\n" + sources_block
        else:
            final_answer = (answer or "Yanıt üretilemedi.").rstrip()
        
        return final_answer

    except Exception as e:
        print(f"❌ Hata: {e}")
        return f"Hata: {e}"


def debug_metadata():
    """Metadata kontrolü"""
    print("\n" + "="*80)
    print("🔍 METADATA DEBUG")
    print("="*80)
    
    try:
        with open("data/processed_docs.jsonl", "r", encoding="utf-8") as f:
            lines = f.readlines()[:3]
            print(f"\n📄 JSONL: {len(lines)} satır okundu")
    except Exception as e:
        print(f"❌ JSONL hata: {e}")
    
    try:
        df = pd.read_parquet("data/processed_docs.parquet")
        print(f"📊 PARQUET: {len(df)} satır")
    except Exception as e:
        print(f"❌ Parquet hata: {e}")
    
    try:
        if os.path.exists("data/tez.pdf"):
            size_mb = os.path.getsize("data/tez.pdf") / (1024 * 1024)
            print(f"📕 PDF: {size_mb:.2f} MB")
    except Exception as e:
        print(f"❌ PDF hata: {e}")
    
    print("="*80 + "\n")


def auto_ingest_from_repo() -> str:
    """Otomatik veri yükleme"""
    debug_metadata()
    logs = []
    
    try:
        p = "data/processed_docs.jsonl"
        if os.path.exists(p):
            with open(p, "rb") as f:
                result = ingest_jsonl(f)
                logs.append(result)
    except Exception as e:
        logs.append(f"JSONL hata: {e}")

    try:
        p = "data/processed_docs.parquet"
        if os.path.exists(p):
            with open(p, "rb") as f:
                result = ingest_parquet(f)
                logs.append(result)
    except Exception as e:
        logs.append(f"Parquet hata: {e}")

    return "\n".join([lg for lg in logs if lg])


# ==================================================================================================
# 5) GRADIO ARAYÜZÜ
# ==================================================================================================
EXAMPLES = [
    "Tezin temel problem tanımı nedir?",
    "Transformer mimarisinin temel yapıtaşları nelerdir?",
    "Kendine dikkat (self-attention) nasıl çalışır?",
    "RNN/LSTM/GRU'nun karşılaştığı temel sorunlar nelerdir?",
    "GPT ve BERT hangi görevlerde daha başarılıdır?",
]

theme = Base(
    primary_hue=colors.blue,
    secondary_hue=colors.violet,
    text_size=sizes.text_md,
    radius_size=sizes.radius_md,
).set(
    body_background_fill="#f7f9fc",
    block_background_fill="#ffffff",
    border_color_primary="#e3e8f0",
    body_text_color="#0f172a",
)

css = """
.gr-example { background: #f1f5ff !important; color: #0f172a !important; border: 1px solid #dbe2f3 !important; }
.gr-example:hover { background: #e6eeff !important; border-color: #c1cff5 !important; }
input, textarea { color: #0f172a !important; }
::placeholder { color: #6b7280 !important; }
"""

def chat_step(user_message: str, history: list, length_choice: str):
    msg = (user_message or "").strip()
    if not msg:
        return history, ""
    bot_reply = answer_fn(msg, history=history or [], length_choice=length_choice)
    new_history = (history or []) + [(msg, bot_reply)]
    return new_history, ""


with gr.Blocks(title="Yapay Zekâ Dil Modelleri • Tez Asistanı", theme=theme, css=css, fill_height=True) as demo:
    gr.Markdown("### Yapay Zekâ Dil Modelleri — Tez Asistanı")

    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            gr.DownloadButton(label="📄 Tezi İndir (PDF)", value="data/tez.pdf")
            length_choice = gr.Radio(choices=["Kısa", "Orta", "Uzun"], value=DEFAULT_RESPONSE_LENGTH, label="Yanıt uzunluğu")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=500)
            input_box = gr.Textbox(placeholder="Sorunuzu yazın...")
            send_btn = gr.Button("Gönder", variant="primary")

            send_btn.click(chat_step, inputs=[input_box, chatbot, length_choice], outputs=[chatbot, input_box])
            input_box.submit(chat_step, inputs=[input_box, chatbot, length_choice], outputs=[chatbot, input_box])

    _ = auto_ingest_from_repo()


if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)