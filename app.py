# app.py
# ==================================================================================================
# Akıllı Tez Rehberi — RAG Chatbot (LangChain + Chroma + Gemini + Gradio)

# Amaç
# - Bu uygulama, belirli bir kaynak korpustan (bu projede: tek bir lisans tezi) "RAG" (Retrieval-
#   Augmented Generation) yöntemiyle soru-cevap üretir.
# - Kaynak korpus, Kaggle'da hazırlanmış JSONL/Parquet çıktılarıdır. Uygulama açılışında bu
#   dosyaları otomatik olarak yükler ("auto-ingest") ve Chroma vektör veritabanına kaydeder.
# - Kullanıcı, Gradio arayüzünden soru yazar. Sistem, soruya en çok benzeyen metin parçalarını
#   bulur ("retrieve"), bu bağlamı bir istem (prompt) ile Gemini 2.0 modeline verir ve cevabı üretir.
# - Cevabın sonunda "Kaynaklar" başlığı altında hangi dosya/sayfa(lar)dan yararlanıldığı listelenir.

# Tasarım İlkeleri
# - Veri, repo içindeki data/ klasöründen otomatik yüklenir.
# - Her adım (ingest → retrieve → prompt → generate → answer) açık ve izlenebilir şekilde
#   kodlanmıştır

# Ortam Değişkenleri (HF Spaces → Settings → Variables and secrets):
# - SECRET: GOOGLE_API_KEY        → Gemini API anahtarı (zorunlu)
# - VAR   : EMBEDDINGS_MODEL      → trmteb/turkish-embedding-model
# - VAR   : GENERATION_MODEL      → gemini-2.0-flash (erişim yoksa gemini-1.5-flash)
# - VAR   : CHROMA_PERSIST_DIR    → .chroma (Chroma'nın kalıcı dizini; Space yeniden açıldığında korunur)
#
# Gerekli Paketler (requirements.txt)
# ==================================================================================================

import os
import re
import json
from typing import List, Tuple

# 1) LLM: Gemini (Google Generative AI)
#    - Sadece yanıt üretiminde kullanılır. Retrieval/embedding aşamaları Space'in container'ında çalışır.
import google.generativeai as genai

# 2) Vektör DB ve Embedding
#    - LangChain-Chroma: Chroma'nın resmi LangChain paketidir (deprecation sorunları yaşamaz).
#    - LangChain-HuggingFace: SentenceTransformers tabanlı gömlemeler için güncel paket.
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 3) Dosya okuma (yalnızca JSONL/Parquet)
import pandas as pd

# 4) Web arayüzü
import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, sizes


# --------------------------------------------------------------------------------------------------
# 0) Ortam değişkenleri ve LLM yapılandırması
#    - HF Spaces'te Secrets/Variables ile gelir; yerelde .env üzerinden verilebilir.
# --------------------------------------------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "trmteb/turkish-embedding-model")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-2.0-flash")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", ".chroma")

# STRICT RAG: açık‑alan fallback kapalı (yalnızca tezden yanıt ver).
ALLOW_OPEN_DOMAIN_FALLBACK = os.getenv("ALLOW_OPEN_DOMAIN_FALLBACK", "false").lower() == "true"

# Sayfa filtreleme: PDF sayfa 13-104 arası tez içeriği (1-12: ön sayfalar, 105+: kaynakça/ekler)
PDF_PAGE_START = int(os.getenv("PDF_PAGE_START", "13"))   # DÜN ÇALIŞAN DEĞER
PDF_PAGE_END = int(os.getenv("PDF_PAGE_END", "104"))     # DÜN ÇALIŞAN DEĞER
PDF_TO_THESIS_OFFSET = int(os.getenv("PDF_TO_THESIS_OFFSET", "-12"))  # DÜN ÇALIŞAN DEĞER

# Gemini istemcisi; API anahtarı zorunludur.
genai.configure(api_key=GOOGLE_API_KEY)

# Üretim (generate) parametreleri:
# - temperature: RAG'de düşük tutulur (0.0–0.3 aralığı), kaynağa sadakat artar.
# - top_p/top_k: Örnekleme çeşitliliği; varsayılanlar korunur, gerekirse ayarlanır.
# - max_output_tokens: Cevabın üst uzunluğu; kesilme yaşanırsa artırılabilir (ör. 768/1024).
GENERATION_CFG = dict(
    temperature=0.25,
    top_p=0.95,
    top_k=40,
    max_output_tokens=1024
)

# Yanıt uzunluğu ön ayarları (STRICT RAG) - GERÇEKTEN FARK EDECEK DEĞERLER
RESPONSE_LENGTH_TO_TOKENS = {
    "Kısa": 400,    # Çok kısa, özet yanıt
    "Orta": 800,    # Orta uzunluk, detaylı yanıt
    "Uzun": 1500    # Çok uzun, kapsamlı yanıt
}
DEFAULT_RESPONSE_LENGTH = os.getenv("DEFAULT_RESPONSE_LENGTH", "Orta")
CURRENT_TOP_K = int(os.getenv("CURRENT_TOP_K", "10"))


# --------------------------------------------------------------------------------------------------
# 1) Metin yardımcıları
#    - clean_text: metni biçimsel olarak normalize eder.
#    - split_into_chunks: uzun metni kaygan pencere (sliding window) ile küçük parçalara böler.
# --------------------------------------------------------------------------------------------------
def clean_text(s: str) -> str:
    """
    Metin temizliği:
    - \x00 gibi bozuk karakterleri kaldırır.
    - Birden fazla boşluk/tab'ı tek boşluğa indirir.
    - Çoklu boş satırları sadeleştirir (3+ → 2).
    Bu temizleme, embedding kalitesini ve okunabilirliği iyileştirir.
    """
    s = (s or "").replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def split_into_chunks(text: str, size: int = 800, overlap: int = 120) -> List[str]:
    """
    Metni kelime bazlı parçalara böler.
    Parametreler:
      - size: hedef parça uzunluğu (kelime)
      - overlap: art arda gelen parçalar arasındaki ortak kelime sayısı
    Not:
      - RAG'de 512–800 kelime iyi bir başlangıç aralığıdır; overlap 80–120 önerilir.
    """
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
    """
    Sayfa numarasının geçerli aralıkta olup olmadığını kontrol eder.
    Sadece tez içeriğinin bulunduğu sayfaları kabul eder.
    İçindekiler, kaynaklar gibi bölümler filtrelenir.
    """
    return PDF_PAGE_START <= page_num <= PDF_PAGE_END


def pdf_to_thesis_page(pdf_page: int) -> int:
    """
    PDF sayfa numarasını tez sayfa numarasına çevirir.
    Örnek: PDF sayfa 13 → Tez sayfa 1 (offset -12 ile)
    """
    return pdf_page + PDF_TO_THESIS_OFFSET


# --------------------------------------------------------------------------------------------------
# 2) Embedding sağlayıcısı ve Chroma vektör veritabanı
#    - Embedding: Türkçe için SentenceTransformers modeli (HF üzerinden çekilir).
#    - Chroma: .chroma klasörüne kalıcı olarak yazar (Space yeniden başlasa da veri korunur).
# --------------------------------------------------------------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
vectorstore = Chroma(
    client=chroma_client,
    collection_name="docs",
    embedding_function=embeddings
)


# --------------------------------------------------------------------------------------------------
# 3) Ingest fonksiyonları
#    - JSONL: her satır bir kaydı temsil eder → {"content": "...", "meta": {...}}
#    - Parquet: tablo formatı; "content" zorunlu, "meta" opsiyonel (dict) ya da sütunlardan derlenir.
#    - Sayfa filtreleme: sadece PDF sayfa 13-104 arasındaki parçalar ingest edilir.
# --------------------------------------------------------------------------------------------------
def ingest_jsonl(file_obj) -> str:
    """
    JSONL dosyasını satır satır okuyup metin + metadata çıkarır ve Chroma'ya ekler.
    Beklenen satır yapısı:
      {"content": "metin parçası...", "meta": {"source":"tez.pdf", "page": 12}}
    Sayfa filtreleme: Sadece PDF_PAGE_START ile PDF_PAGE_END arasındaki sayfalar işlenir
    Dönüş:
      - İşlenen parça sayısını belirten durum mesajı
    """
    try:
        lines = file_obj.read().decode("utf-8").splitlines()
        texts, metas = [], []
        for ln in lines:
            row = json.loads(ln)
            content = clean_text(row.get("content", ""))
            if not content:
                continue
            meta = row.get("meta", {}) or {}
            
            # Sayfa filtreleme: sadece geçerli sayfaları işle
            page_num = meta.get("page")
            if page_num and not is_valid_page(int(page_num)):
                continue
                
            texts.append(content)
            metas.append(meta)
        if not texts:
            return "JSONL boş ya da geçerli kayıt bulunamadı."
        vectorstore.add_texts(texts=texts, metadatas=metas)
        return f"JSONL ingest tamamlandı: {len(texts)} parça eklendi."
    except Exception as e:
        return f"JSONL ingest hatası: {e}"


def ingest_parquet(file_obj) -> str:
    """
    Parquet dosyasını okuyup "content" ve (varsa) "meta" bilgilerini alır ve Chroma'ya ekler.
    Kurallar:
      - 'content' sütunu zorunludur (string)
      - 'meta' sütunu yoksa title/source/page/page_start/page_end sütunlarından metadata derlenir
      - Sayfa filtreleme: Sadece geçerli sayfa aralığındaki kayıtlar işlenir
    """
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
            
            # Sayfa filtreleme: sadece geçerli sayfaları işle
            page_num = meta.get("page")
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


# --------------------------------------------------------------------------------------------------
# 4) Retrieval ve Prompt oluşturma
#    - retrieve: Sorguya en çok benzeyen k metin parçasını Chroma'dan getirir.
#    - build_prompt: Sistem talimatı + bağlam + soru birleşiminden Gemini istemini kurar.
#    - generate_with_gemini: Yanıt üretir; metin döndürür.
# --------------------------------------------------------------------------------------------------
SYSTEM_MSG = (
    "Aşağıdaki bağlam parçalarını kullanarak yanıt ver. "
    "Kaynaklarda yoksa 'Bilmiyorum' de. "
    "Yanıtı tam cümlelerle bitir; maddeleri yarım bırakma."
)


def retrieve(query: str, k: int):
    """
    Sorgu embedding'i ile Chroma'dan en ilgili k belge parçasını getirir.
    Not:
      - Bazı sürümlerde 'similarity_search_with_relevance_scores' olmayabilir; bu durumda
        klasik 'similarity_search' ile geri düşer.
    """
    try:
        results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
        docs = [doc for doc, _score in results]
        return docs
    except Exception:
        docs = vectorstore.similarity_search(query, k=k)
        return docs


def build_prompt(query: str, docs, length_choice: str) -> str:
    """
    Gemini'ye verilecek istem (prompt):
      - Sistem talimatı (modelin davranış sınırları)
      - Bağlam: numaralı satırlar; kaynak adı ve sayfa bilgisi görünür
      - Kullanıcı sorusu
      - Yanıt uzunluğu talimatı
    """
    # Yanıt uzunluğu talimatı
    length_instructions = {
        "Kısa": "Kısa ve öz bir yanıt ver. Sadece temel bilgileri belirt.",
        "Orta": "Detaylı ama özlü bir yanıt ver. Önemli noktaları açıkla.",
        "Uzun": "Kapsamlı ve detaylı bir yanıt ver. Tüm ilgili bilgileri, örnekleri ve açıklamaları dahil et."
    }
    
    ctx_lines = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        page = meta.get("page", "?")
        
        # PDF sayfa numarasını tez sayfa numarasına dönüştür
        try:
            pdf_page_int = int(page)
            thesis_page = pdf_to_thesis_page(pdf_page_int)
            page_display = f"syf. {thesis_page}"
        except (ValueError, TypeError):
            page_display = f"syf. {page}"
            
        ctx_lines.append(f"[{i}] ({src} {page_display}) {d.page_content}")
    
    context = "\n\n".join(ctx_lines) if ctx_lines else "(bağlam yok)"
    length_instruction = length_instructions.get(length_choice, length_instructions["Orta"])
    
    return f"{SYSTEM_MSG}\n\n{length_instruction}\n\nBağlam:\n{context}\n\nSoru: {query}\nYanıt:"


def generate_with_gemini(prompt: str, max_tokens: int | None = None) -> str:
    """
    Gemini'den yanıt üretir ve düz metin olarak döndürür.
    Hata durumları üst katmanda yakalanır; burada sadece model çağrısı yapılır.
    """
    cfg = dict(GENERATION_CFG)
    if max_tokens is not None:
        cfg["max_output_tokens"] = int(max_tokens)
    model = genai.GenerativeModel(GENERATION_MODEL, generation_config=cfg)
    resp = model.generate_content(prompt)
    return (resp.text or "").strip()


def polish_style(raw_answer: str) -> str:
    """
    Üslup parlatma: Bilgi içeriğine ve 'Kaynak/Kaynaklar' bloğuna dokunmadan
    yalnızca anlatım akıcılığını iyileştirir. Yeni bilgi ekleme/çıkarma yapmaz.
    """
    if not raw_answer:
        return raw_answer
    prompt = (
        "Aşağıdaki cevabın yalnızca anlatımını akıcı, doğal ve tutarlı hale getir. "
        "Yeni bilgi ekleme, çıkarma yapma. 'Kaynak' veya 'Kaynaklar' bloğuna dokunma.\n\n"
        "Cevap:\n" + raw_answer
    )
    polished = generate_with_gemini(prompt, max_tokens=min(len(raw_answer) + 200, 1600))
    return polished or raw_answer


def answer_fn(message: str, history: List[Tuple[str, str]], length_choice: str) -> str:
    """
    ChatInterface tarafından çağrılır.
    STRICT RAG:
      - Bağlam yoksa: "Bu konu tezde bulunamadı." (kaynak yazma)
      - Bağlam varsa: RAG yanıt + Kaynaklar (tez sayfa numarasına dönüştürülmüş)
    """
    try:
        # Basit selamlama ve tez dışı sorular için kontrol
        simple_greetings = ["merhaba", "selam", "hello", "hi", "nasılsın", "iyi misin"]
        if message.lower().strip() in simple_greetings:
            return "Merhaba! Yapay Zekâ Dil Modelleri tezi hakkında sorularınızı sorabilirsiniz."
        
        docs = retrieve(message, k=CURRENT_TOP_K)
        
        if not docs:
            return "Bu konu tezde bulunamadı."

        prompt = build_prompt(message, docs, length_choice)
        max_tokens = RESPONSE_LENGTH_TO_TOKENS.get(length_choice, RESPONSE_LENGTH_TO_TOKENS["Orta"])
        answer = generate_with_gemini(prompt, max_tokens=max_tokens)

        # Kaynakları sade ve tekilleştirilmiş biçimde göster (DÜN ÇALIŞAN FORMAT)
        pages_by_source = {}
        for d in docs:
            m = d.metadata or {}
            display_name = "Yapay Zekâ Dil Modelleri"
            pdf_page = m.get("page", "?")
            try:
                pdf_page_int = int(pdf_page)
                thesis_page = pdf_to_thesis_page(pdf_page_int)
                pages_by_source.setdefault(display_name, set()).add(str(thesis_page))
            except (ValueError, TypeError):
                pages_by_source.setdefault(display_name, set()).add(str(pdf_page))

        def sort_key(p: str):
            head = str(p).split("-")[0]
            return int(head) if head.isdigit() else 10**9

        items = []
        for src, pages in pages_by_source.items():
            ordered = ", ".join(sorted(pages, key=sort_key))
            items.append(f"- {src} syf. {ordered}")
        
        sources_block = "Kaynak: " + items[0][2:] if len(items) == 1 else "Kaynaklar:\n" + "\n".join(items)

        combined = (answer or "Yanıt üretilemedi.").rstrip() + "\n\n" + sources_block
        final_answer = polish_style(combined) or combined
        
        return final_answer

    except Exception as e:
        return f"Hata: {e}"


# --------------------------------------------------------------------------------------------------
# 5) Otomatik ingest (deploy esnasında hiçbir kullanıcı aksiyonu gerektirmeden veri yükler)
#    - data/processed_docs.jsonl
#    - data/processed_docs.parquet
# --------------------------------------------------------------------------------------------------
def auto_ingest_from_repo() -> str:
    """
    Uygulama başlarken veri klasöründeki dosyaları ingest eder.
    Not:
      - Aynı koleksiyona tekrar tekrar ingest etmeyi engellemek için ileride "idempotent"
        bir kontrol (ör. koleksiyon boş mu?) eklenebilir. MVP'de gerek görülmemiştir.
    """
    logs = []
    try:
        p = "data/processed_docs.jsonl"
        if os.path.exists(p):
            with open(p, "rb") as f:
                result = ingest_jsonl(f)
                logs.append(result)
        else:
            print("❌ JSONL dosyası bulunamadı!")
    except Exception as e:
        logs.append(f"AUTO JSONL hata: {e}")

    try:
        p = "data/processed_docs.parquet"
        if os.path.exists(p):
            with open(p, "rb") as f:
                result = ingest_parquet(f)
                logs.append(result)
        else:
            print("❌ Parquet dosyası bulunamadı!")
    except Exception as e:
        logs.append(f"AUTO Parquet hata: {e}")

    final_result = "\n".join([lg for lg in logs if lg])
    return final_result


# --------------------------------------------------------------------------------------------------
# 6) Gradio arayüzü
#    - Bu sürümde dosya yükleme kapalıdır; veri açılışta otomatik yüklenir.
#    - Sol panel: Tez indirme + estetik içindekiler + yanıt uzunluğu seçimi
#    - Sağ panel: Sohbet arayüzü
# --------------------------------------------------------------------------------------------------
EXAMPLES = [
    "Tezin temel problem tanımı nedir?",
    "Transformer mimarisinin temel yapıtaşları nelerdir?",
    "Kendine dikkat (self-attention) nasıl çalışır?",
    "RNN/LSTM/GRU'nun karşılaştığı temel sorunlar nelerdir?",
    "GPT ve BERT hangi görevlerde daha başarılıdır?",
    "Temel NLP teknikleri nelerdir?",
    "Yapay zekâ nasıl tanımlanır? Kapsadığı alt alanlar nelerdir?",
    "Çok modlu modellerin öne çıkan örnekleri hangileri?",
    "Etik bölümünde hangi riskler tartışılıyor?",
    "Gelecek çalışmalar için öneriler nelerdir?",
    "Tezi bana anlatır mısın?",  # YENİ EKLENEN SORU
]

# Tema: açık, yüksek okunabilirlik ve sade anahtarlar (Gradio ile uyumlu)
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
    block_title_text_color="#0b1220",
    link_text_color="#1d4ed8",
    button_primary_background_fill="#2563eb",
    button_primary_text_color="#ffffff",
    input_background_fill="#ffffff",
    input_border_color="#cbd5e1",
)

# CSS (estetik kartlar ve diğer stiller)
css = """
.gr-example {
  background: #f1f5ff !important;
  color: #0f172a !important;
  border: 1px solid #dbe2f3 !important;
}
.gr-example:hover {
  background: #e6eeff !important;
  border-color: #c1cff5 !important;
}
input, textarea, .gr-text-input input, .gr-textbox textarea {
  color: #0f172a !important;
}
::placeholder {
  color: #6b7280 !important;
}
button.primary:hover {
  background-color: #1d4ed8 !important;
}
/* Sohbet balonları */
.gr-chatbot .message.user { background:#eef2ff !important; border-color:#dbe2f3 !important; }
.gr-chatbot .message.bot  { background:#ffffff !important; border-color:#e3e8f0 !important; }
/* İçindekiler — estetik kartlar */
.toc-container { 
  display: flex; 
  flex-direction: column; 
  gap: 8px; 
  margin: 10px 0; 
}
.toc-card {
  padding: 12px 14px;
  border-radius: 8px;
  background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
  border: 1px solid #e2e8f0;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  transition: all 0.2s ease;
}
.toc-card:hover {
  border-color: #3b82f6;
  box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
  transform: translateY(-1px);
}
.toc-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: 600;
  color: #1e293b;
  margin-bottom: 4px;
}
.toc-page {
  color: #64748b;
  font-size: 0.9em;
  font-weight: 500;
}
.toc-subtitle {
  color: #475569;
  font-size: 0.85em;
  line-height: 1.4;
  margin-top: 6px;
}
"""

# ----- Basit sohbet adımı: kullanıcı mesajı → yanıt; input'u temizle -----
def chat_step(user_message: str, history: list[tuple[str, str]], length_choice: str):
    msg = (user_message or "").strip()
    if not msg:
        return history, ""
    bot_reply = answer_fn(msg, history=history or [], length_choice=length_choice)
    new_history = (history or []) + [(msg, bot_reply)]
    return new_history, ""


with gr.Blocks(title="Yapay Zekâ Dil Modelleri • Kaynaklı Soru‑Cevap", theme=theme, css=css, fill_height=True) as demo:
    # Ana başlık
    gr.Markdown(
        """
        <div style="padding:10px 0 4px 0;">
          <h2 style="margin:0;color:#0b1220;">Yapay Zekâ Dil Modelleri — Tez Asistanı</h2>
          <div style="color:#334155">
            Bu arayüz, 'Yapay Zekâ Dil Modelleri' tezi temel alınarak sorularınıza yanıt verir; ilgili pasajları bulur ve kaynak sayfalarıyla birlikte sunar.
          </div>
        </div>
        """,
    )

    with gr.Row():
        # Sol panel: Tez indirme + estetik içindekiler + yanıt uzunluğu seçimi
        with gr.Column(scale=1, min_width=400):
            gr.Markdown("### 📄 Tez Dokümanı")
            gr.DownloadButton(label="📄 Tezi İndir (PDF)", value="data/tez.pdf")

            gr.Markdown("### 📚 İçindekiler")
            gr.HTML(
                """
                <div class="toc-container">
                  <div class="toc-card">
                    <div class="toc-header">
                      <span>1. GİRİŞ</span>
                      <span class="toc-page">syf. 1</span>
                    </div>
                  </div>
                  <div class="toc-card">
                    <div class="toc-header">
                      <span>2. YAPAY ZEKÂ VE DOĞAL DİL İŞLEME</span>
                      <span class="toc-page">syf. 2</span>
                    </div>
                  </div>
                  <div class="toc-card">
                    <div class="toc-header">
                      <span>3. DİL MODELLEMEDE ML ve DL</span>
                      <span class="toc-page">syf. 15</span>
                    </div>
                  </div>
                  <div class="toc-card">
                    <div class="toc-header">
                      <span>4. DİL MODELLERİ</span>
                      <span class="toc-page">syf. 31</span>
                    </div>
                  </div>
                  <div class="toc-card">
                    <div class="toc-header">
                      <span>5. TRANSFORMER TABANLI MODELLER</span>
                      <span class="toc-page">syf. 41</span>
                    </div>
                  </div>
                  <div class="toc-card">
                    <div class="toc-header">
                      <span>6. GÜNCEL YÖNELİMLER ve ETİK</span>
                      <span class="toc-page">syf. 71</span>
                    </div>
                  </div>
                  <div class="toc-card">
                    <div class="toc-header">
                      <span>7. SONUÇ ve DEĞERLENDİRME</span>
                      <span class="toc-page">syf. 92</span>
                    </div>
                  </div>
                </div>
                """
            )

            # Yanıt uzunluğu seçimi
            length_choice = gr.Radio(
                choices=["Kısa", "Orta", "Uzun"], 
                value=DEFAULT_RESPONSE_LENGTH, 
                label="Yanıt uzunluğu"
            )

        # Sağ panel: Sohbet arayüzü
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=500, avatar_images=(None, None))
            input_box = gr.Textbox(placeholder="Sorunuzu yazın ve Enter'a basın...", scale=1)
            send_btn = gr.Button("Gönder", variant="primary")

            # Yanıt uzunluğu seçimini chat_step'e parametre olarak geç
            send_btn.click(
                chat_step, 
                inputs=[input_box, chatbot, length_choice], 
                outputs=[chatbot, input_box]
            )
            input_box.submit(
                chat_step, 
                inputs=[input_box, chatbot, length_choice], 
                outputs=[chatbot, input_box]
            )

            with gr.Row():
                for q in EXAMPLES:
                    gr.Button(q, variant="secondary", scale=1).click(
                        lambda s=q: s, outputs=input_box
                    ).then(
                        chat_step, 
                        inputs=[input_box, chatbot, length_choice], 
                        outputs=[chatbot, input_box]
                    )

    # Uygulama açılışında otomatik ingest (logu UI'da göstermiyoruz)
    _ = auto_ingest_from_repo()


# Yerel geliştirme için (HF Spaces'te launch çağrısı gerekmez)
if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)