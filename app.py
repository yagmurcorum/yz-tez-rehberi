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
PDF_PAGE_START = int(os.getenv("PDF_PAGE_START", "13"))
PDF_PAGE_END = int(os.getenv("PDF_PAGE_END", "104"))
PDF_TO_THESIS_OFFSET = int(os.getenv("PDF_TO_THESIS_OFFSET", "0"))

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

# Yanıt uzunluğu ön ayarları (STRICT RAG)
RESPONSE_LENGTH_TO_TOKENS = {
    "Kısa": 200,
    "Orta": 800,
    "Uzun": 1500
}
DEFAULT_RESPONSE_LENGTH = os.getenv("DEFAULT_RESPONSE_LENGTH", "Orta")
CURRENT_TOP_K = int(os.getenv("CURRENT_TOP_K", "5"))


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


# YENİ: Basit tokenizasyon ve sorgudan anahtar çıkarımı (genel; sayfaya/konuya özgü değil)
def tokenize_for_keywords(text: str) -> list[str]:
    """
    YENİ:
    - Küçük harfe indir, TR karakterleri basit normalize et
    - Harf/rakam dışını boşlukla değiştir
    - 1 karakterlik parçaları ele
    """
    txt = (text or "").lower()
    tr_map = str.maketrans("çğıöşüâîû", "cgiosuaiu")
    txt = txt.translate(tr_map)
    txt = re.sub(r"[^a-z0-9ğüşıöç ]", " ", txt)
    tokens = [t for t in txt.split() if len(t) > 1]
    return tokens


# GÜNCELLEME: Sadece sorgudan türeyen anahtarlar (stop-words hariç). Sayfaya/konuya özgü değil.
def build_query_keywords(query: str) -> set[str]:
    tokens = tokenize_for_keywords(query)
    stop = {
        "ve","ile","mi","nedir","nelerdir","hangi","temel","alan","olarak","da","de","bir","icin",
        "nasil","ne","kim","neydi","neye","hakkinda","uzerine","ileti","olan","midir","midirki"
    }
    return {t for t in tokens if t not in stop and len(t) > 2}


# YENİ: Sorgudan basit ikili ifadeler (bigram) üret (genel eşleşmeyi güçlendirir)
def build_query_bigrams(query: str) -> set[str]:
    toks = [t for t in tokenize_for_keywords(query) if len(t) > 2]
    return {" ".join([toks[i], toks[i + 1]]) for i in range(len(toks) - 1)}


def is_valid_page(page_num: int) -> bool:
    """
    Sayfa filtreleme: PDF sayfa 13-104 arası tez içeriği
    (1-12: ön sayfalar, 105+: kaynakça/ekler)
    """
    return PDF_PAGE_START <= page_num <= PDF_PAGE_END


def pdf_to_thesis_page(pdf_page: int) -> int:
    """
    PDF sayfa numarasını tez sayfa numarasına çevirir.
    Örnek: PDF sayfa 1 → Tez sayfa 1 (offset 0 ile)
    """
    return pdf_page + PDF_TO_THESIS_OFFSET


# --------------------------------------------------------------------------------------------------
# 2) Embedding sağlayıcısı ve Chroma vektör veritabanı
#    - Embedding: Türkçe için SentenceTransformers modeli (HF üzerinden çekilir).
#    - Chroma: .chroma klasörüne kalıcı olarak yazar (Space yeniden başlasa da veri korunur).
# --------------------------------------------------------------------------------------------------
# YENİ: CHROMA dizinini garantiye al
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

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
            
            page_num = meta.get("page_start")
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


# --------------------------------------------------------------------------------------------------
# 4) Retrieval ve Prompt oluşturma
#    - retrieve: Sorguya en çok benzeyen k metin parçasını Chroma'dan getirir.
#    - build_prompt: Sistem talimatı + bağlam + soru birleşiminden Gemini istemini kurar.
#    - generate_with_gemini: Yanıt üretir; metin döndürür.
# --------------------------------------------------------------------------------------------------
SYSTEM_MSG = (
    "Aşağıdaki bağlam parçalarını kullanarak yanıt ver. "
    "Bağlamda verilen TÜM ilgili bilgileri, tarihleri, isimleri ve detayları MUTLAKA dahil et. "
    "Eğer bağlamda yeterli bilgi yoksa: 'Bu konu tezde yeterli detayla bulunamadı.' de. "
    "Yanıtı tam cümlelerle bitir; maddeleri yarım bırakma."
)


def retrieve(query: str, k: int):
    """
    Sorgu embedding'i ile Chroma'dan en ilgili k belge parçasını getirir.
    GÜNCELLEME:
    - Skorlar distance/negatif olabilir → 0..1 arası benzerliğe normalize edilir.
    - Leksikal yeniden sıralama (re-rank) tüm sorgular için uygulanır.
    """
    try:
        # Daha zengin havuz için 2x sonuç çek (min 8)
        results = vectorstore.similarity_search_with_score(query, k=max(k * 2, 8))
        # results: List[(Document, score)]  -> score çoğunlukla distance (küçük = iyi)
        query_keywords = build_query_keywords(query)
        query_bigrams  = build_query_bigrams(query)

        raw_items = []
        max_hits = 1
        for doc, raw_score in results:
            # 1) Embedding skorunu 0..1 benzerliğe çevir
            try:
                if raw_score is None:
                    emb_sim = 0.5
                elif float(raw_score) >= 0:
                    emb_sim = 1.0 / (1.0 + float(raw_score))   # distance -> similarity
                else:
                    emb_sim = 1.0 / (1.0 + abs(float(raw_score)))
            except Exception:
                emb_sim = 0.5

            # 2) Leksikal eşleşme (kelime + bigram)
            text = (doc.page_content or "").lower()
            text_tokens = tokenize_for_keywords(text)
            word_hits   = sum(1 for t in text_tokens if t in query_keywords)
            phrase_hits = sum(1 for bg in query_bigrams if bg in text)
            hits = word_hits + 2 * phrase_hits
            max_hits = max(max_hits, hits)

            raw_items.append((doc, emb_sim, hits))

        # 3) Birleştir ve sırala
        reranked = []
        for doc, emb_sim, hits in raw_items:
            lexical_norm = hits / max_hits if max_hits > 0 else 0.0
            combined = 0.85 * emb_sim + 0.15 * lexical_norm
            reranked.append((combined, doc))

        reranked.sort(key=lambda x: x[0], reverse=True)
        docs = [doc for _score, doc in reranked[:k]]
        if not docs:
            return vectorstore.similarity_search(query, k=k)
        return docs
    except Exception:
        return vectorstore.similarity_search(query, k=k)


def page_label(meta: dict) -> str:
    """
    Metadata'dan sayfa etiketini çıkarır.
    Öncelik sırası: page_label > logical_page > page_start/page_end > page (offset ile)
    """
    if meta.get("page_label"):
        return str(meta["page_label"])
    if meta.get("logical_page"):
        return str(meta["logical_page"])
    if meta.get("page_start") and meta.get("page_end"):
        start = str(meta["page_start"])
        end = str(meta["page_end"])
        return start if start == end else f"{start}-{end}"
    if meta.get("page_start"):
        return str(meta["page_start"])
    if meta.get("page") is not None:
        try:
            return str(int(meta["page"]) + PDF_TO_THESIS_OFFSET)
        except Exception:
            return str(meta["page"])
    return "?"


def build_prompt(query: str, docs, length_choice: str) -> str:
    """
    Gemini'ye verilecek istem (prompt):
      - Sistem talimatı (modelin davranış sınırları)
      - Bağlam: numaralı satırlar; kaynak adı ve sayfa bilgisi görünür
      - Kullanıcı sorusu
      - Yanıt uzunluğu talimatı
    """
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
    """
    Gemini'den yanıt üretir ve düz metin olarak döndürür.
    """
    cfg = dict(GENERATION_CFG)
    if max_tokens is not None:
        cfg["max_output_tokens"] = int(max_tokens)
    model = genai.GenerativeModel(GENERATION_MODEL, generation_config=cfg)
    resp = model.generate_content(prompt)
    return (resp.text or "").strip()


def answer_fn(message: str, history: List[Tuple[str, str]], length_choice: str) -> str:
    """
    ChatInterface tarafından çağrılır.
    RAG pipeline: retrieve → build_prompt → generate → format_response
    GÜNCELLEME:
    - "Tezde bulunamadı" veya "yanıt üretilemedi" durumlarında kaynak/uyarı GÖSTERİLMEZ.
    - Uyarı yalnızca kaynak bloğu varsa eklenir.
    """
    try:
        simple_greetings = ["merhaba", "selam", "hello", "hi", "nasılsın", "iyi misin"]
        if message.lower().strip() in simple_greetings:
            return "Merhaba! Yapay Zekâ Dil Modelleri tezi hakkında sorularınızı sorabilirsiniz."

        docs = retrieve(message, k=CURRENT_TOP_K)
        if not docs:
            return "Bu konu tezde bulunamadı veya sorunuzla yeterince ilgili değil."

        prompt = build_prompt(message, docs, length_choice)
        max_tokens = RESPONSE_LENGTH_TO_TOKENS.get(length_choice, RESPONSE_LENGTH_TO_TOKENS["Orta"])
        answer = generate_with_gemini(prompt, max_tokens=max_tokens)
        if not answer:
            return "Yanıt üretilemedi."

        # Model "bulunamadı" vb. diyorsa aynen döndür; kaynak/uyarı ekleme.
        low_answer = answer.lower()
        if ("bulunamadı" in low_answer) or ("yeterli detay" in low_answer):
            return answer

        # Kaynak sayfaları topla
        pages_by_source = {}
        for d in docs:
            m = d.metadata or {}
            display_name = "Yapay Zekâ Dil Modelleri"
            page_display = page_label(m)
            if page_display != "?":
                pages_by_source.setdefault(display_name, set()).add(page_display)

        # Kaynak bloğu
        if pages_by_source:
            def sort_key(p: str):
                head = str(p).split("-")[0]
                return int(head) if head.isdigit() else 10**9
            items = []
            for src, pages in pages_by_source.items():
                ordered = ", ".join(sorted(pages, key=sort_key))
                items.append(f"- {src} s. {ordered}")
            sources_block = "Kaynak: " + items[0][2:] if len(items) == 1 else "Kaynaklar:\n" + "\n".join(items)

            # Yalnızca kaynak bloğu varsa uyarı ekle
            warning_note = "\n\nℹ️ Bu yanıt birden fazla sayfadan derlenmiştir. Tam bilgi için kaynak sayfalara göz atın."
            final_answer = (answer or "Yanıt üretilemedi.").rstrip() + "\n\n" + sources_block + warning_note
            return final_answer

        # Kaynak yoksa sadece yanıtı döndür
        return (answer or "Yanıt üretilemedi.").rstrip()

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
    """
    logs = []
    try:
        p = "data/processed_docs.jsonl"
        if os.path.exists(p):
            with open(p, "rb") as f:
                result = ingest_jsonl(f)
                logs.append(result)
    except Exception as e:
        logs.append(f"AUTO JSONL hata: {e}")

    try:
        p = "data/processed_docs.parquet"
        if os.path.exists(p):
            with open(p, "rb") as f:
                result = ingest_parquet(f)
                logs.append(result)
    except Exception as e:
        logs.append(f"AUTO Parquet hata: {e}")

    # YENİ: ingest sonrası koleksiyon sayımı logla (HF Spaces konsolu için faydalı)
    try:
        cnt = vectorstore._collection.count()
        logs.append(f"[Chroma] persist_dir={CHROMA_PERSIST_DIR} | collection='docs' | count={cnt}")
    except Exception as e:
        logs.append(f"[Chroma] count okunamadı: {e}")

    return "\n".join([lg for lg in logs if lg])


# --------------------------------------------------------------------------------------------------
# 6) Gradio arayüzü
#    - Bu sürümde dosya yükleme kapalıdır; veri açılışta otomatik yüklenir.
#    - Sol panel: Tez indirme + estetik içindekiler + yanıt uzunluğu seçimi
#    - Sağ panel: Sohbet arayüzü
#    Yazar/Danışman/Kapsam başlıkta kalıcı gösterilir.
# --------------------------------------------------------------------------------------------------
EXAMPLES = [
    "Tezin temel problem tanımı nedir?",
    "Transformer mimarisinin temel yapıtaşları nelerdir?",
    "Kendine dikkat (self-attention) nasıl çalışır?",
    "RNN/LSTM/GRU'nun karşılaştığı temel sorunlar nelerdir?",
    "GPT ve BERT hangi görevlerde daha başarılıdır?",
    "Temel doğal dil işleme teknikleri nelerdir?",
    "Yapay zekâ nasıl tanımlanır? Kapsadığı alt alanlar nelerdir?",
    "Çok modlu modellerin öne çıkan örnekleri hangileri?",
    "Etik bölümünde hangi riskler tartışılıyor?",
    "Gelecek çalışmalar için öneriler nelerdir?",
    "Tezi bana anlatır mısın?",
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
.gr-chatbot .message.user { background:#eef2ff !important; border-color:#dbe2f3 !important; }
.gr-chatbot .message.bot  { background:#ffffff !important; border-color:#e3e8f0 !important; }
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

def chat_step(user_message: str, history: list[tuple[str, str]], length_choice: str):
    """
    Kullanıcı mesajını işleyip chatbot yanıtını döndürür.
    """
    msg = (user_message or "").strip()
    if not msg:
        return history, ""
    bot_reply = answer_fn(msg, history=history or [], length_choice=length_choice)
    new_history = (history or []) + [(msg, bot_reply)]
    return new_history, ""


with gr.Blocks(title="Yapay Zekâ Dil Modelleri • Kaynaklı Soru‑Cevap", theme=theme, css=css, fill_height=True) as demo:
    gr.Markdown(
        """
        <div style="padding:10px 0 4px 0;">
          <h2 style="margin:0;color:#0b1220;">Yapay Zekâ Dil Modelleri — Tez Asistanı</h2>
          <div style="color:#334155;margin-top:8px;">
            Bu arayüz, 'Yapay Zekâ Dil Modelleri' tezi temel alınarak sorularınıza yanıt verir; ilgili pasajları bulur ve kaynak sayfalarıyla birlikte sunar.
          </div>
          <div style="color:#64748b;font-size:0.9em;margin-top:8px;">
            📝 <strong>Yazar:</strong> Yağmur ÇORUM |
            👨‍🏫 <strong>Danışman:</strong> Prof. Dr. Burak ORDİN |
            📚 <strong>Kapsam:</strong> Bölüm 1-7 (Ana İçerik)
          </div>
        </div>
        """,
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=400):
            gr.Markdown("### 📄 Tez Dokümanı")
            gr.DownloadButton(label="📄 Tezi İndir (PDF)", value="data/tez.pdf")

            gr.Markdown("### 📚 İçindekiler")
            gr.HTML(
                """
                <div class="toc-container">
                  <div class="toc-card"><div class="toc-header"><span>1. GİRİŞ</span></div></div>
                  <div class="toc-card"><div class="toc-header"><span>2. YAPAY ZEKÂ VE DOĞAL DİL İŞLEME</span></div></div>
                  <div class="toc-card"><div class="toc-header"><span>3. DİL MODELLEMEDE ML ve DL</span></div></div>
                  <div class="toc-card"><div class="toc-header"><span>4. DİL MODELLERİ</span></div></div>
                  <div class="toc-card"><div class="toc-header"><span>5. TRANSFORMER TABANLI MODELLER</span></div></div>
                  <div class="toc-card"><div class="toc-header"><span>6. GÜNCEL YÖNELİMLER ve ETİK</span></div></div>
                  <div class="toc-card"><div class="toc-header"><span>7. SONUÇ ve DEĞERLENDİRME</span></div></div>
                </div>
                """
            )

            length_choice = gr.Radio(
                choices=["Kısa", "Orta", "Uzun"],
                value=DEFAULT_RESPONSE_LENGTH,
                label="Yanıt uzunluğu"
            )

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=500, avatar_images=(None, None), type="messages")
            input_box = gr.Textbox(placeholder="Sorunuzu yazın ve Enter'a basın...", scale=1)
            send_btn = gr.Button("Gönder", variant="primary")

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

    logs = auto_ingest_from_repo()
    if logs:
        print(logs)


if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)