from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import re
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import easyocr

from .models import ImportPreviewRequest, ImportPreviewResponse, OcrExtractResponse
from .schema_infer import infer_schema

# ======================
#  Config & constantes
# ======================
REQUIRE_JWT = False

# Limites (overridables via variables d'env)
OCR_DPI = int(os.getenv("OCR_DPI", "160"))               # 160 (au lieu de 220) pour réduire la RAM
OCR_MAX_PAGES = int(os.getenv("OCR_MAX_PAGES", "3"))     # max pages à traiter
OCR_MAX_LONG = int(os.getenv("OCR_MAX_LONG", "1800"))    # côté long max (px) après rendu
MAX_PDF_MB = float(os.getenv("MAX_PDF_MB", "10"))        # taille max du fichier PDF

# ======================
#  App FastAPI
# ======================
app = FastAPI(title="OCR/Import Service", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# EasyOCR (CPU)
reader = easyocr.Reader(['fr', 'en'], gpu=False)


def verify_jwt_if_needed(req: Request):
    if not REQUIRE_JWT:
        return
    auth = req.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Missing Bearer token")
    # TODO: décoder/valider le JWT (clé publique) quand tu seras prêt.


# ======================
#  Helpers OCR
# ======================
def pdf_bytes_to_images(pdf_bytes: bytes, dpi: int = OCR_DPI, max_pages: int = OCR_MAX_PAGES) -> List[np.ndarray]:
    """
    Rend les pages PDF en images RGB (numpy arrays) via PyMuPDF, avec un DPI choisi.
    Limite à max_pages pour éviter les OOM.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images: List[np.ndarray] = []
    try:
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(np.array(img))
    finally:
        doc.close()
    return images


def shrink_image_np(img: np.ndarray, max_long: int = OCR_MAX_LONG) -> np.ndarray:
    """
    Redimensionne l'image si son côté le plus long dépasse max_long.
    """
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= max_long:
        return img
    scale = max_long / long_side
    new_size = (int(w * scale), int(h * scale))
    pil = Image.fromarray(img)
    pil = pil.resize(new_size, Image.LANCZOS)
    return np.array(pil)


def split_table_like(line: str) -> List[str]:
    """
    Heuristique "tableau": split par tabulation OU par >= 2 espaces contigus.
    (Les tabs du PDF d'origine disparaissent souvent après OCR → on tolère 2+ espaces)
    """
    line = line.strip()
    if "\t" in line:
        parts = [p.strip() for p in line.split("\t") if p.strip()]
    else:
        parts = [p.strip() for p in re.split(r"\s{2,}", line) if p.strip()]
    return parts


# ======================
#  Endpoints
# ======================
@app.post("/api/ocr/extract", response_model=OcrExtractResponse)
async def ocr_extract(request: Request, file: UploadFile = File(...)):
    verify_jwt_if_needed(request)

    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(400, "PDF attendu")

    content = await file.read()
    if (len(content) / (1024 * 1024)) > MAX_PDF_MB:
        raise HTTPException(413, f"Fichier trop volumineux (> {MAX_PDF_MB} MB)")

    # 1) PDF -> images (avec limites)
    try:
        images = pdf_bytes_to_images(content, dpi=OCR_DPI, max_pages=OCR_MAX_PAGES)
    except Exception as e:
        raise HTTPException(400, f"Echec rendu PDF: {e}")

    if not images:
        return OcrExtractResponse(text="", rows=None)

    # 2) OCR par page (avec shrink pour limiter la RAM)
    all_lines: List[str] = []
    try:
        for img in images:
            img = shrink_image_np(img, max_long=OCR_MAX_LONG)
            # paragraph=True pour grouper ; mets False si tu veux des lignes fragmentées
            lines = reader.readtext(img, detail=0, paragraph=True)
            if isinstance(lines, list):
                all_lines.extend([str(x) for x in lines])
    except (MemoryError, RuntimeError) as e:
        # torch peut lever RuntimeError "not enough memory"
        raise HTTPException(
            413,
            f"Document trop lourd pour l’OCR avec les limites actuelles "
            f"(dpi={OCR_DPI}, pages<={OCR_MAX_PAGES}, max_long={OCR_MAX_LONG}px). "
            f"Réduis le PDF ou abaisse les limites (OCR_DPI/OCR_MAX_PAGES/OCR_MAX_LONG). Détail: {e}"
        )

    text = "\n".join(all_lines).strip()

    # 3) Heuristique "tableau"
    rows = []
    for line in all_lines:
        parts = split_table_like(line)
        if len(parts) >= 3:
            rows.append({f"col{i+1}": v for i, v in enumerate(parts)})

    return OcrExtractResponse(text=text, rows=rows or None)


@app.post("/api/import/preview", response_model=ImportPreviewResponse)
def import_preview(request: Request, req: ImportPreviewRequest):
    verify_jwt_if_needed(request)
    rows = req.rows or []
    schema = req.schema or infer_schema(rows)
    warnings: List[str] = []
    if len(rows) > 50000:
        warnings.append("Gros volume détecté: l’aperçu est tronqué côté client.")
    return ImportPreviewResponse(sample=rows[:200], schema=schema, warnings=warnings)
