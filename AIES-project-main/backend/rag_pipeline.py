"""
RAG Pipeline — FSSAI Regulation Retrieval-Augmented Generation

Uses:
  - sentence-transformers (all-MiniLM-L6-v2) for embedding
  - ChromaDB as the vector store
  - Retrieves exact FSSAI regulatory clauses for each compliance violation
"""

import os
import re
import chromadb
from sentence_transformers import SentenceTransformer

# ── Paths ──────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
REGULATIONS_FILE = os.path.join(DATA_DIR, "fssai_regulations.txt")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")

# ── Globals (lazy-loaded) ──────────────────────────────────────────
_model: SentenceTransformer | None = None
_collection = None
_initialized = False


def _get_model() -> SentenceTransformer:
    """Lazy-load the sentence-transformer embedding model."""
    global _model
    if _model is None:
        print("[RAG] Loading embedding model: all-MiniLM-L6-v2 ...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[RAG] Embedding model loaded.")
    return _model


def _chunk_document(text: str, chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    """
    Split the FSSAI regulation document into overlapping chunks.
    Each chunk preserves section context via headers.
    """
    chunks = []
    lines = text.strip().split("\n")

    current_section = ""
    current_subsection = ""
    current_text = ""

    for line in lines:
        # Track section headers for context
        if line.startswith("### "):
            current_section = line.strip("# ").strip()
            current_text += f"\n{line}\n"
        elif line.startswith("#### "):
            current_subsection = line.strip("# ").strip()
            current_text += f"\n{line}\n"
        else:
            current_text += line + "\n"

        # When chunk is large enough, save it and start a new one with overlap
        if len(current_text) >= chunk_size:
            chunks.append({
                "text": current_text.strip(),
                "section": current_section,
                "subsection": current_subsection,
            })
            # Keep last `overlap` characters for continuity
            current_text = current_text[-overlap:]

    # Add the final chunk
    if current_text.strip():
        chunks.append({
            "text": current_text.strip(),
            "section": current_section,
            "subsection": current_subsection,
        })

    return chunks


def initialize_vector_store(force_rebuild: bool = False) -> None:
    """
    Build (or load) the ChromaDB vector store from the FSSAI regulations.
    This embeds all regulation chunks using all-MiniLM-L6-v2.
    """
    global _collection, _initialized

    if _initialized and not force_rebuild:
        return

    model = _get_model()

    # Initialize ChromaDB persistent client
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Check if collection already exists with data
    try:
        _collection = client.get_collection("fssai_regulations")
        count = _collection.count()
        if count > 0 and not force_rebuild:
            print(f"[RAG] ChromaDB collection loaded with {count} chunks.")
            _initialized = True
            return
        else:
            # Rebuild
            client.delete_collection("fssai_regulations")
    except Exception:
        pass

    # Create fresh collection
    _collection = client.create_collection(
        name="fssai_regulations",
        metadata={"description": "FSSAI Labelling & Display Regulations, 2020"},
    )

    # Read and chunk the regulations
    print("[RAG] Reading FSSAI regulations document...")
    with open(REGULATIONS_FILE, "r", encoding="utf-8") as f:
        raw_text = f.read()

    chunks = _chunk_document(raw_text)
    print(f"[RAG] Created {len(chunks)} chunks from regulation document.")

    # Embed all chunks
    print("[RAG] Embedding chunks with all-MiniLM-L6-v2 ...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    # Add to ChromaDB
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"section": c["section"], "subsection": c["subsection"]} for c in chunks]

    _collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    print(f"[RAG] Vector store initialized with {len(chunks)} regulation chunks.")
    _initialized = True


def retrieve_regulations(query: str, n_results: int = 3) -> list[dict]:
    """
    Retrieve the most relevant FSSAI regulation chunks for a given query.

    Args:
        query: Natural language query about a compliance issue
        n_results: Number of top results to return

    Returns:
        List of dicts with keys: text, section, subsection, distance
    """
    if not _initialized:
        initialize_vector_store()

    model = _get_model()

    # Embed the query
    query_embedding = model.encode([query]).tolist()

    # Search ChromaDB
    results = _collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
    )

    # Format results
    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "text": results["documents"][0][i],
            "section": results["metadatas"][0][i].get("section", ""),
            "subsection": results["metadatas"][0][i].get("subsection", ""),
            "distance": results["distances"][0][i] if results.get("distances") else None,
        })

    return retrieved


def generate_explanation(check_type: str, status: str, extracted_data: str, detail: str) -> dict:
    """
    Generate an explainable compliance explanation using RAG-retrieved context.

    Instead of hallucinating, retrieves the exact FSSAI clause and builds
    a grounded explanation.

    Args:
        check_type: Type of check (e.g., 'fssai_license', 'date_marking')
        status: 'pass', 'fail', or 'warn'
        extracted_data: What was extracted from the label
        detail: Additional detail about the finding

    Returns:
        Dict with regulation_text, section, explanation, and fix (if applicable)
    """
    # Build a retrieval query tailored to the check type
    query_map = {
        "fssai_license": "FSSAI license number format 14 digits requirement regulation",
        "fssai_state_code": "FSSAI license state code Indian administrative divisions",
        "fssai_presence": "FSSAI license number displayed prominently on package",
        "fssai_logo": "FSSAI logo emblem shield and license number requirement",
        "dietary_symbol": "vegetarian non-vegetarian declaration symbol requirement",
        "dietary_color": "green circle green square brown triangle brown square symbol color",
        "dietary_ingredients_conflict": "vegetarian non-vegetarian animal origin ingredients conflict",
        "allergen_declaration": "allergen declaration contains statement major allergens",
        "date_marking": "date marking best before expiry use by date requirement",
        "nutritional_info": "nutritional information panel energy protein fat sodium requirement",
    }

    query = query_map.get(check_type, f"FSSAI regulation {check_type}")

    # If failed, make the query more specific
    if status == "fail":
        query += f" non-compliance penalty"
    elif status == "warn":
        query += f" verification review"

    # Retrieve relevant regulation chunks
    retrieved = retrieve_regulations(query, n_results=3)

    if not retrieved:
        return {
            "regulation_text": "Unable to retrieve regulation text.",
            "section": "N/A",
            "explanation": detail,
            "fix": "Please consult FSSAI (Labelling & Display) Regulations, 2020 directly.",
        }

    # Use the most relevant chunk
    best = retrieved[0]
    regulation_text = best["text"]
    section = best.get("subsection") or best.get("section") or "FSSAI Regulations 2020"

    # Build grounded explanation
    if status == "pass":
        explanation = (
            f"✅ COMPLIANT — {detail}\n\n"
            f"This satisfies the requirement under {section} of the FSSAI "
            f"(Labelling & Display) Regulations, 2020."
        )
        fix = None
    elif status == "fail":
        explanation = (
            f"❌ NON-COMPLIANT — {detail}\n\n"
            f"As per {section}: {_extract_key_requirement(regulation_text, check_type)}"
        )
        fix = _generate_fix(check_type, extracted_data)
    else:
        explanation = (
            f"⚠️ NEEDS REVIEW — {detail}\n\n"
            f"Relevant regulation ({section}): {_extract_key_requirement(regulation_text, check_type)}\n"
            f"Manual verification recommended due to OCR confidence limitations."
        )
        fix = _generate_fix(check_type, extracted_data)

    return {
        "regulation_text": regulation_text[:500],  # Truncate for display
        "section": section,
        "explanation": explanation,
        "fix": fix,
        "retrieved_chunks": [
            {"section": r["section"], "subsection": r["subsection"], "distance": r["distance"]}
            for r in retrieved
        ],
    }


def _extract_key_requirement(regulation_text: str, check_type: str) -> str:
    """Extract the key requirement sentence from the regulation text."""
    sentences = re.split(r'(?<=[.!?])\s+', regulation_text)

    # Keywords to look for per check type
    keywords = {
        "fssai_license": ["14 digits", "license number", "FSSAI license", "^[12]"],
        "date_marking": ["best before", "use by", "expiry", "date of manufacture"],
        "allergen_declaration": ["allergen", "contains", "8 major", "declaration"],
        "nutritional_info": ["nutritional", "energy", "protein", "per 100g"],
        "dietary_symbol": ["vegetarian", "green", "brown", "triangle", "circle", "symbol"],
    }

    target_keywords = keywords.get(check_type, [])

    # Find the most relevant sentence
    for sentence in sentences:
        lower = sentence.lower()
        if any(kw.lower() in lower for kw in target_keywords):
            return sentence.strip()

    # Fallback: return the first meaningful sentence
    for sentence in sentences:
        if len(sentence) > 30:
            return sentence.strip()

    return regulation_text[:200]


def _generate_fix(check_type: str, extracted_data: str) -> str:
    """Generate an actionable fix recommendation."""
    fixes = {
        "fssai_license": (
            "Ensure the FSSAI license number is exactly 14 digits, starting with "
            "'1' (Central) or '2' (State), followed by a valid 2-digit state code. "
            "Display it prominently alongside the FSSAI logo."
        ),
        "date_marking": (
            "Add clearly visible date markings with proper labels: "
            "'Best Before DD/MM/YYYY', 'Use By DD/MM/YYYY', or 'Expiry Date DD/MM/YYYY'. "
            "Include both manufacturing/packing date and best before/expiry date."
        ),
        "allergen_declaration": (
            "Add a dedicated allergen declaration immediately after the ingredient list. "
            "Format: 'Contains: [allergen names]' in bold or contrasting font. "
            "Include 'May contain' statement for cross-contamination risks."
        ),
        "nutritional_info": (
            "Add a complete Nutritional Information panel in tabular format declaring: "
            "Energy (kcal), Protein (g), Carbohydrate (g), Total Sugars (g), Added Sugars (g), "
            "Total Fat (g), Saturated Fat (g), Trans Fat (g), and Sodium (mg) per 100g/ml."
        ),
        "dietary_symbol": (
            "Display the appropriate dietary symbol on the front of the package: "
            "Veg = green filled circle in green square, "
            "Non-Veg = brown filled triangle in brown square. "
            "Minimum size: 3mm×3mm (PDP ≤100 sq cm) or 5mm×5mm (PDP >100 sq cm)."
        ),
    }
    return fixes.get(check_type, "Please consult FSSAI guidelines for corrective action.")
