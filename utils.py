# final_utils.py

import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
import os
import collections
from tqdm import tqdm

def load_model(model_path='./model'):
    """Loads the Sentence Transformer model from a local directory."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found. Please run download_model.py")
    print("Loading semantic model...")
    return SentenceTransformer(model_path, device='cpu')

def extract_outline(doc):
    """
    Extracts a basic outline from the document's table of contents.
    Falls back to a simple heuristic if no ToC is available.
    """
    toc = doc.get_toc()
    if toc:
        return [{"level": f"H{level}", "text": text.strip(), "page": page} for level, text, page in toc if level <= 3]
    
    # Basic fallback (can be replaced with your more advanced R1A version if needed)
    outline = []
    for page_num in range(doc.page_count):
        for block in doc[page_num].get_text("blocks", sort=True):
            if block[6] == 0 and len(block[4].split()) < 15: # Simple heuristic
                text = block[4].strip().replace('\n', ' ')
                if text:
                    outline.append({"level": "H2", "text": text, "page": page_num + 1})
    return outline

def get_section_text(doc, start_page, end_page):
    """Extracts all text within a section, from its start page to its end page."""
    full_text = ""
    # Ensure page numbers are within valid range
    start_page_valid = max(0, start_page - 1)
    end_page_valid = min(doc.page_count, end_page)
    for page_num in range(start_page_valid, end_page_valid):
        full_text += doc.load_page(page_num).get_text("text") + "\n"
    return full_text.strip()

def chunk_text_by_paragraph(text, min_chunk_size=50):
    """Splits text into paragraphs and filters out very short ones."""
    chunks = text.split('\n\n')  # Paragraphs are often separated by double newlines
    return [chunk.strip() for chunk in chunks if len(chunk.split()) > min_chunk_size]

def find_and_rank_sections_V2(doc_paths, query, model, show_progress=False):
    """
    The new V2 logic: Chunks sections, finds the most relevant chunk,
    and ranks the parent sections based on their best chunk.
    """
    all_chunks = []
    
    # Phase 1: Parse docs, extract sections, and create text chunks
    for doc_path in tqdm(doc_paths, desc="Parsing and Chunking", disable=not show_progress):
        doc_filename = os.path.basename(doc_path)
        try:
            doc = fitz.open(doc_path)
            outline = extract_outline(doc)
            if not outline: continue

            for i, section in enumerate(outline):
                start_page = section["page"]
                end_page = doc.page_count if (i + 1) >= len(outline) else outline[i+1]["page"]
                section_text = get_section_text(doc, start_page, end_page)
                
                paragraph_chunks = chunk_text_by_paragraph(section_text)
                
                for chunk_text in paragraph_chunks:
                    all_chunks.append({
                        "chunk_text": chunk_text,
                        "metadata": {
                            "document": doc_filename,
                            "page_number": start_page,
                            "section_title": section["text"],
                        }
                    })
            doc.close()
        except Exception as e:
            print(f"Warning: Could not process {doc_filename}. Error: {e}")

    if not all_chunks:
        return []
        
    # Phase 2: Encode the query and all chunks
    print("Encoding query and text chunks...")
    query_embedding = model.encode(query, convert_to_tensor=True, show_progress_bar=show_progress)
    chunk_embeddings = model.encode([c['chunk_text'] for c in all_chunks], convert_to_tensor=True, show_progress_bar=show_progress)
    
    # Phase 3: Calculate scores and find the best chunk for each section
    scores = util.cos_sim(query_embedding, chunk_embeddings)[0]
    
    section_scores = {}
    for i, score in enumerate(scores):
        score = score.item()
        metadata = all_chunks[i]['metadata']
        section_id = f"{metadata['document']}_{metadata['section_title']}"
        
        # If this chunk has a higher score than any previous chunk from the same section, update it
        if section_id not in section_scores or score > section_scores[section_id]['score']:
            section_scores[section_id] = {
                "document": metadata['document'],
                "page_number": metadata['page_number'],
                "section_title": metadata['section_title'],
                "score": score,
                "refined_text": all_chunks[i]['chunk_text'] # This is our sub-section text!
            }
            
    # Phase 4: Sort the unique sections by their best score
    ranked_results = sorted(section_scores.values(), key=lambda x: x['score'], reverse=True)
    return ranked_results