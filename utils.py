# utils.py

import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
import torch
import os
import re
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

def extract_outline(doc):
    """Enhanced outline extraction with fallback strategies."""
    outline = []
    
    # Strategy 1: Try to extract from document's table of contents
    toc = doc.get_toc()
    if toc:
        for level, text, page in toc:
            if level <= 3 and text.strip():
                outline.append({
                    "level": f"H{level}", 
                    "text": clean_heading_text(text.strip()), 
                    "page": page
                })
        
        if outline:
            return outline
    
    # Strategy 2: Use font size and formatting analysis
    outline = extract_by_font_analysis(doc)
    if outline:
        return outline
    
    # Strategy 3: Pattern-based extraction
    return extract_by_patterns(doc)

def clean_heading_text(text):
    """Clean and normalize heading text."""
    # Remove excessive whitespace and normalize
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove common artifacts
    text = re.sub(r'^[\d\.\-\s]+', '', text)  # Remove leading numbers/dots
    return text[:200]  # Limit length

def extract_by_font_analysis(doc):
    """Extract headings based on font size analysis."""
    outline = []
    font_sizes = defaultdict(list)
    
    # Analyze font sizes across the document
    for page_num in range(min(10, doc.page_count)):  # Sample first 10 pages
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        if span["text"].strip():
                            font_sizes[span["size"]].append({
                                "text": span["text"].strip(),
                                "page": page_num + 1,
                                "flags": span["flags"]
                            })
    
    # Determine heading hierarchy based on font sizes
    sorted_sizes = sorted(font_sizes.keys(), reverse=True)
    
    for page_num in range(doc.page_count):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text and len(text.split()) < 15:  # Likely heading
                            size = span["size"]
                            level = determine_heading_level(size, sorted_sizes)
                            if level <= 3:
                                outline.append({
                                    "level": f"H{level}",
                                    "text": clean_heading_text(text),
                                    "page": page_num + 1
                                })
    
    return remove_duplicate_headings(outline)

def determine_heading_level(font_size, sorted_sizes):
    """Determine heading level based on font size ranking."""
    if len(sorted_sizes) < 2:
        return 2
    
    # Map font sizes to heading levels
    for i, size in enumerate(sorted_sizes[:3]):
        if font_size >= size:
            return i + 1
    return 3

def extract_by_patterns(doc):
    """Extract headings using pattern matching."""
    outline = []
    heading_patterns = [
        r'^[A-Z][A-Z\s]{5,50}$',  # ALL CAPS headings
        r'^\d+\.?\s+[A-Z].*',      # Numbered headings
        r'^[A-Z][a-z\s]{3,50}$',   # Title case headings
    ]
    
    for page_num in range(doc.page_count):
        page = doc[page_num]
        blocks = page.get_text("blocks")
        
        for block in blocks:
            text = block[4].strip()
            if text and len(text.split()) < 15:
                for pattern in heading_patterns:
                    if re.match(pattern, text):
                        outline.append({
                            "level": "H2",
                            "text": clean_heading_text(text),
                            "page": page_num + 1
                        })
                        break
    
    return remove_duplicate_headings(outline)

def remove_duplicate_headings(outline):
    """Remove duplicate headings while preserving order."""
    seen = set()
    unique_outline = []
    
    for item in outline:
        key = (item["text"].lower(), item["page"])
        if key not in seen:
            seen.add(key)
            unique_outline.append(item)
    
    return unique_outline

def load_model(model_path='./model'):
    """Loads the Sentence Transformer model from a local directory."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found at {model_path}")
    
    logger.info("Loading semantic model...")
    model = SentenceTransformer(model_path)
    
    # Optimize for CPU inference
    model.eval()
    if hasattr(model, '_modules'):
        for module in model._modules.values():
            if hasattr(module, 'eval'):
                module.eval()
    
    return model

def get_section_text(doc, section_page, end_page, max_chars=5000):
    """Enhanced section text extraction with length limiting."""
    full_text = ""
    
    for page_num in range(section_page - 1, min(end_page, doc.page_count)):
        page_text = doc.load_page(page_num).get_text("text")
        full_text += page_text + "\n"
        
        # Limit text length to avoid processing very long sections
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars] + "..."
            break
    
    return clean_text(full_text)

def clean_text(text):
    """Clean and normalize text content."""
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common PDF artifacts
    text = re.sub(r'\x0c', '', text)  # Form feed characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
    
    return text.strip()

def calculate_enhanced_relevance(section_text, persona, job_to_be_done, model, section_metadata):
    """Calculate enhanced relevance score with multiple factors."""
    
    # Base semantic similarity
    query = f"Persona: {persona}. Task: {job_to_be_done}"
    query_embedding = model.encode(query, convert_to_tensor=True)
    section_embedding = model.encode(section_text, convert_to_tensor=True)
    base_score = util.cos_sim(query_embedding, section_embedding).item()
    
    # Persona-specific weighting
    persona_score = calculate_persona_alignment(section_text, persona)
    
    # Task-specific weighting
    task_score = calculate_task_relevance(section_text, job_to_be_done)
    
    # Section importance (based on position, length, etc.)
    section_importance = calculate_section_importance(section_metadata)
    
    # Weighted combination
    final_score = (
        base_score * 0.4 +
        persona_score * 0.25 +
        task_score * 0.25 +
        section_importance * 0.1
    )
    
    return final_score

def calculate_persona_alignment(text, persona):
    """Calculate how well the text aligns with the persona's expertise."""
    persona_keywords = {
        'researcher': ['methodology', 'analysis', 'study', 'research', 'findings', 'data', 'results'],
        'student': ['example', 'definition', 'concept', 'explanation', 'basic', 'introduction'],
        'analyst': ['trend', 'performance', 'metric', 'analysis', 'comparison', 'evaluation'],
        'manager': ['strategy', 'planning', 'decision', 'management', 'process', 'implementation'],
        'developer': ['code', 'implementation', 'technical', 'system', 'architecture', 'design']
    }
    
    text_lower = text.lower()
    max_score = 0
    
    for persona_type, keywords in persona_keywords.items():
        if persona_type.lower() in persona.lower():
            score = sum(1 for keyword in keywords if keyword in text_lower) / len(keywords)
            max_score = max(max_score, score)
    
    return min(max_score, 1.0)

def calculate_task_relevance(text, job_to_be_done):
    """Calculate relevance to the specific task."""
    task_keywords = extract_key_terms(job_to_be_done)
    text_lower = text.lower()
    
    relevance_score = 0
    for keyword in task_keywords:
        if keyword.lower() in text_lower:
            relevance_score += 1
    
    return min(relevance_score / max(len(task_keywords), 1), 1.0)

def extract_key_terms(text):
    """Extract key terms from job description."""
    # Simple keyword extraction - could be enhanced with NLP
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
    return [word for word in words if word not in stop_words and len(word) > 3]

def calculate_section_importance(section_metadata):
    """Calculate section importance based on metadata."""
    importance = 0.5  # Base importance
    
    # Earlier sections might be more important (introduction, overview)
    if section_metadata.get('page', 1) <= 5:
        importance += 0.2
    
    # Sections with certain keywords in title
    title = section_metadata.get('title', '').lower()
    important_keywords = ['introduction', 'overview', 'summary', 'conclusion', 'methodology', 'results']
    
    for keyword in important_keywords:
        if keyword in title:
            importance += 0.1
            break
    
    return min(importance, 1.0)

def find_and_rank_sections(model, doc_collection, persona, job_to_be_done):
    """Enhanced main logic for Round 1B with improved ranking."""
    
    query = f"Persona: {persona}. Task: {job_to_be_done}"
    logger.info(f"Processing query: {query}")
    
    all_sections = []
    
    # Process each document
    for doc_path in doc_collection:
        doc_filename = os.path.basename(doc_path)
        logger.info(f"Processing document: {doc_filename}")
        
        try:
            doc = fitz.open(doc_path)
            outline = extract_outline(doc)
            
            if not outline:
                logger.warning(f"Could not extract outline from {doc_filename}")
                continue
            
            # Process each section
            for i, section in enumerate(outline):
                start_page = section["page"]
                end_page = doc.page_count if (i + 1) == len(outline) else outline[i+1]["page"]
                
                section_text = get_section_text(doc, start_page, end_page)
                
                if not section_text or len(section_text.split()) < 20:
                    continue
                
                # Calculate enhanced relevance score
                section_metadata = {
                    'title': section["text"],
                    'page': start_page,
                    'length': len(section_text.split())
                }
                
                relevance_score = calculate_enhanced_relevance(
                    section_text, persona, job_to_be_done, model, section_metadata
                )
                
                all_sections.append({
                    "document": doc_filename,
                    "page_number": start_page,
                    "section_title": section["text"],
                    "score": relevance_score,
                    "text_preview": section_text[:200] + "..." if len(section_text) > 200 else section_text
                })
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error processing {doc_filename}: {str(e)}")
            continue
    
    # Enhanced sorting with diversity consideration
    sorted_sections = sorted(all_sections, key=lambda x: x["score"], reverse=True)
    
    # Apply diversity filter to avoid too many sections from same document
    final_sections = apply_diversity_filter(sorted_sections)
    
    # Format for output
    final_ranked_sections = []
    for rank, section in enumerate(final_sections):
        final_ranked_sections.append({
            "document": section["document"],
            "page_number": section["page_number"],
            "section_title": section["section_title"],
            "importance_rank": rank + 1,
            "confidence_score": round(section["score"], 3)
        })
    
    return final_ranked_sections

def apply_diversity_filter(sorted_sections, max_per_doc=3):
    """Apply diversity filtering to ensure balanced representation."""
    doc_counts = defaultdict(int)
    filtered_sections = []
    
    for section in sorted_sections:
        doc_name = section["document"]
        if doc_counts[doc_name] < max_per_doc:
            filtered_sections.append(section)
            doc_counts[doc_name] += 1
        
        # Stop when we have enough diverse sections
        if len(filtered_sections) >= 20:
            break
    
    return filtered_sections

def analyze_subsections(model, doc_collection, top_sections, persona, job_to_be_done):
    """Implement proper sub-section analysis (worth 40 points)."""
    
    logger.info("Starting sub-section analysis...")
    subsection_results = []
    
    for section in top_sections[:5]:  # Analyze top 5 sections
        try:
            # Find and open the document
            doc_path = None
            for path in doc_collection:
                if os.path.basename(path) == section["document"]:
                    doc_path = path
                    break
            
            if not doc_path:
                continue
                
            doc = fitz.open(doc_path)
            
            # Get the full section text
            section_text = get_section_text(doc, section["page_number"], section["page_number"] + 2)
            
            # Break into sub-sections (paragraphs)
            subsections = break_into_subsections(section_text)
            
            # Analyze each sub-section
            for i, subsection in enumerate(subsections):
                if len(subsection.split()) < 30:  # Skip very short subsections
                    continue
                
                # Calculate relevance for this specific subsection
                query = f"Persona: {persona}. Task: {job_to_be_done}"
                query_embedding = model.encode(query, convert_to_tensor=True)
                subsection_embedding = model.encode(subsection, convert_to_tensor=True)
                relevance_score = util.cos_sim(query_embedding, subsection_embedding).item()
                
                if relevance_score > 0.3:  # Only include relevant subsections
                    # Generate refined text (summary/key points)
                    refined_text = generate_refined_text(subsection, persona, job_to_be_done)
                    
                    subsection_results.append({
                        "document": section["document"],
                        "refined_text": refined_text,
                        "page_number": section["page_number"],
                        "relevance_score": round(relevance_score, 3),
                        "subsection_index": i + 1
                    })
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error in subsection analysis for {section.get('document', 'unknown')}: {str(e)}")
            continue
    
    # Sort by relevance and return top results
    subsection_results.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    # Format for output (remove internal scoring fields)
    final_subsections = []
    for result in subsection_results[:10]:  # Top 10 subsections
        final_subsections.append({
            "document": result["document"],
            "refined_text": result["refined_text"],
            "page_number": result["page_number"]
        })
    
    return final_subsections

def break_into_subsections(text):
    """Break section text into meaningful sub-sections."""
    # Split by double line breaks (paragraph boundaries)
    paragraphs = re.split(r'\n\s*\n', text)
    
    # Filter out very short paragraphs and combine related ones
    subsections = []
    current_subsection = ""
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if len(paragraph.split()) < 10:  # Very short paragraph
            current_subsection += " " + paragraph
        else:
            if current_subsection:
                subsections.append(current_subsection.strip())
                current_subsection = ""
            
            if len(paragraph.split()) > 50:  # Long paragraph - treat as separate subsection
                subsections.append(paragraph)
            else:
                current_subsection = paragraph
    
    # Add any remaining content
    if current_subsection:
        subsections.append(current_subsection.strip())
    
    return [s for s in subsections if len(s.split()) >= 20]

def generate_refined_text(subsection_text, persona, job_to_be_done):
    """Generate refined/summarized text for the subsection."""
    
    # Simple extractive summarization - get key sentences
    sentences = re.split(r'[.!?]+', subsection_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    if len(sentences) <= 2:
        return subsection_text[:300] + "..." if len(subsection_text) > 300 else subsection_text
    
    # Score sentences based on key terms from job_to_be_done
    job_keywords = extract_key_terms(job_to_be_done)
    
    sentence_scores = []
    for sentence in sentences:
        score = 0
        sentence_lower = sentence.lower()
        for keyword in job_keywords:
            if keyword in sentence_lower:
                score += 1
        sentence_scores.append((sentence, score))
    
    # Sort by score and take top sentences
    sentence_scores.sort(key=lambda x: x[1], reverse=True)
    top_sentences = [s[0] for s in sentence_scores[:3]]
    
    refined = ". ".join(top_sentences)
    
    # Limit length
    if len(refined) > 400:
        refined = refined[:400] + "..."
    
    return refined if refined else subsection_text[:300] + "..."
