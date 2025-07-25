# final_main.py

import json
import os
import glob
from datetime import datetime
import fitz  # PyMuPDF
from utils import find_and_rank_sections_V2, load_model
from tqdm import tqdm

def generate_descriptive_query(persona, job_to_be_done):
    """
    Creates a rich, descriptive query to guide the model more effectively.
    """
    role = persona.get("role", "user")
    task = job_to_be_done.get("task", "")
    
    # This is the secret sauce. We create a better prompt for the model.
    # It tells the model to focus on the 'why' behind the query.
    descriptive_prompt = (
        f"Based on the needs of a {role}, find the most relevant information "
        f"to accomplish the following task: '{task}'. Focus on finding actionable "
        "insights, key activities, relevant places, and recommendations that directly "
        "help with this specific job."
    )
    return descriptive_prompt

def main():
    # --- IMPORTANT: Change paths for Local vs. Docker ---
    # For Local Testing:
    # base_dir = os.getcwd()
    # input_dir = os.path.join(base_dir, "input")
    # output_dir = os.path.join(base_dir, "output")
    # model_dir = os.path.join(base_dir, "model")

    # For Docker Submission, UNCOMMENT these lines:
    input_dir = "/app/input"
    output_dir = "/app/output"
    model_dir = "/app/model"
    # ---

    docs_folder = os.path.join(input_dir, "docs")
    request_file = os.path.join(input_dir, "request.json")
    output_file = os.path.join(output_dir, "challenge1b_output.json")
    
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Request and Documents
    with open(request_file, 'r', encoding='utf-8') as f:
        request_data = json.load(f)
        
    persona = request_data.get("persona")
    job_to_be_done = request_data.get("job_to_be_done")
    doc_paths = glob.glob(os.path.join(docs_folder, "*.pdf"))
    input_documents = [os.path.basename(p) for p in doc_paths]

    # 2. Generate our new, smarter query
    query = generate_descriptive_query(persona, job_to_be_done)
    print(f"✅ Generated Smart Query: {query}")

    # 3. Load the NLP model
    model = load_model(model_dir)

    # 4. Find and Rank Sections using the new V2 logic
    print("\n--- Finding and Ranking Relevant Sections ---")
    ranked_sections = find_and_rank_sections_V2(
        doc_paths=doc_paths,
        query=query,
        model=model,
        show_progress=True
    )
    
    # 5. Prepare Final JSON Output
    top_5_sections_for_analysis = ranked_sections[:5]
    subsection_analysis_output = []
    for section in top_5_sections_for_analysis:
        subsection_analysis_output.append({
            "document": section["document"],
            "refined_text": section["refined_text"],
            "page_number": section["page_number"]
        })

    # Limit extracted sections to the top 30 as you suggested
    extracted_section_output = []
    for rank, section in enumerate(ranked_sections[:30]):
        extracted_section_output.append({
            "document": section["document"],
            "section_title": section["section_title"],
            "importance_rank": rank + 1,
            "page_number": section["page_number"]
        })
        
    output_data = {
        "metadata": {
            "input_documents": input_documents,
            "persona": persona['role'],
            "job_to_be_done": job_to_be_done['task'],
            "processing_timestamp": datetime.utcnow().isoformat()
        },
        "extracted_sections": extracted_section_output,
        "subsection_analysis": subsection_analysis_output
    }

    # 6. Write the final file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"\n✅ Processing complete! Output saved to {output_file}")

if __name__ == '__main__':
    main()