# main.py

import json
import os
import glob
from datetime import datetime
from utils import find_and_rank_sections, load_model, analyze_subsections
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    input_dir = "/app/input"
    output_dir = "/app/output"
    model_dir = "/app/model"
    
    docs_dir = os.path.join(input_dir, "docs")
    request_file = os.path.join(input_dir, "request.json")
    output_file = os.path.join(output_dir, "challenge1b_output.json")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. Load inputs
        with open(request_file, 'r') as f:
            request_data = json.load(f)
        
        persona = request_data.get("persona")
        job_to_be_done = request_data.get("job_to_be_done")
        
        # Find all PDF documents
        doc_collection = glob.glob(os.path.join(docs_dir, "*.pdf"))
        input_documents = [os.path.basename(p) for p in doc_collection]
        
        if not all([persona, job_to_be_done, doc_collection]):
            logger.error("Error: Missing persona, job_to_be_done, or documents.")
            return
        
        logger.info(f"Processing {len(doc_collection)} documents for persona: {persona}")
        logger.info(f"Job to be done: {job_to_be_done}")
        
        # 2. Load the model
        model = load_model(model_dir)
        
        # 3. Find and rank the sections with enhanced algorithm
        extracted_sections = find_and_rank_sections(model, doc_collection, persona, job_to_be_done)
        
        # 4. Perform sub-section analysis on top sections
        logger.info("Performing sub-section analysis...")
        subsection_analysis = analyze_subsections(model, doc_collection, extracted_sections[:10], persona, job_to_be_done)
        
        # 5. Prepare the final JSON output
        output_data = {
            "Metadata": {
                "input_documents": input_documents,
                "persona": persona,
                "job_to_be_done": job_to_be_done,
                "processing_timestamp": datetime.utcnow().isoformat() + "Z",
                "total_sections_analyzed": len(extracted_sections),
                "subsections_generated": len(subsection_analysis)
            },
            "Extracted Section": extracted_sections,
            "Sub-section Analysis": subsection_analysis
        }
        
        # 6. Write the output file
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=4)
        
        logger.info(f"Processing complete. Output saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        # Create minimal output for error case
        error_output = {
            "Metadata": {
                "input_documents": [],
                "persona": persona if 'persona' in locals() else "Unknown",
                "job_to_be_done": job_to_be_done if 'job_to_be_done' in locals() else "Unknown",
                "processing_timestamp": datetime.utcnow().isoformat() + "Z",
                "error": str(e)
            },
            "Extracted Section": [],
            "Sub-section Analysis": []
        }
        
        with open(output_file, 'w') as f:
            json.dump(error_output, f, indent=4)

if __name__ == '__main__':
    main()
