# download_model.py

from sentence_transformers import SentenceTransformer
import os

def download_model():
    """Download and save the model locally."""
    model_name = 'all-MiniLM-L6-v2'
    model_dir = './model'
    
    print(f"Downloading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"Saving model to: {model_dir}")
    model.save(model_dir)
    
    print("Model download complete!")
    
    # Verify model size
    model_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(model_dir)
                    for filename in filenames) / (1024 * 1024)  # MB
    
    print(f"Model size: {model_size:.2f} MB")
    
    if model_size > 1000:  # 1GB limit
        print("WARNING: Model size exceeds 1GB limit!")
    else:
        print("Model size is within limits.")

if __name__ == '__main__':
    download_model()
