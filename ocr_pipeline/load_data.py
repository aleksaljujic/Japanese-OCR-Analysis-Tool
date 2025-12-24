import pandas as pd
import json
from pathlib import Path

def load_dataset(archive_path):
    archive = Path(archive_path)
    data = []
    
    for folder in archive.iterdir():
        if not folder.is_dir():
            continue
        
        # image (jpg, jpeg, png)
        img_path = None
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            imgs = list(folder.glob(ext))
            if imgs:
                img_path = imgs[0]
                break
        
        # JSON
        json_files = list(folder.glob('*.json'))
        json_path = json_files[0] if json_files else None
        
        if img_path and json_path:
            # Load ground truth
            ground_truth = extract_text_from_json(json_path)
            
            data.append({
                'folder_name': folder.name,
                'image_path': str(img_path),
                'json_path': str(json_path),
                'ground_truth': ground_truth
            })
    
    df = pd.DataFrame(data)
    return df

def extract_text_from_json(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        labels = []
        for shape in data.get("shapes", []):
            text = shape.get("label")
            if text:
                labels.append(text)

        # Merge
        return "\n".join(labels)

    except Exception as e:
        print("JSON ERROR:", json_path, e)
        return None


