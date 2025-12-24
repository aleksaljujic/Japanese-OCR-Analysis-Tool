from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import os
from pathlib import Path
import sys
import cv2
import numpy as np
from typing import Optional
import tempfile
import shutil
from paddleocr import PaddleOCR
from preprocessing import crop_image, setThresholdSauvola
from load_data import extract_text_from_json 

app = FastAPI()

# OCR model
ocr = PaddleOCR(
    lang='japan',
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True,
    det_db_thresh=0.5, 
    det_db_box_thresh=0.6,
    rec_batch_num=1
)

# Pydantic models
class OCRPathRequest(BaseModel):
    img_path: str
    gt_path: Optional[str] = None

class OCRResponse(BaseModel):
    status: str
    ocr_text: str
    ground_truth: Optional[str] = None

def run_ocr_th(img_path):
    img = cv2.imread(img_path)
    cropped_img = crop_image(img)
    bin_img = setThresholdSauvola(cropped_img)

    if len(bin_img.shape) == 2:
        bin_img = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
    
    result = ocr.predict(bin_img)
    
    all_texts = []
    for res in result:
        json_result = res.json
        rec_texts = json_result['res']['rec_texts']
        all_texts.extend(rec_texts)

    full_text = '\n'.join(all_texts)
    return full_text

# Endpoint 1:
@app.post("/ocr/from-path", response_model=OCRResponse)
async def ocr_from_path(request: OCRPathRequest):
    """
    OCR paths.
    
    Body:
    {
        "img_path": "path/to/image.jpg",
        "gt_path": "path/to/ground_truth.json"  // optional
    }
    """
    try:
        if not os.path.exists(request.img_path):
            raise HTTPException(status_code=404, detail=f"Image not found: {request.img_path}")
        
        ocr_text = run_ocr_th(request.img_path)
        
        ground_truth = None
        if request.gt_path:
            if not os.path.exists(request.gt_path):
                raise HTTPException(status_code=404, detail=f"Ground truth not found: {request.gt_path}")
            ground_truth = extract_text_from_json(request.gt_path)
        
        return OCRResponse(
            status="success",
            ocr_text=ocr_text,
            ground_truth=ground_truth
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint 2: Upload 
@app.post("/ocr/upload", response_model=OCRResponse)
async def ocr_from_upload(
    image: UploadFile = File(...),
    ground_truth: Optional[UploadFile] = File(None)
):
    """
    OCR sa upload-ovanim fajlovima.
    
    Form-data:
    - image: slika (JPEG/PNG)
    - ground_truth: JSON fajl (optional)
    """
    temp_img_path = None
    temp_gt_path = None
    
    try:
        # Save temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename)[1]) as temp_img:
            shutil.copyfileobj(image.file, temp_img)
            temp_img_path = temp_img.name
        
        # Run OCR
        ocr_text = run_ocr_th(temp_img_path)
        
        # GT = True
        gt_text = None
        if ground_truth:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_gt:
                shutil.copyfileobj(ground_truth.file, temp_gt)
                temp_gt_path = temp_gt.name
            gt_text = extract_text_from_json(temp_gt_path)
        
        return OCRResponse(
            status="success",
            ocr_text=ocr_text,
            ground_truth=gt_text
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Delete temp
        if temp_img_path and os.path.exists(temp_img_path):
            os.unlink(temp_img_path)
        if temp_gt_path and os.path.exists(temp_gt_path):
            os.unlink(temp_gt_path)

@app.get("/")
async def root():
    return {"message": "Japanese OCR API is running"}

# Run: uvicorn api:app --reload