import streamlit as st
import requests
import json
from PIL import Image
import tempfile
import os
from jiwer import cer, wer
import difflib

def save_uploaded_file(uploaded_file):
    """Sačuvaj upload-ovani fajl privremeno i vrati putanju"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def extract_text_from_json(json_file):
    """Ekstraktuj tekst iz JSON fajla"""
    data = json.load(json_file)
    
    if isinstance(data, dict):
        if 'text' in data:
            return data['text']
        elif 'annotations' in data:
            texts = [ann.get('text', '') for ann in data['annotations']]
            return '\n'.join(texts)
    
    return str(data)

def calculate_metrics(ocr_text, ground_truth):
    """Izračunaj OCR metrike"""
    # Character Error Rate
    cer_score = cer(ground_truth, ocr_text)
    
    # Word Error Rate
    wer_score = wer(ground_truth, ocr_text)
    
    # Accuracy (1 - CER)
    accuracy = max(0, 1 - cer_score) * 100
    
    # Character-level precision
    gt_chars = set(ground_truth)
    ocr_chars = set(ocr_text)
    if len(ocr_chars) > 0:
        precision = len(gt_chars & ocr_chars) / len(ocr_chars) * 100
    else:
        precision = 0
    
    # Similarity ratio using difflib
    similarity = difflib.SequenceMatcher(None, ground_truth, ocr_text).ratio() * 100
    
    return {
        "cer": cer_score,
        "wer": wer_score,
        "accuracy": accuracy,
        "precision": precision,
        "similarity": similarity
    }

OCR_URL = os.getenv("OCR_URL", "http://localhost:8000")
AGENT_URL = os.getenv("AGENT_URL", "http://localhost:8001")

def call_ocr_api(uploaded_image, uploaded_json=None):
    """Šalje fajlove direktno kao bajtove OCR API-ju"""
    url = f"{OCR_URL}/ocr/upload"
    
    # Priprema fajlova za slanje (Multipart/form-data)
    files = {
        'image': (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)
    }
    
    if uploaded_json:
        files['ground_truth'] = (uploaded_json.name, uploaded_json.getvalue(), 'application/json')
        
    response = requests.post(url, files=files, timeout=60)
    return response.json()

def call_analyze_api(ocr_text, ground_truth):
    """Pozovi Analyze API"""
    url = f"{AGENT_URL}/analyze-ocr"
    response = requests.post(
        url,
        json={"ocr_text": ocr_text, "ground_truth": ground_truth},
        timeout=60
    )
    return response.json()["analysis"]

# Streamlit UI
st.set_page_config(page_title="Japanese OCR Analysis", layout="wide")

st.title("Japanese OCR Analysis Tool")
st.markdown("---")

# Sidebar za upload
with st.sidebar:
    st.header("📁 Upload Files")
    
    uploaded_image = st.file_uploader(
        "Upload Image (JPEG/PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload Japanese ID card image"
    )
    
    uploaded_json = st.file_uploader(
        "Upload Ground Truth (JSON) - Optional",
        type=["json"],
        help="Upload JSON file with ground truth text (optional for analysis)"
    )
    
    st.markdown("---")
    
    run_button = st.button("RUN", type="primary", use_container_width=True)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 Input Image")
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, use_container_width=True)
    else:
        st.info("👈 Upload an image to get started")

with col2:
    st.subheader("📄 Ground Truth")
    if uploaded_json:
        try:
            uploaded_json.seek(0)
            gt_text = extract_text_from_json(uploaded_json)
            st.text_area("Ground Truth Text", gt_text, height=200, disabled=True)
        except Exception as e:
            st.error(f"Error reading JSON: {str(e)}")
    else:
        st.info("👈 Upload ground truth JSON file (optional)")

# Processing
if run_button:
    if not uploaded_image:
        st.error("❌ Please upload an image first!")
    else:
        try:
            # VIŠE NE SAČUVAVAMO FAJLOVE PRIVREMENO
            # Pozovi OCR API direktno sa objektima iz Streamlita
            with st.spinner("🔄 Running OCR..."):
                # Menjamo poziv: šaljemo direktno uploaded_image i uploaded_json
                ocr_result = call_ocr_api(uploaded_image, uploaded_json)
                
                if "ocr_text" not in ocr_result:
                    st.error(f"OCR failed: {ocr_result}")
                else:
                    ocr_text = ocr_result["ocr_text"]
                    gt_text = ocr_result.get("ground_truth")
                    
                    st.success("✅ OCR completed!")
                    
                    # --- Ostatak koda za Results (header, tabovi...) ostaje ISTI ---
                    st.markdown("---")
                    st.header("📊 Results")
                    # ... nastavi sa tvojim kodom ...
                    # Ako nema ground truth, prikaži samo OCR output
                    if gt_text is None:
                        st.subheader("OCR Generated Text")
                        st.text_area("OCR Result", ocr_text, height=400, key="ocr_output")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Characters Detected", len(ocr_text))
                        with col_b:
                            st.metric("Lines Detected", len(ocr_text.split('\n')))
                        
                        st.info("💡 Upload ground truth JSON to enable comparison and AI analysis")
                    
                    # Ako ima ground truth, prikaži sve tabove
                    else:
                        # Izračunaj metrike
                        metrics = calculate_metrics(ocr_text, gt_text)
                        
                        tab1, tab2, tab3, tab4 = st.tabs(["OCR Output", "Metrics", "Comparison", "AI Analysis"])
                        
                        with tab1:
                            st.subheader("OCR Generated Text")
                            st.text_area("OCR Result", ocr_text, height=300, key="ocr_output")
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Characters Detected", len(ocr_text))
                            with col_b:
                                st.metric("Lines Detected", len(ocr_text.split('\n')))
                        
                        with tab2:
                            st.subheader("📈 Performance Metrics")
                            
                            # Metrike u kolone
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    "Character Error Rate (CER)", 
                                    f"{metrics['cer']:.4f}",
                                    delta=f"{-metrics['cer']:.4f}" if metrics['cer'] < 0.5 else None,
                                    delta_color="inverse"
                                )
                                st.caption("Lower is better. CER < 0.1 is excellent")
                            
                            with col2:
                                st.metric(
                                    "Accuracy", 
                                    f"{metrics['accuracy']:.2f}%",
                                    delta=f"{metrics['accuracy'] - 50:.1f}%" if metrics['accuracy'] > 50 else None
                                )
                                st.caption("Higher is better. Based on 1 - CER")
                            
                            with col3:
                                st.metric(
                                    "Similarity", 
                                    f"{metrics['similarity']:.2f}%",
                                    delta=f"{metrics['similarity'] - 50:.1f}%" if metrics['similarity'] > 50 else None
                                )
                                st.caption("SequenceMatcher ratio. 100% = perfect match")
                            
                            col4, col5 = st.columns(2)
                            
                            with col4:
                                st.metric(
                                    "Word Error Rate (WER)", 
                                    f"{metrics['wer']:.4f}",
                                    delta=f"{-metrics['wer']:.4f}" if metrics['wer'] < 0.5 else None,
                                    delta_color="inverse"
                                )
                                st.caption("Lower is better. Measures word-level errors")
                            
                            with col5:
                                st.metric(
                                    "Character Precision", 
                                    f"{metrics['precision']:.2f}%"
                                )
                                st.caption("% of detected characters that are correct")
                            
                            # Progress bars
                            st.markdown("---")
                            st.subheader("Visual Metrics")
                            
                            st.write("**Accuracy**")
                            st.progress(metrics['accuracy'] / 100)
                            
                            st.write("**Similarity**")
                            st.progress(metrics['similarity'] / 100)
                            
                            # Interpretacija
                            st.markdown("---")
                            st.subheader("📊 Interpretation")
                            
                            if metrics['cer'] < 0.05:
                                st.success("🎉 Excellent OCR quality! CER < 0.05")
                            elif metrics['cer'] < 0.1:
                                st.success("✅ Good OCR quality. CER < 0.1")
                            elif metrics['cer'] < 0.2:
                                st.warning("⚠️ Moderate OCR quality. CER < 0.2")
                            else:
                                st.error("❌ Poor OCR quality. CER > 0.2")
                        
                        with tab3:
                            col_left, col_right = st.columns(2)
                            
                            with col_left:
                                st.markdown("**OCR Text**")
                                st.code(ocr_text, language=None)
                            
                            with col_right:
                                st.markdown("**Ground Truth**")
                                st.code(gt_text, language=None)
                            
                            # Diff visualization
                            st.markdown("---")
                            st.subheader("🔍 Character-by-Character Diff")
                            
                            diff = difflib.ndiff(ocr_text, gt_text)
                            diff_html = []
                            for i, s in enumerate(diff):
                                if s[0] == ' ':
                                    diff_html.append(s[-1])
                                elif s[0] == '-':
                                    diff_html.append(f'<span style="background-color: #b31102; text-decoration: line-through;">{s[-1]}</span>')
                                elif s[0] == '+':
                                    diff_html.append(f'<span style="background-color: #03ab03;">{s[-1]}</span>')
                            
                            st.markdown(''.join(diff_html), unsafe_allow_html=True)
                            st.caption("🔴 Red = Missing from OCR | 🟢 Green = Extra in OCR")
                        
                        with tab4:
                            st.subheader("🤖 AI-Powered Analysis")
                            
                            with st.spinner("Analyzing with GPT-4..."):
                                try:
                                    analysis = call_analyze_api(ocr_text, gt_text)
                                    st.markdown(analysis)
                                    
                                    # Download button
                                    st.download_button(
                                        label="📥 Download Analysis",
                                        data=analysis,
                                        file_name="ocr_analysis.txt",
                                        mime="text/plain"
                                    )
                                except Exception as e:
                                    st.error(f"Analysis API error: {str(e)}")
                                    st.info("💡 Make sure Analysis API is running on port 8002")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        Made by Aleksa | Powered by PaddleOCR & GPT-4o
    </div>
    """,
    unsafe_allow_html=True
)