"""
Micro-Scale Sentinel - Streamlit Application
AI-powered microplastic detection for holographic microscopy images.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import pandas as pd
import json
import os
import base64
from datetime import datetime, timedelta
# Import our custom modules
from src.preprocessing import preprocess_image, extract_features
from src.classifier import classify_particle

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Micro-Scale Sentinel 🔬",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stProgress > div > div > div > div {background-color: #1f77b4;}
    .success-box {padding: 20px; border-radius: 10px; background-color: #d4edda; border: 1px solid #c3e6cb;}
    .info-box {padding: 15px; border-radius: 8px; background-color: #d1ecf1; border: 1px solid #bee5eb; margin: 10px 0;}
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
if 'results' not in st.session_state:
    st.session_state.results = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'features' not in st.session_state:
    st.session_state.features = None

# =============================================================================
# SIDEBAR - CONFIGURATION & EDUCATION
# =============================================================================
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")
    
    # Load API Key from secrets (hidden from users)
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ AI Service Connected", icon="🔒")
    except Exception as e:
        st.error("⚠️ API Configuration Missing")
        st.info("Admin: Please configure GEMINI_API_KEY in Streamlit Secrets")
        st.stop()  # Stop app if no API key
    
    st.markdown("---")
    
    # Image Upload
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Drop holographic microscopy image",
        type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
        help="Supported: PNG, JPG, TIF. Recommended size: 512x512 to 2048x2048 pixels"
    )
    
    # OR Generate Sample
    st.markdown("**OR**")
    use_sample = st.button("🎲 Generate Test Image", help="Create a synthetic particle for testing")
    
    st.markdown("---")
    
    # Analysis Settings
    st.subheader("🔧 Analysis Settings")
    scale_um = st.number_input(
        "Scale (μm per pixel)", 
        value=0.5, 
        min_value=0.1, 
        max_value=10.0,
        help="Calibration: how many microns does one pixel represent?"
    )
    
    debug_mode = st.checkbox(
        "🔍 Debug Mode", 
        value=False,
        help="Show intermediate processing steps"
    )
    
    # Educational sections remain the same...
    st.markdown("---")
    
    # Educational Section: Physics Reference
    with st.expander("📚 Physics Reference", expanded=False):
        st.markdown("""
        **Refractive Index (RI) Guide:**
        
        *Plastics (higher RI):*
        - PET: 1.575 (water bottles)
        - PS: 1.55 (styrofoam)
        - HDPE: 1.54 (containers)
        - PVC: 1.54 (pipes)
        - PP: 1.49 (ropes)
        
        *Biological (lower RI):*
        - Diatoms: 1.35-1.40 (glass shells)
        - Copepods: 1.33-1.38 (transparent)
        - Water: 1.33 (reference)
        
        **Why this matters:** Light bends differently through plastic vs organisms, creating distinct halo patterns in holographic images.
        """)
    
    with st.expander("❓ What are Microplastics?", expanded=False):
        st.markdown("""
        Microplastics are plastic particles <5mm in size. In environmental monitoring:
        
        - **Primary**: Manufactured small (microbeads, nurdles)
        - **Secondary**: Breakdown of larger plastics
        
        **Detection challenge:** They look similar to organic particles (algae, sediment) under microscopes. This AI uses **physics-based features** (refractive index, shape) to distinguish them.
        """)
# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_sample_image():
    """Generates a synthetic holographic image for testing."""
    size = 512
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Add background noise
    noise = np.random.normal(128, 15, (size, size)).astype(np.uint8)
    img = cv2.addWeighted(img, 0, noise, 1, 0)
    
    # Draw a particle with interference fringes
    center = (size // 2, size // 2)
    radius = 40
    
    # Main particle body
    cv2.circle(img, center, radius, 180, -1)
    cv2.circle(img, center, radius, 100, 2)
    
    # Add interference rings
    for r in range(radius + 10, radius + 60, 8):
        cv2.circle(img, center, r, 140, 1)
    
    # Add asymmetry
    cv2.ellipse(img, (center[0] + 10, center[1] - 5), (radius-5, radius-10), 
                30, 0, 360, 160, 2)
    
    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    return img_rgb, "synthetic_microplastic.png"

def get_image_download_link(img_array, filename, text):
    """Creates a download link for processed images"""
    pil_img = Image.fromarray(img_array)
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

# =============================================================================
# MAIN APPLICATION AREA
# =============================================================================
st.title("🔬 Micro-Scale Sentinel")
st.markdown("*AI-powered microplastic detection for holographic microscopy*")
st.markdown("---")

# Handle Image Input
image = None
filename = ""

if use_sample:
    image, filename = create_sample_image()
    st.success("✅ Generated synthetic test image (simulated plastic particle with interference patterns)")
elif uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    filename = uploaded_file.name
    st.success(f"✅ Loaded: {filename}")

if image is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Original Image")
        st.image(image, use_container_width=True, caption=f"Input: {filename}")
        h, w = image.shape[:2]
        st.caption(f"Dimensions: {w}×{h} pixels | Scale: {scale_um} μm/pixel | FOV: {w*scale_um:.1f}×{h*scale_um:.1f} μm")
    
    with col2:
        st.subheader("⚙️ Preprocessing")
        if debug_mode:
            with st.spinner("Enhancing image..."):
                processed = preprocess_image(image, debug=True)
                st.image(processed, use_container_width=True, caption="CLAHE Enhanced + Denoised")
        else:
            processed = preprocess_image(image, debug=False)
            st.image(processed, use_container_width=True, caption="Enhanced Contrast & Noise Reduced")
        st.session_state.processed_image = processed
    
    st.markdown("---")
    st.subheader("🤖 AI Analysis")
    if 'last_analysis' in st.session_state:
    time_diff = datetime.now() - st.session_state['last_analysis']
    if time_diff.seconds < 5:  # 5 second cooldown
        st.warning("Please wait a few seconds between analyses")
        st.stop()
    
analyze_btn = st.button("🔍 Analyze Particle", type="primary", use_container_width=True)
        
        if analyze_btn:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Step 1/3: Extracting physics-based features...")
                progress_bar.progress(25)
                
                features = extract_features(processed, scale_um_per_pixel=scale_um)
                st.session_state.features = features
                
                if debug_mode:
                    with st.expander("🔍 Extracted Features"):
                        st.json(features)
                
                status_text.text("Step 2/3: Gemini AI analyzing optical properties...")
                progress_bar.progress(60)
                
                result = classify_particle(processed, features, api_key)
                st.session_state.results = result
                
                status_text.text("Step 3/3: Finalizing results...")
                progress_bar.progress(100)
                
                status_text.empty()
                progress_bar.empty()
                st.success("✅ Analysis Complete!")
                
            except Exception as e:
                status_text.empty()
                progress_bar.empty()
                st.error(f"❌ Analysis failed: {str(e)}")
                st.info("💡 **Troubleshooting:** Check your API key, ensure image contains visible particles, or try Debug Mode.")
    
    if st.session_state.results is not None:
        result = st.session_state.results
        features = st.session_state.features
        
        st.markdown("---")
        st.subheader("📊 Classification Results")
        
        mcol1, mcol2, mcol3 = st.columns(3)
        
        with mcol1:
            classification = result.get('classification', 'UNCERTAIN')
            if classification == "MICROPLASTIC":
                st.metric("Classification", "🔴 MICROPLASTIC", "Pollutant Detected")
            elif classification == "BIOLOGICAL":
                st.metric("Classification", "🟢 BIOLOGICAL", "Natural Particle")
            else:
                st.metric("Classification", "🟡 UNCERTAIN", "Needs Review")
        
        with mcol2:
            conf_mp = result.get('confidence_microplastic', 0)
            st.metric("Plastic Confidence", f"{conf_mp}%")
        
        with mcol3:
            conf_bio = result.get('confidence_biological', 0)
            st.metric("Biological Confidence", f"{conf_bio}%")
        
        tab1, tab2, tab3 = st.tabs(["📝 Detailed Report", "🔬 Technical Evidence", "📥 Export Data"])
        
        with tab1:
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.markdown("#### AI Reasoning")
                reasoning = result.get('reasoning', 'No reasoning provided')
                st.info(reasoning)
                rec = result.get('recommendation', 'UNCERTAIN')
                if rec == "DEFINITE":
                    st.success("🎯 **Recommendation:** Definite classification - high confidence")
                elif rec == "PROBABLE":
                    st.warning("⚠️ **Recommendation:** Probable classification - review suggested")
                else:
                    st.error("❓ **Recommendation:** Uncertain - manual expert review required")
            
            with col_b:
                st.markdown("#### Identified Type")
                if classification == "MICROPLASTIC":
                    poly_type = result.get('polymer_type', 'Unknown')
                    st.write(f"**Polymer:** {poly_type}")
                    st.caption("Common sources: bottles, packaging, textiles")
                elif classification == "BIOLOGICAL":
                    org_type = result.get('organism_type', 'Unknown')
                    st.write(f"**Organism:** {org_type}")
                    st.caption("Natural environmental particle")
        
        with tab2:
            st.markdown("#### Physics-Based Evidence")
            evidence = result.get('evidence', {})
            if evidence:
                ecol1, ecol2 = st.columns(2)
                with ecol1:
                    st.write("**Diffraction Pattern:**")
                    st.write(evidence.get('diffraction_pattern', 'N/A'))
                    st.write("**Morphology Analysis:**")
                    st.write(evidence.get('morphology', 'N/A'))
                with ecol2:
                    st.write("**Refractive Index Analysis:**")
                    st.write(evidence.get('refractive_index_analysis', 'N/A'))
                    st.write("**Size Analysis:**")
                    st.write(evidence.get('size_analysis', 'N/A'))
            
            if features:
                st.markdown("#### Quantified Features")
                fcol1, fcol2, fcol3, fcol4 = st.columns(4)
                with fcol1:
                    st.metric("Size", f"{features.get('size_um', 0):.1f} μm")
                with fcol2:
                    st.metric("Circularity", f"{features.get('circularity', 0):.2f}")
                with fcol3:
                    st.metric("RI Estimate", f"{features.get('refractive_index_estimate', 0):.3f}")
                with fcol4:
                    st.metric("Aspect Ratio", f"{features.get('aspect_ratio', 0):.2f}")
        
        with tab3:
            st.markdown("#### Download Results")
            if features and result:
                export_data = {
                    'filename': filename,
                    'timestamp': datetime.now().isoformat(),
                    'classification': result.get('classification'),
                    'confidence_plastic_pct': result.get('confidence_microplastic'),
                    'confidence_biological_pct': result.get('confidence_biological'),
                    'polymer_type': result.get('polymer_type'),
                    'organism_type': result.get('organism_type'),
                    'size_um': features.get('size_um'),
                    'circularity': features.get('circularity'),
                    'aspect_ratio': features.get('aspect_ratio'),
                    'refractive_index_estimate': features.get('refractive_index_estimate'),
                    'reasoning': result.get('reasoning')
                }
                df = pd.DataFrame([export_data])
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV Report",
                    data=csv,
                    file_name=f"analysis_{filename.split('.')[0]}.csv",
                    mime='text/csv'
                )
                st.dataframe(df.T, use_container_width=True)

else:
    st.info("👈 **Get started:** Upload a holographic microscopy image or generate a test sample in the sidebar")
    st.markdown("""
    ### How It Works
    
    1. **🔬 Physics Principle**: Plastics have higher refractive index (1.4-1.6) than water/organisms (1.33-1.40)
       - This creates distinct diffraction halos in holographic images
    
    2. **🤖 AI Reasoning**: Gemini analyzes optical interference patterns, morphology, and transparency
    
    3. **📊 Classification**: Returns confidence scores for plastic vs biological origin
    """)

st.markdown("---")
st.caption("🔬 Micro-Scale Sentinel v1.0 | Built for Environmental Engineering Education | Powered by Google Gemini & Streamlit")
