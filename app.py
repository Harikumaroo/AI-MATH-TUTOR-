"""Streamlit front-end for AI Math Tutor - Image-Based Algebra Solver with Dynamic Themes"""
import streamlit as st
from PIL import Image
import io
import os
import logging

from utils.image_utils import preprocess_for_ocr, to_bytes
from utils.env_utils import get_api_key
from vision.ocr import OCREngine, llm_convert_to_latex
from solver.equation_solver import parse_latex_to_sympy, solve_equation, generate_steps
from checker.mistake_checker import detect_mistakes

logging.getLogger().setLevel(logging.INFO)

# Page Configuration
st.set_page_config(
    page_title="AI Math Tutor — Image Solver",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Themes
THEME_CSS = {
    "🌌 Dark AI Studio": """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        h1, h2, h3, h4 {
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
        }
        .main {
            background: linear-gradient(135deg, #0B0F19 0%, #111827 50%, #0F172A 100%);
            color: #F8FAFC;
        }
        .stSidebar {
            background-color: #1E293B !important;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }
        .hero-card {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.15) 100%);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            backdrop-filter: blur(8px);
        }
        .solution-card {
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 14px;
            padding: 20px;
            margin-bottom: 16px;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        }
        .step-box {
            background: rgba(15, 23, 42, 0.6);
            border-left: 4px solid #6366F1;
            border-radius: 8px;
            padding: 14px 18px;
            margin-bottom: 12px;
        }
        .pill-badge {
            background: linear-gradient(90deg, #6366F1 0%, #8B5CF6 100%);
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-block;
            margin-bottom: 8px;
        }
        .stButton>button {
            background: linear-gradient(90deg, #6366F1 0%, #4F46E5 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            padding: 8px 20px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
        }
    </style>
    """,
    
    "📜 Academic Light": """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        h1, h2, h3, h4 {
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
        }
        .main {
            background-color: #F8FAFC;
            color: #0F172A;
        }
        .stSidebar {
            background-color: #FFFFFF !important;
            border-right: 1px solid #E2E8F0;
        }
        .hero-card {
            background: linear-gradient(135deg, #EFF6FF 0%, #EEF2FF 100%);
            border: 1px solid #BFDBFE;
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }
        .solution-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 20px;
            margin-bottom: 16px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
        }
        .step-box {
            background: #F1F5F9;
            border-left: 4px solid #2563EB;
            border-radius: 8px;
            padding: 14px 18px;
            margin-bottom: 12px;
        }
        .pill-badge {
            background: #2563EB;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-block;
            margin-bottom: 8px;
        }
        .stButton>button {
            background: #2563EB;
            color: white;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            padding: 8px 20px;
        }
    </style>
    """,
    
    "🔮 Cyberpunk Neon": """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;600&family=Outfit:wght@600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Fira Code', monospace;
        }
        h1, h2, h3, h4 {
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
        }
        .main {
            background-color: #070913;
            color: #00F0FF;
        }
        .stSidebar {
            background-color: #0D111E !important;
            border-right: 1px solid #00F0FF;
        }
        .hero-card {
            background: rgba(13, 17, 30, 0.9);
            border: 2px solid #00F0FF;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 0 20px rgba(0, 240, 255, 0.3);
        }
        .solution-card {
            background: #0D111E;
            border: 1px solid #A855F7;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
            box-shadow: 0 0 15px rgba(168, 85, 247, 0.2);
        }
        .step-box {
            background: #141A2E;
            border-left: 4px solid #00F0FF;
            border-radius: 6px;
            padding: 14px 18px;
            margin-bottom: 12px;
        }
        .pill-badge {
            background: #A855F7;
            color: #00F0FF;
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 0.85rem;
            font-weight: 700;
            border: 1px solid #00F0FF;
            display: inline-block;
            margin-bottom: 8px;
        }
        .stButton>button {
            background: transparent;
            color: #00F0FF;
            border: 2px solid #00F0FF;
            border-radius: 6px;
            font-weight: 700;
            padding: 8px 20px;
            box-shadow: 0 0 10px rgba(0, 240, 255, 0.4);
        }
        .stButton>button:hover {
            background: #00F0FF;
            color: #070913;
            box-shadow: 0 0 25px rgba(0, 240, 255, 0.8);
        }
    </style>
    """
}

def main():
    # Sidebar Theme Control
    st.sidebar.markdown("### 🎨 Theme Selector")
    theme_choice = st.sidebar.selectbox(
        "Choose App Aesthetic",
        options=list(THEME_CSS.keys()),
        index=0
    )
    
    # Inject Theme CSS
    st.markdown(THEME_CSS[theme_choice], unsafe_allow_html=True)
    
    # Hero Header Section
    st.markdown(
        """
        <div class="hero-card">
            <span class="pill-badge">⚡ AI-POWERED SOLVER</span>
            <h1 style="margin:0; padding-top:4px;">🧮 AI Math Tutor</h1>
            <p style="margin-top:8px; opacity:0.85; font-size:1.05rem;">
                Upload or capture an image of an algebra equation. The tutor extracts the math via OCR, standardizes LaTeX, solves symbolically via SymPy, and checks for mistakes step-by-step.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sidebar Controls
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📥 Input Source")
    uploaded = st.sidebar.file_uploader("Upload Problem Image", type=["jpg", "jpeg", "png"])
    cam = st.sidebar.camera_input("Capture from Camera")
    
    st.sidebar.markdown("---")
    has_llm = bool(get_api_key("GEMINI_API_KEY") or get_api_key("GROQ_API_KEY") or get_api_key("OPENAI_API_KEY"))
    use_llm = st.sidebar.checkbox("Enable LLM LaTeX Conversion", value=has_llm, help="Uses AI Vision / LLM to accurately convert complex images to LaTeX")
    
    # Check sample images
    sample_dir = "samples" if os.path.exists("samples") else "."
    sample_files = [f for f in ["eq_1.png", "eq_2.png", "eq_3.png", "eq_4.png"] if os.path.exists(os.path.join(sample_dir, f))]
    
    img = None
    selected_sample = None
    
    if uploaded is not None:
        img = Image.open(uploaded)
    elif cam is not None:
        img = Image.open(io.BytesIO(cam.getvalue()))

    # Sample Preset Selector if no image uploaded
    if img is None and sample_files:
        st.markdown("#### 💡 Or try a sample problem preset:")
        cols = st.columns(len(sample_files))
        for idx, sample_name in enumerate(sample_files):
            with cols[idx]:
                sample_path = os.path.join(sample_dir, sample_name)
                sample_img = Image.open(sample_path)
                st.image(sample_img, caption=sample_name, use_container_width=True)
                if st.button(f"Solve {sample_name}", key=f"btn_{sample_name}"):
                    img = sample_img
                    selected_sample = sample_name

    if img is None:
        st.info("👆 Upload an image, capture via camera, or select a sample equation above to begin.")
        
        # Sidebar Status Footer
        st.sidebar.markdown("---")
        has_gemini = bool(get_api_key("GEMINI_API_KEY"))
        has_groq = bool(get_api_key("GROQ_API_KEY"))
        has_openai = bool(get_api_key("OPENAI_API_KEY"))
        st.sidebar.markdown(f"**LLM API Status:**")
        st.sidebar.markdown(f"- GEMINI_API_KEY: {'✅ Active' if has_gemini else '❌ Not Set'}")
        st.sidebar.markdown(f"- GROQ_API_KEY: {'✅ Active' if has_groq else '❌ Not Set'}")
        st.sidebar.markdown(f"- OPENAI_API_KEY: {'✅ Active' if has_openai else '❌ Not Set'}")
        return

    # 2-Column Main Workspace
    left_col, right_col = st.columns([1, 1.2], gap="large")

    with left_col:
        st.markdown("<div class='solution-card'>", unsafe_allow_html=True)
        st.subheader("📷 Image & OCR Breakdown")
        st.image(img, caption="Input Problem Image", use_container_width=True)

        with st.spinner("Processing OCR & Math Extraction..."):
            pre = preprocess_for_ocr(img)
            ocr_engine = OCREngine()
            raw_text = ocr_engine.extract_text(pre)
            math_expr = ocr_engine.extract_math(pre)

        st.markdown("**Raw OCR Output:**")
        st.text_area("Extracted Text", value=raw_text, height=90, label_visibility="collapsed")

        st.markdown("**Detected Expression (Heuristic):**")
        st.text_input("Cleaned Expression", value=math_expr, label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown("<div class='solution-card'>", unsafe_allow_html=True)
        st.subheader("🎯 Solution & Step-by-Step Solver")

        latex = None
        if use_llm:
            with st.spinner("Converting to LaTeX via LLM..."):
                latex = llm_convert_to_latex(raw_text, image=to_bytes(pre))
        else:
            latex = math_expr.strip()

        st.markdown("**LaTeX Representation:**")
        try:
            st.latex(latex)
        except Exception:
            st.code(latex, language="latex")

        # Solve via SymPy
        sym = None
        try:
            sym = parse_latex_to_sympy(latex)
        except Exception as e:
            st.warning(f"⚠️ Could not parse formula into a 1-variable SymPy algebraic equation: {e}")
            st.info("ℹ️ The formula/function above is rendered in LaTeX. Automatic root-solving applies to standard single-variable equations (e.g. 2x + 5 = 15).")

        if sym is not None:
            with st.spinner("Solving equation..."):
                result = solve_equation(sym)
                steps = generate_steps(sym)
                mistakes = detect_mistakes(latex, sym)

            # Final Answer Box
            sols = result.get("solutions")
            st.markdown("<div style='background:rgba(99, 102, 241, 0.15); border:1px solid #6366F1; border-radius:10px; padding:16px; margin:16px 0;'>", unsafe_allow_html=True)
            st.markdown("### 🏆 Final Answer")
            if sols is None or len(sols) == 0:
                st.warning("No explicit symbolic solution found.")
            else:
                st.markdown(f"**Root Solution(s):** `{sols}`")
                for sol in sols:
                    try:
                        st.latex(f"x = {sol}")
                    except Exception:
                        st.write(sol)
            st.markdown("</div>", unsafe_allow_html=True)

            # Step-by-step Solution Cards
            st.markdown("### 📝 Step-by-Step Breakdown")
            for i, (title, content) in enumerate(steps, start=1):
                st.markdown(
                    f"""
                    <div class="step-box">
                        <strong>Step {i}: {title}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                try:
                    st.latex(content)
                except Exception:
                    st.write(content)

            # Mistake Checker Warnings
            st.markdown("### 🔍 Mistake Analysis")
            if mistakes:
                st.error("Potential common errors flagged:")
                for m in mistakes:
                    st.markdown(f"- ⚠️ {m}")
            else:
                st.success("✅ No obvious algebraic errors or inconsistencies detected.")

        st.markdown("</div>", unsafe_allow_html=True)

    # Sidebar Status Footer
    st.sidebar.markdown("---")
    has_gemini = bool(get_api_key("GEMINI_API_KEY"))
    has_groq = bool(get_api_key("GROQ_API_KEY"))
    has_openai = bool(get_api_key("OPENAI_API_KEY"))
    st.sidebar.markdown(f"**LLM API Status:**")
    st.sidebar.markdown(f"- GEMINI_API_KEY: {'✅ Active' if has_gemini else '❌ Not Set'}")
    st.sidebar.markdown(f"- GROQ_API_KEY: {'✅ Active' if has_groq else '❌ Not Set'}")
    st.sidebar.markdown(f"- OPENAI_API_KEY: {'✅ Active' if has_openai else '❌ Not Set'}")

if __name__ == '__main__':
    main()
