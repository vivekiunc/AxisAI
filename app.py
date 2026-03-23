"""
Streamlit Web App — Postural Cobb Angle Estimator
Run with: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os

from cobb_angle_estimator import CobbAngleEstimator
from report_generator import generate_pdf_report

# ─── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Spinal Curvature Screener",
    page_icon="🦴",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0d1117; color: #e2e8f0; }
.metric-card {
    background: linear-gradient(135deg, #161b27 0%, #1a2035 100%);
    border: 1px solid #2a3050;
    border-radius: 14px;
    padding: 20px 24px;
    margin: 6px 0;
}
.metric-label {
    font-size: 11px; font-weight: 600; color: #6b7a99;
    text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 6px;
}
.metric-value { font-size: 36px; font-weight: 700; color: #f0f4ff; line-height: 1.1; }
.metric-unit { font-size: 16px; font-weight: 400; color: #8898bb; }
.severity-badge {
    display: inline-block; padding: 8px 20px; border-radius: 999px;
    font-size: 14px; font-weight: 700; letter-spacing: 0.5px; margin-top: 6px;
}
.section-header {
    font-size: 13px; font-weight: 600; color: #4f8ef7;
    text-transform: uppercase; letter-spacing: 2px; margin: 24px 0 10px 0;
}
.disclaimer {
    background: #1a1408; border-left: 3px solid #d4922a;
    padding: 14px 18px; border-radius: 0 10px 10px 0;
    color: #c49a45; font-size: 13px; line-height: 1.6; margin: 18px 0 8px 0;
}
[data-testid="stSidebar"] { background: #0d1117 !important; border-right: 1px solid #1e2640 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:10px 0 4px 0">
    <div style="font-size:42px">🦴</div>
    <div style="font-size:28px; font-weight:700; color:#f0f4ff; letter-spacing:-0.5px">Spinal Curvature Screener</div>
    <div style="font-size:14px; color:#6b7a99; margin-top:4px">Postural Cobb angle estimation · AI-assisted screening</div>
</div>
<div class="disclaimer">
⚠️ <strong>Medical Disclaimer:</strong> This tool estimates <em>postural curvature</em> from surface body landmarks,
not X-ray imaging. Results are a <strong>screening approximation only</strong> — not a clinical Cobb angle.
Always consult a physician for medical evaluation.
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ─── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    confidence = st.slider("Detection Confidence", 0.1, 1.0, 0.5, 0.05,
        help="Minimum pose landmark confidence. Lower = more permissive.")
    st.markdown("---")
    st.markdown("### 📋 Report Options")
    patient_name = st.text_input("Subject Name (optional)", placeholder="e.g. Patient A")
    clinician_notes = st.text_area("Clinician Notes (optional)",
        placeholder="Add any relevant observations...", height=100)
    st.markdown("---")
    st.markdown("### 📷 Photo Tips")
    st.markdown("""
- Subject facing **directly away** from camera
- Full torso visible — shoulders **and** hips in frame
- Even lighting, minimal shadows
- Form-fitting clothing preferred
- Camera at **mid-back height**
    """)
    st.markdown("---")
    st.markdown("### 📊 Severity Scale")
    for label, color, rng in [
        ("Normal",             "#3cc86e", "< 10°"),
        ("Mild Scoliosis",     "#f5a623", "10° – 25°"),
        ("Moderate Scoliosis", "#e07820", "25° – 40°"),
        ("Severe Scoliosis",   "#d0312d", "> 40°"),
    ]:
        st.markdown(
            f'<span style="color:{color}; font-size:13px">● <b>{label}</b></span>'
            f'<span style="color:#888; font-size:12px"> {rng}</span>',
            unsafe_allow_html=True)

# ─── Main layout ──────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

raw_img = None
with col_left:
    st.markdown('<div class="section-header">Upload Photo</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Back photo", type=["jpg", "jpeg", "png", "webp"],
                                label_visibility="collapsed")
    if uploaded:
        img_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        raw_img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB),
                 caption="Uploaded photo", use_column_width=True)
        analyze_btn = st.button("🔍 Analyze Spinal Curvature", type="primary", use_container_width=True)
    else:
        st.markdown("""
        <div style="border:2px dashed #2a3868; border-radius:12px; padding:50px 20px;
                    text-align:center; color:#4a5580; margin-top:10px">
            <div style="font-size:36px">📸</div>
            <div style="font-size:15px; margin-top:8px">Upload a back photo to begin</div>
            <div style="font-size:12px; margin-top:4px">JPG, PNG, or WebP</div>
        </div>""", unsafe_allow_html=True)
        analyze_btn = False

# ─── Analysis ─────────────────────────────────────────────────────
if uploaded and analyze_btn and raw_img is not None:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, raw_img)
        tmp_path = tmp.name

    with st.spinner("Detecting landmarks and computing curvature..."):
        try:
            with CobbAngleEstimator(min_detection_confidence=confidence) as estimator:
                result = estimator.analyze(tmp_path)
        except Exception as e:
            st.error(f"Analysis error: {e}")
            result = None
        finally:
            os.unlink(tmp_path)

    if result is None:
        with col_right:
            st.error(
                "**Could not detect body landmarks.**\n\n"
                "Try:\n"
                "- Ensure the full torso (shoulders + hips) is visible\n"
                "- Use a well-lit photo\n"
                "- Lower the confidence threshold in the sidebar\n"
                "- Confirm subject is facing **away** from the camera"
            )
    else:
        # ── Annotated image ───────────────────────────────────────
        with col_right:
            st.markdown('<div class="section-header">Analysis Result</div>', unsafe_allow_html=True)
            annotated_rgb = cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb,
                     caption="Yellow = spine midline · Blue = shoulders · Purple = hips",
                     use_column_width=True)

        # ── Metric cards ──────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-header">Measurements</div>', unsafe_allow_html=True)

        severity_hex = {
            "Normal":             "#3cc86e",
            "Mild Scoliosis":     "#f5a623",
            "Moderate Scoliosis": "#e07820",
            "Severe Scoliosis":   "#d0312d",
        }.get(result.severity, "#aaaaaa")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Cobb Angle</div>
                <div class="metric-value">{result.cobb_angle:.1f}<span class="metric-unit">°</span></div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Classification</div>
                <div style="margin-top:8px">
                    <span class="severity-badge"
                          style="background:{severity_hex}22; color:{severity_hex}; border:1.5px solid {severity_hex}">
                        {result.severity}
                    </span>
                </div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Shoulder Tilt</div>
                <div class="metric-value">{abs(result.shoulder_tilt_deg):.1f}<span class="metric-unit">°</span></div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Hip Tilt</div>
                <div class="metric-value">{abs(result.hip_tilt_deg):.1f}<span class="metric-unit">°</span></div>
            </div>""", unsafe_allow_html=True)

        # Confidence bar
        st.markdown(f"""
        <div style="margin-top:10px; font-size:12px; color:#6b7a99">
            Detection Confidence: <b style="color:#c0cce8">{result.confidence*100:.0f}%</b>
        </div>""", unsafe_allow_html=True)
        st.progress(result.confidence)

        # Clinical advisory
        if result.cobb_angle >= 25:
            st.error(f"🚨 **{result.severity} ({result.cobb_angle:.1f}°)** — Physician referral recommended. X-ray required for definitive evaluation.")
        elif result.cobb_angle >= 10:
            st.warning(f"⚠️ **{result.severity} ({result.cobb_angle:.1f}°)** — Consider consulting a physician for further assessment.")
        else:
            st.success(f"✅ **{result.severity} ({result.cobb_angle:.1f}°)** — No significant postural curvature detected.")

        # ── Exports ───────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-header">Export Results</div>', unsafe_allow_html=True)
        dl1, dl2 = st.columns(2)

        with dl1:
            ok, img_buf = cv2.imencode(".jpg", result.annotated_image)
            if ok:
                st.download_button(
                    label="⬇️ Download Annotated Image",
                    data=img_buf.tobytes(),
                    file_name="spinal_analysis_annotated.jpg",
                    mime="image/jpeg",
                    use_container_width=True,
                )

        with dl2:
            with st.spinner("Generating PDF..."):
                try:
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as rf:
                        cv2.imwrite(rf.name, raw_img)
                        orig_tmp = rf.name
                    pdf_bytes = generate_pdf_report(
                        result=result,
                        image_path=orig_tmp,
                        patient_name=patient_name or "Anonymous",
                        notes=clinician_notes,
                    )
                    os.unlink(orig_tmp)
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_bytes,
                        file_name="spinal_curvature_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")

        # ── Interpretation guide ──────────────────────────────────
        with st.expander("📖 Interpretation Guide & Method Details"):
            st.markdown("""
| Angle | Classification | Recommended Action |
|-------|---------------|-------------------|
| < 10° | Normal | Routine monitoring |
| 10° – 25° | Mild Scoliosis | Consider physician referral |
| 25° – 40° | Moderate Scoliosis | Physician referral; X-ray indicated |
| > 40° | Severe Scoliosis | Urgent physician evaluation |

**Method:** MediaPipe Pose detects shoulder, hip, ear, and knee landmarks.
Midpoints form a spine centerline approximation. A 2nd-degree polynomial is
fit through these points, and the angle between tangent vectors at the top and
bottom of the curve is computed — analogous to the Cobb method.
Surface landmarks systematically **underestimate** true vertebral Cobb angles.
            """)