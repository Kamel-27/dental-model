import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page configuration for a premium look
st.set_page_config(
    page_title="AI Periodontal Diagnosis System",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Glassmorphism, beautiful gradients, hover effects, modern typography)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main Background & Container styling */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #020617 100%);
        color: #f8fafc;
    }
    
    /* Header Card styling */
    .header-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 30px;
        backdrop-filter: blur(16px);
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        text-align: center;
        background-image: radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.15) 0%, transparent 40%),
                          radial-gradient(circle at 90% 80%, rgba(236, 72, 153, 0.15) 0%, transparent 40%);
    }
    
    .header-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(to right, #818cf8, #e879f9, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 400;
    }
    
    /* Premium glassmorphic cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(12px);
        margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.2);
    }
    
    /* Custom Diagnosis Badges */
    .badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-gingivitis {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-stage1 {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .badge-stage2 {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .badge-stage3 {
        background-color: rgba(220, 38, 38, 0.2);
        color: #ef4444;
        border: 1px solid rgba(220, 38, 38, 0.4);
    }
    .badge-stage4 {
        background-color: rgba(139, 92, 246, 0.2);
        color: #a78bfa;
        border: 1px solid rgba(139, 92, 246, 0.4);
    }
    
    /* Modern Section Headers */
    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #c7d2fe;
        margin-bottom: 15px;
        border-left: 4px solid #6366f1;
        padding-left: 10px;
    }
    
    /* Custom probability progress bar wrappers */
    .prob-bar-wrapper {
        margin-bottom: 12px;
    }
    .prob-bar-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.9rem;
        color: #cbd5e1;
        margin-bottom: 4px;
    }
    .prob-bar-bg {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        height: 10px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.8s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)

# App Path setup and model loading
workspace_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(workspace_dir, "best_lightgbm_model.joblib")

@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    else:
        return None

model = load_model(model_path)

# UI Layout
st.markdown("""
<div class="header-card">
    <div class="header-title">🦷 AI Periodontal Diagnostics System</div>
    <div class="header-subtitle">State-of-the-Art Periodontal Staging & Classification powered by LightGBM & 2018 Guidelines</div>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error(f"⚠️ Model file `best_lightgbm_model.joblib` was not found in: `{workspace_dir}`. Please make sure the model is trained and saved in this directory.")
else:
    # Set up columns for layout
    col_input, col_output = st.columns([1.1, 0.9])
    
    with col_input:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📝 Diagnostics Input Form</div>', unsafe_allow_html=True)
        
        # Tabs for Patient Questionnaire vs Clinical metrics
        tab_patient, tab_clinical = st.tabs([
            "📋 Patient Questionnaire (Oral Health PDF)", 
            "🔬 Clinical Metrics & OHIP-14 (Advanced)"
        ])
        
        # ------------------ TAB 1: Patient Questionnaire (From PDF) ------------------
        with tab_patient:
            st.info("💡 Fill in the standard questionnaire details as provided in the Oral Health Questionnaire for Adults PDF.")
            
            # Question 4: Age
            age = st.slider("Q4. How old are you today? (Years)", min_value=18, max_value=95, value=40)
            
            # Question 1: Sex (gender)
            gender_select = st.selectbox("Q1a. Sex", ["Female", "Male"])
            gender_encoded = 0.0 if gender_select == "Female" else 1.0
            
            # Question 1: Location
            location_select = st.selectbox("Q1b. Location / Area of Residence", ["Urban", "Rural"])
            location_encoded = 1.0 if location_select == "Urban" else 0.0
            
            # Question 2: Medical History
            med_select = st.selectbox(
                "Q2. Medical History (Systemic Diseases)", 
                [
                    "Free (No significant medical history)", 
                    "Hypertension", 
                    "Diabetes", 
                    "Diabetes + Hypertension", 
                    "Hypotension", 
                    "Cancer"
                ]
            )
            med_mapping = {
                "Free (No significant medical history)": "free",
                "Hypertension": "hypertension",
                "Diabetes": "diabetes",
                "Diabetes + Hypertension": "diabetes + hypertension",
                "Hypotension": "hypotension",
                "Cancer": "cancer"
            }
            med_val = med_mapping[med_select]
            
            # Question 3: History of trauma
            trauma_select = st.selectbox("Q3. History of trauma to the teeth or face?", ["No", "Yes"])
            trauma_encoded = 1.0 if trauma_select == "Yes" else 0.0
            
            # Question 15: Education completed
            edu_select = st.selectbox(
                "Q15. What level of education have you completed?",
                [
                    "No formal schooling (Level 1)",
                    "Less than primary school (Level 2)",
                    "Primary school completed (Level 3)",
                    "Secondary school completed (Level 4)",
                    "High school completed (Level 5)",
                    "College/university completed (Level 6)",
                    "Postgraduate degree (Level 7)"
                ],
                index=4 # Defaulting to High School completed
            )
            edu_mapping = {
                "No formal schooling (Level 1)": 1.0,
                "Less than primary school (Level 2)": 2.0,
                "Primary school completed (Level 3)": 3.0,
                "Secondary school completed (Level 4)": 4.0,
                "High school completed (Level 5)": 5.0,
                "College/university completed (Level 6)": 6.0,
                "Postgraduate degree (Level 7)": 7.0
            }
            education = edu_mapping[edu_select]
            
        # ------------------ TAB 2: Clinical Metrics & OHIP-14 ------------------
        with tab_clinical:
            st.warning("⚠️ These measurements are taken during a professional dental checkup. Median values from the dataset are provided by default.")
            
            # Clinical metrics sliders/inputs pre-filled with medians
            pocket_depth = st.slider("Mean Pocket Depth (mm)", min_value=1.0, max_value=5.0, value=2.33, step=0.01)
            cal = st.slider("Clinical Attachment Loss (CAL) (mm)", min_value=0.0, max_value=6.0, value=2.00, step=0.1)
            recession = st.slider("Gingival Recession (mm)", min_value=0.0, max_value=4.0, value=1.00, step=0.1)
            bop = st.slider("Bleeding on Probing (BOP Ratio)", min_value=0.0, max_value=1.0, value=1.00, step=0.01)
            plaque_index = st.slider("Plaque Index", min_value=0.0, max_value=3.0, value=2.00, step=0.01)
            
            # OHIP-14 Questions in an expander
            with st.expander("📝 Oral Health Impact Profile (OHIP-14 Questionnaire)"):
                st.write("Indicate how often you have experienced the following problems in the past 12 months:")
                ohip_options = {
                    "Never (0)": 0.0,
                    "Hardly Ever (1)": 1.0,
                    "Occasionally (2)": 2.0,
                    "Fairly Often (3)": 3.0,
                    "Very Often (4)": 4.0
                }
                
                # Defaulting to "Occasionally" (2.0) or "Never" (0.0) based on typical medians
                q1 = ohip_options[st.selectbox("1. Trouble pronouncing words because of teeth/mouth?", list(ohip_options.keys()), index=0)]
                q2 = ohip_options[st.selectbox("2. Felt your sense of taste has worsened?", list(ohip_options.keys()), index=0)]
                q3 = ohip_options[st.selectbox("3. Painful aching in your mouth?", list(ohip_options.keys()), index=2)]
                q4 = ohip_options[st.selectbox("4. Uncomfortable to eat any foods because of teeth/mouth?", list(ohip_options.keys()), index=2)]
                q5 = ohip_options[st.selectbox("5. Been self-conscious because of teeth/mouth?", list(ohip_options.keys()), index=2)]
                q6 = ohip_options[st.selectbox("6. Felt tense because of teeth/mouth?", list(ohip_options.keys()), index=0)]
                q7 = ohip_options[st.selectbox("7. Felt your diet has been unsatisfactory?", list(ohip_options.keys()), index=0)]
                q8 = ohip_options[st.selectbox("8. Had to interrupt meals because of teeth/mouth?", list(ohip_options.keys()), index=0)]
                q9 = ohip_options[st.selectbox("9. Found it difficult to relax because of teeth/mouth?", list(ohip_options.keys()), index=0)]
                q10 = ohip_options[st.selectbox("10. Been a bit embarrassed because of teeth/mouth?", list(ohip_options.keys()), index=0)]
                q11 = ohip_options[st.selectbox("11. Been a bit irritable with other people?", list(ohip_options.keys()), index=0)]
                q12 = ohip_options[st.selectbox("12. Had difficulty doing your usual jobs?", list(ohip_options.keys()), index=0)]
                q13 = ohip_options[st.selectbox("13. Felt that life in general was less satisfying?", list(ohip_options.keys()), index=0)]
                q14 = ohip_options[st.selectbox("14. Totally unable to function because of teeth/mouth?", list(ohip_options.keys()), index=0)]
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_output:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔮 Real-Time Diagnostic Prediction</div>', unsafe_allow_html=True)
        
        # 1. Build prediction input DataFrame matching the EXACT feature names of the LightGBM model
        input_data = {
            'age': float(age),
            'education': float(education),
            'Pocket_depht': float(pocket_depth),
            'Clinical_attatchment_loss': float(cal),
            'Recession': float(recession),
            'Bleeding_on_Probing': float(bop),
            'Plaque_index': float(plaque_index),
            'Q_1': float(q1),
            'Q_2': float(q2),
            'Q3': float(q3),
            'Q4': float(q4),
            'Q5': float(q5),
            'Q6': float(q6),
            'Q7': float(q7),
            'Q8': float(q8),
            'Q9_': float(q9),
            'Q10': float(q10),
            'Q11': float(q11),
            'Q12': float(q12),
            'Q13': float(q13),
            'Q14': float(q14),
            'gender_encoded': float(gender_encoded),
            'location_encoded': float(location_encoded),
            'trauma_encoded': float(trauma_encoded),
            'med_cancer': 1.0 if med_val == 'cancer' else 0.0,
            'med_diabetes': 1.0 if med_val == 'diabetes' else 0.0,
            'med_diabetes_+_hypertension': 1.0 if med_val == 'diabetes + hypertension' else 0.0,
            'med_free': 1.0 if med_val == 'free' else 0.0,
            'med_hypertension': 1.0 if med_val == 'hypertension' else 0.0,
            'med_hypotension': 1.0 if med_val == 'hypotension' else 0.0
        }
        
        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        
        # 2. Run LightGBM Prediction
        try:
            pred_class = model.predict(input_df)[0]
            pred_proba = model.predict_proba(input_df)[0]
            
            # Map predictions to labels and styling
            classes_meta = {
                0: {"name": "Gingivitis", "badge": "badge-gingivitis", "desc": "Inflammation of the gums, typically reversible with proper hygiene.", "color": "#34d399", "bg": "rgba(16, 185, 129, 0.2)"},
                1: {"name": "Stage I Periodontitis", "badge": "badge-stage1", "desc": "Initial periodontitis with early bone loss and pocket depths ≤4mm.", "color": "#fbbf24", "bg": "rgba(245, 158, 11, 0.2)"},
                2: {"name": "Stage II Periodontitis", "badge": "badge-stage2", "desc": "Established periodontitis with moderate bone loss and pocket depths ≤5mm.", "color": "#f87171", "bg": "rgba(239, 68, 68, 0.2)"},
                3: {"name": "Stage III Periodontitis", "badge": "badge-stage3", "desc": "Severe periodontitis with significant bone loss, deep pocket depths ≥6mm, and risk of tooth loss.", "color": "#ef4444", "bg": "rgba(220, 38, 38, 0.25)"},
                4: {"name": "Stage IV Periodontitis", "badge": "badge-stage4", "desc": "Advanced periodontitis with extensive bone loss, joint mobility, severe dysfunction, and high risk of losing all teeth.", "color": "#a78bfa", "bg": "rgba(139, 92, 246, 0.25)"}
            }
            
            curr_meta = classes_meta[pred_class]
            
            st.markdown(f"""
            <div style="background: {curr_meta['bg']}; border: 2px solid {curr_meta['color']}; border-radius: 12px; padding: 20px; text-align: center; margin-bottom: 25px;">
                <span class="badge {curr_meta['badge']}">PREDICTED STATE</span>
                <h2 style="color: {curr_meta['color']}; font-weight: 800; font-size: 2.2rem; margin-top: 10px; margin-bottom: 8px;">{curr_meta['name']}</h2>
                <p style="color: #cbd5e1; font-size: 1rem; max-width: 90%; margin: 0 auto; line-height: 1.4;">{curr_meta['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display Diagnostic Probability Bars
            st.markdown('<div style="font-weight: 600; color: #e2e8f0; font-size: 1.05rem; margin-bottom: 15px;">📊 Probability Distribution</div>', unsafe_allow_html=True)
            
            for class_idx, meta in classes_meta.items():
                prob_pct = pred_proba[class_idx] * 100
                st.markdown(f"""
                <div class="prob-bar-wrapper">
                    <div class="prob-bar-labels">
                        <span>{meta['name']}</span>
                        <span style="font-weight: 600; color: {meta['color']}">{prob_pct:.2f}%</span>
                    </div>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width: {prob_pct}%; background-color: {meta['color']}; box-shadow: 0 0 10px {meta['color']}80;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            # Clinical Recommendations
            st.markdown('<div style="margin-top: 30px;" class="section-title">🩺 Recommended Action Plan</div>', unsafe_allow_html=True)
            if pred_class == 0:
                st.success("✅ **Recommendation:** Routine professional dental scaling, twice-daily toothbrushing with fluoride toothpaste, and daily dental flossing. Review in 6 months.")
            elif pred_class == 1:
                st.info("⚠️ **Recommendation:** Professional scaling and root planing (SRP), periodontal charting, oral hygiene instructions, and a 3-month follow-up evaluation.")
            elif pred_class == 2:
                st.warning("🚨 **Recommendation:** Active periodontal therapy (debridement, scaling, root planing), possible antimicrobial therapy, rigorous home care evaluation, and 3-month maintenance.")
            else:
                st.error("🛑 **Recommendation:** Urgent Periodontal Specialist referral! Requires deep debridement, surgical evaluation (flap surgery/bone graft), stabilization of hypermobile teeth, and custom rehabilitation.")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.info("Please make sure that the input fields have valid values.")
            
        st.markdown('</div>', unsafe_allow_html=True)
