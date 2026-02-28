import streamlit as st
from transformers import pipeline

# Load model with caching to prevent reloading on every click
@st.cache_resource
def load_model():
    # We specify top_k=None to ensure we get a LIST of all emotions
    return pipeline("text-classification", 
                    model="j-hartmann/emotion-english-distilroberta-base", 
                    top_k=None)

def detect_gaslighting(text):
    classifier = load_model()
    # classifier(text) returns a list of lists: [[{'label':..., 'score':...}, ...]]
    all_emotions = classifier(text)[0] 
    
    # Simple keyword check for tactics
    tactics_map = {
        "Denial": ["never said", "didn't happen", "imagining"],
        "Blame-Shifting": ["your fault", "if you hadn't", "you made me"],
        "Invalidation": ["overreacting", "too sensitive", "crazy", "delusional"]
    }
    
    found_tactics = [t for t, keywords in tactics_map.items() if any(k in text.lower() for k in keywords)]
    return all_emotions, found_tactics

# --- Streamlit UI ---
st.set_page_config(page_title="Gaslighting Detector", page_icon="🛡️")

st.title("🛡️ NLP Gaslighting Detector")
st.markdown(f"**Project Lead:** A. Eshwar Chary")
st.divider()

user_input = st.text_area("Enter conversational text to analyze:", 
                          placeholder="e.g., 'I never said that, you're just being too sensitive.'",
                          height=150)

if st.button("Analyze Pattern", type="primary"):
    if user_input.strip():
        with st.spinner("Analyzing linguistic patterns..."):
            try:
                emotions, tactics = detect_gaslighting(user_input)
                
                # Layout for results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Emotion Profile")
                    # Safely find the highest score
                    top_emotion = max(emotions, key=lambda x: x['score'])
                    
                    label = top_emotion['label'].upper()
                    score = round(top_emotion['score'] * 100, 1)
                    
                    if label in ["ANGER", "DISGUST"]:
                        st.error(f"Dominant: {label} ({score}%)")
                    else:
                        st.info(f"Dominant: {label} ({score}%)")

                with col2:
                    st.subheader("🔍 Tactics Detected")
                    if tactics:
                        for t in tactics:
                            st.warning(f"⚠️ {t}")
                    else:
                        st.success("No common gaslighting keywords found.")

                # --- Final Verdict ---
                st.divider()
                if len(tactics) > 0 and top_emotion['label'] in ['anger', 'disgust', 'neutral']:
                    st.error("### Final Verdict: HIGH RISK")
                    st.write("The system detected a combination of aggressive emotional tone and manipulative linguistic patterns (Denial/Invalidation).")
                else:
                    st.success("### Final Verdict: LOW RISK")
                    st.write("No strong indicators of systematic gaslighting were detected in this snippet.")

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
    else:
        st.warning("Please enter some text to begin analysis.")

st.sidebar.info("This tool uses Natural Language Processing to identify psychological manipulation patterns in text.")