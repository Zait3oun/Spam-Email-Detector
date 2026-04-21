import streamlit as st
import joblib
import os

# =====================
# Page Config
# =====================
st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="centered"
)

# =====================
# Paths
# =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "spam_model.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

# =====================
# Load model (safe)
# =====================
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        tfidf = joblib.load(TFIDF_PATH)
        return model, tfidf
    except Exception as e:
        return None, str(e)

model, tfidf = load_model()

if model is None:
    st.error("❌ Error loading model files")
    st.stop()

# =====================
# UI Styling
# =====================
st.markdown("""
<style>
.title {
    text-align: center;
    font-size: 44px;
    font-weight: bold;
    color: #38bdf8;
}

.subtitle {
    text-align: center;
    font-size: 16px;
    color: #94a3b8;
    margin-bottom: 25px;
}

.box {
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# =====================
# Header
# =====================
st.markdown('<div class="title">📧 Spam Email Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered classification system</div>', unsafe_allow_html=True)

# =====================
# Input
# =====================
text = st.text_area("✍️ Enter email content:")

# =====================
# Prediction
# =====================
if st.button("🔍 Analyze Email"):
    if not text.strip():
        st.warning("⚠️ Please enter some text first")
    else:
        try:
            # تحويل النص
            vector = tfidf.transform([text])

            # prediction
            prediction = model.predict(vector)[0]

            # result
            if prediction == 1:
                st.error("🚨 SPAM EMAIL DETECTED!")
                st.markdown("⚠️ Be careful! This email looks suspicious.")
            else:
                st.success("✅ NOT SPAM")
                st.markdown("✔️ This email seems safe.")

        except Exception as e:
            st.error("❌ Prediction failed")
            st.code(str(e))

# =====================
# Footer
# =====================
st.markdown("---")
st.caption("💡 Built with Streamlit | AI Spam Classifier")