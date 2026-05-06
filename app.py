import streamlit as st
import pandas as pd
import pickle
import re

# ===== Load Models =====
vectorization = pickle.load(open("vectorizer.pkl", "rb"))
LR = pickle.load(open("LR.pkl", "rb"))
DT = pickle.load(open("DT.pkl", "rb"))
GB = pickle.load(open("GB.pkl", "rb"))
RF = pickle.load(open("RF.pkl", "rb"))

# ===== Text Cleaning Function =====
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape("""!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# ===== Output Label =====
def output_label(n):
    if n == 0:
        return "❌ Fake News"
    else:
        return "✅ Not Fake News"

# ===== Prediction Function =====
def predict_news(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)

    new_def_test["text"] = new_def_test["text"].apply(wordopt)

    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)

    pred_LR = LR.predict(new_xv_test)[0]
    pred_DT = DT.predict(new_xv_test)[0]
    pred_GBC = GB.predict(new_xv_test)[0]
    pred_RFC = RF.predict(new_xv_test)[0]

    return {
        "LR": output_label(pred_LR),
        "DT": output_label(pred_DT),
        "GBC": output_label(pred_GBC),
        "RFC": output_label(pred_RFC)
    }

# ===== Streamlit UI =====
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detector")
st.write("Analyze whether a news article is real or fake using Machine Learning models.")

news_input = st.text_area("✍️ Enter News Text Here")

# ===== Button Action =====
if st.button("🔍 Analyze News"):

    if news_input.strip() == "":
        st.warning("⚠️ Please enter some news text.")
    else:
        with st.spinner("Analyzing..."):
            results = predict_news(news_input)

        st.subheader("📊 Model Predictions")

        st.write(f"**Logistic Regression:** {results['LR']}")
        st.write(f"**Decision Tree:** {results['DT']}")
        st.write(f"**Gradient Boosting:** {results['GBC']}")
        st.write(f"**Random Forest:** {results['RFC']}")

        # ===== Majority Voting =====
        votes = list(results.values())
        real_count = votes.count("✅ Not Fake News")
        fake_count = votes.count("❌ Fake News")

        st.subheader("🧠 Final Verdict")

        if real_count > fake_count:
            st.success(f"✅ This news is likely REAL ({real_count}/4 models agree)")
        else:
            st.error(f"❌ This news is likely FAKE ({fake_count}/4 models agree)")

        # ===== Confidence Bar =====
        st.progress(real_count / 4)