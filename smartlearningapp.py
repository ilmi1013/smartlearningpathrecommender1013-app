import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------------
# 1️⃣ App Title
# -------------------------------
st.title("📘 Smart Learning Path Recommender")
st.write("Predict the **best learning mode** for O-Level Mathematics students based on their background and performance.")

# -------------------------------
# 2️⃣ File uploader
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload your dataset (.xlsx)", type=["xlsx"])

if uploaded_file is not None:

    @st.cache_data
    def load_data(file):
        return pd.read_excel(file)

    df = load_data(uploaded_file)
    df.columns = df.columns.str.strip()

    # -------------------------------
    # 3️⃣ Define features & target
    # -------------------------------
    target_col = "learning mode prefer to learn for O-Level Mathematics"

    categorical_features = [
        "Gender",
        "Current learning mode in school for mathemathics",
        "learning mode experienced for O-Level Mathematics",
        "19. How confident do you feel in your ability to score well in Mathematics through blended learning?",
        "10. What is the overall level of engagement you feel during physical classes?",
        "24. What challenges do you face in blended learning? (Select all that apply.)    [I find it hard to stay focused or motivated.]",
        "12.Indicate your level of agreement with the following statements regarding physical learning.? (Select all that apply.)   [It’s hard to focus with all the distractions in class.]"
    ]

    numeric_features = [
        "Mathematics marks range  in  Term Test 2024/ 2025 [1st Term]",
        "Mathematics marks range  in  Term Test 2024/ 2025 [2nd Term]",
        "Mathematics marks range  in  Term Test 2024/ 2025 [3rd Term]"
    ]

    features = categorical_features + numeric_features

    # -------------------------------
    # 4️⃣ Helper function for numeric ranges
    # -------------------------------
    def range_to_midpoint(val):
        try:
            if pd.isna(val):
                return None
            val_str = str(val).replace('%','')
            if '-' in val_str:
                low, high = val_str.split('-')
                return (float(low.strip()) + float(high.strip())) / 2
            else:
                return float(val_str.strip())
        except:
            return None

    for col in numeric_features:
        df[col] = df[col].apply(range_to_midpoint)
        df[col] = df[col].fillna(df[col].median())

    # Drop rows with missing target or features
    df_model = df[features + [target_col]].dropna().copy()

    # -------------------------------
    # 5️⃣ Encode categorical features
    # -------------------------------
    label_encoders = {}
    for col in categorical_features + [target_col]:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        label_encoders[col] = le

    # -------------------------------
    # 6️⃣ Train-test split and model
    # -------------------------------
    X = df_model[features]
    y = df_model[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.subheader("📊 Model Performance")
    st.write(f"**Accuracy:** {accuracy:.2f}")

    # -------------------------------
    # 7️⃣ User Input Form
    # -------------------------------
    st.subheader("📝 Enter Student Details")
    user_input = {}

    for col in categorical_features:
        options = list(df[col].dropna().unique())
        user_input[col] = st.selectbox(f"{col}", options)

    for col in numeric_features:
        user_input[col] = st.text_input(f"{col} (e.g., 55-60)", "")

    # -------------------------------
    # 8️⃣ Predict Button
    # -------------------------------
    if st.button("🔮 Predict Learning Mode"):
        input_df = pd.DataFrame([user_input])

        # Encode categorical
        for col in categorical_features:
            le = label_encoders[col]
            val = str(input_df.at[0, col])
            if val in le.classes_:
                input_df[col] = le.transform([val])
            else:
                input_df[col] = -1

        # Convert numeric
        for col in numeric_features:
            input_df[col] = range_to_midpoint(input_df.at[0, col])

        # Predict
        pred_encoded = model.predict(input_df)[0]
        pred_label = label_encoders[target_col].inverse_transform([pred_encoded])[0]

        st.success(f"✅ Recommended Learning Mode: **{pred_label}**")

else:
    st.warning("⚠️ Please upload your Excel dataset to continue.")
