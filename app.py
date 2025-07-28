import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

model = joblib.load('portfolio_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# header
st.set_page_config(page_title="Investment Recommender")
st.title("Investment Portfolio Recommender")
st.markdown("Welcome! Answer a few quick questions to get your ideal investment mix.")

st.markdown("---")  

# sidebar
st.sidebar.header("Enter Your Details")

age = st.sidebar.slider("Age", 18, 70, 30)
income = st.sidebar.number_input("Annual Income (₹)", min_value=100000, max_value=3000000, value=500000, step=50000)

knowledge = st.sidebar.selectbox("Investment Knowledge", ['Low', 'Medium', 'High'])
risk = st.sidebar.selectbox("Risk Appetite", ['Low', 'Medium', 'High'])
goal = st.sidebar.selectbox("Investment Goal", ['Retirement', 'Wealth Growth', 'Short-Term Gain'])

def encode_input(value, col_name):
    le = label_encoders[col_name]
    if value not in le.classes_:
        raise ValueError(f"Label '{value}' is not recognized for column '{col_name}'.")
    return le.transform([value])[0]

encoded_knowledge = encode_input(knowledge, 'Knowledge')
encoded_risk = encode_input(risk, 'Risk')
encoded_goal = encode_input(goal, 'Goal')

input_data = np.array([[age, income, encoded_knowledge, encoded_risk, encoded_goal]])

if st.sidebar.button("Recommend Portfolio"):
    prediction = model.predict(input_data)[0]
    portfolio = label_encoders['Portfolio'].inverse_transform([prediction])[0]  

    st.subheader("Recommended Portfolio")
    st.success(f"**{portfolio}**")

    if portfolio == "Aggressive":
        labels = ['Stocks', 'Mutual Funds', 'Bonds']
        sizes = [70, 20, 10]
    elif portfolio == "Moderate":
        labels = ['Stocks', 'Mutual Funds', 'Bonds']
        sizes = [40, 40, 20]
    else:
        labels = ['Stocks', 'Mutual Funds', 'Bonds']
        sizes = [10, 20, 70]

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.0f%%', startangle=90)
    ax.axis('equal')

    st.subheader("Suggested Allocation")
    st.pyplot(fig)

else:
    st.info("Fill in the details in the sidebar and click **Recommend Portfolio** to get started.")
