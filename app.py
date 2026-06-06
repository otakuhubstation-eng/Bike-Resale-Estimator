import pickle
import pandas as pd
import streamlit as st

st.markdown(
    """
    <style>
    .stApp {
        background-image: linear-gradient(135deg, red, yellow);
    }
    
    [class="stVerticalBlock st-emotion-cache-1gz5zxc e1rw0b1u3"] {    
        background-color: rgba(255, 255, 255, 0.3) ;
        backdrop-filter: blur(8px) ;
        border-radius: 12px ;
        padding: 25px ;
        border: 20px ;
        margin: 5px ;
    }
    
    [class="st-emotion-cache-4cktc5 en7m6i60]{
    background-color:#f0f2f6; padding:20px; border-radius:10px; text-align:center;}
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_data
def load_data():
    df = pd.read_csv('Used_Bikes.csv')
    return df

@st.cache_resource
def load_models():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, encoder, scaler

df = load_data()
model, encoder, scaler = load_models()

with st.container(border=True):
    st.markdown(f"""
                    <div style="background-color: rgba(255,255,255,0.2);
                        padding:20px; 
                        border-radius:10px; 
                        text-align:center;">
                        <h1 style="color:#2B2926;">Bike Resale Estimator</h1>
                    </div>
                """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        # Brand dropdown
        brand = st.selectbox("Select Brand", sorted(df["brand"].unique()))

    with col2:
        # power dropdown — depends on brand
        df_filtered_power = df[df["brand"] == brand]
        filtered_power = df[df["brand"] == brand]["power"].unique()
        power = st.selectbox("Select power", sorted(filtered_power))

    with col3:
        # Model dropdown — depends on power
        filtered_models = df_filtered_power[df_filtered_power["power"] == power]["bike_name"].unique()
        bike_name = st.selectbox("Select Model", sorted(filtered_models))

    col4, col5, col6 = st.columns(3)

    with col4:
        owner = st.selectbox("Owner", sorted(df["owner"].unique()))
    with col5:
        age = st.selectbox("age", sorted(df["age"].unique()))
    with col6:
        city = st.selectbox("city", sorted(df["city"].unique()))

    kms_driven = st.number_input("Kms Driven", min_value=1, max_value=1000000, value=500, step=1)

    if st.button("Predict"):
        # Step 1 — make a dataframe with the same column names as training
        input_df = pd.DataFrame([[bike_name, city, owner, power, kms_driven, age]],
                                columns=['bike_name', 'city', 'owner', 'power', 'kms_driven', 'age'])

        # Step 2 — encode the same categorical columns
        cat_encoded = encoder.transform(input_df[['bike_name', 'city', 'owner', 'power']])
        cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out())

        # Step 3 — join with numerical columns (same order as training)
        final_input = input_df[['kms_driven', 'age']].join(cat_encoded_df)

        # Step 4 — scale
        final_scaled = scaler.transform(final_input)

        # Step 5 — predict
        prediction = model.predict(final_scaled)
        st.markdown(f"""
                <div style="background-color: rgba(255,255,255,0.3);
                    backdrop-filter: blur(10px);
                    padding:20px; 
                    border-radius:10px; 
                    text-align:center;">
                    <h3 style="color:#333;">Estimated Bike Price</h3>
                    <h1 style="color:#2ecc71;">₹ {prediction[0]:,.0f}</h1>
                </div>
            """, unsafe_allow_html=True)