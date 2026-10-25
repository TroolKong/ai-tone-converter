import streamlit as st
from google import genai

# Set up the web page title and icon
st.set_page_config(page_title="AI Tone Converter", page_icon="⚡")

st.title("⚡ AI Email Tone Converter")
st.write("Transform any draft email into your desired tone instantly using Gemini.")

# Sidebar for API Key input
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter Gemini API Key:", type="password")

# User inputs for the app
user_email = st.text_area(
    "Paste your draft email here:",
    height=150,
    placeholder="e.g. Hey, where is the report? I needed it yesterday."
)

selected_tone = st.selectbox(
    "Select desired tone:",
    ["Professional & Polite", "Casual & Friendly", "Direct & Concise", "Persuasive"]
)

# Action button
if st.button("Rewrite Email"):
    if not api_key:
        st.error("Please enter your Gemini API Key in the sidebar!")
    elif not user_email.strip():
        st.warning("Please paste a draft email first.")
    else:
        try:
            # Initialize client with user provided key
            client = genai.Client(api_key=api_key)

            prompt = f"Rewrite the following email to be {selected_tone}:\n\n{user_email}"

            with st.spinner("Rewriting email with Gemini..."):
                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=prompt,
                )

            st.success("Done!")
            st.subheader("Generated Result:")
            st.write(response.text)

        except Exception as e:
            st.error(f"An error occurred: {e}")