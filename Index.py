import streamlit as st

def get_file_content(file_path)-> str: 
    with open(file_path, 'r') as f:
        markdown_string = f.read()
    return markdown_string


st.set_page_config(
    page_title="Index page",
    page_icon="👋",
)

st.write("# Welcome to Index page! 👋")

st.sidebar.success("Select a Cv application above.")

readme_text = st.markdown(get_file_content('./TEST.md'))
