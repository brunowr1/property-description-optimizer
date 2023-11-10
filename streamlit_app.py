import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import time
import json
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI
from langchain.prompts import PromptTemplate

"""
# Property ad optimizer
Add your property `description`, a few data points and zap! We will optimize your ad based on the ads that helped the fastest sold homes near you.
"""

############################################
# GET DATA FROM REALTOR
############################################

def get_comps():
    with open('reator_data.json', 'r') as f:
        data = json.load(f)

    filtered_data = []
    for r in data:
        fields = ['url', 'text', 'soldOn', 'lastSoldPrice', 'listPrice', 'baths', 'beds', 'sqft', 'year_built']
        values = {k:v for k, v in r.items() if k in fields}
        values['soldOn'] = datetime.strptime(values['soldOn'][:10], '%Y-%m-%d')
        values['listedDate'] = r['history'][0]['listing']['list_date']
        values['listedDate'] = datetime.strptime(values['listedDate'][:10], '%Y-%m-%d')
        values['timeToSell'] = (values['soldOn'] - values['listedDate']).days
        filtered_data.append(values)

    filtered_data = pd.DataFrame(filtered_data)

    return filtered_data[filtered_data.timeToSell > 1].sort_values('timeToSell')[:5]['text'].tolist()

############################################
# PROMPT
############################################
prompt_template = """
You are a realtor advertising a new home on Zillow. Your goal is to sell it as soon as possible. 
Below are templates of home advertisement descriptions that sold very fast in the same market with similar configurations. 

--- TEMPLATES BEGIN HERE ---
{context}
--- TEMPLATES END HERE ---

Think step-by-step about what makes these templates sell faster than the copy provided. 
Rewrite the text below based on the learnings extracted from templates. 
{question}
Explain your rationale after providing you answer.

---
Helpful Answer:
"""

############################################
# GET ANSWER FROM CHATGPT
############################################

def get_answer(tpl, prompt, samples):
    tpl = PromptTemplate.from_template(tpl)
    llm = OpenAI(openai_api_key=st.secrets('OPENAI_API_KEY'), model_name='gpt-4')
    docs = RecursiveCharacterTextSplitter(chunk_size = 5000, chunk_overlap = 10).create_documents(samples)
    chain = load_qa_chain(llm, prompt=tpl, verbose=True)
    return chain.run(input_documents=docs, question=prompt)

############################################
# GENERATE RESPONSE
############################################

def generate(ad_text, zip, beds, baths, pool):
    samples = get_comps()
    with st.spinner('Processing'):
        answer = get_answer(prompt_template, ad_text, samples)
        st.write(answer)
        st.success('Done!')

############################################
# FORM
############################################

with st.form("property_form"):
    ad_text = st.text_area("Please enter the ad you want to optimize", height=300)
    zip = st.text_input("What is property zipcode?", max_chars=5)
    beds = st.number_input("How many beds?", max_value=9, min_value=1)
    baths = st.number_input("How many baths?", max_value=9, min_value=1)
    pool = st.checkbox("Has a pool?")
    submitted = st.form_submit_button(label="Generate")
    if submitted:
       if len(ad_text) < 100:
            st.error("Your property ad needs at least 100 characters")
            st.stop()
       if len(zip) != 5:
            st.error("Please enter a valid zip")
            st.stop()
       generate(ad_text, zip, beds, baths, pool)