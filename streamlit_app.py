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
    llm = OpenAI(openai_api_key=st.secrets['OPENAI_API_KEY'], model_name='gpt-4')
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

dt = "A RARE FIND IN LAKENONA! GENTLY USED AND FAIRLY NEW LUXURY HOUSE AVAILABLE AT ENCLAVE AT VILLAGE WALK, LOCATED IN THE HEART OF LAKE NONA, THE FASTEST GROWING MASTER PLANNED CITY WITH WORLD CLASS AMENITIES. This luxury executive SUNCHASE model with a balcony upstairs is situated on a peaceful tree-lined street with beautiful curb appeal, the home is built on a .22-acre homesite. Mature trees and a manicured yard frame the attractive faade complete with stonework, barrel tiled roof and a brick paver driveway. House features a secondary master bedroom on the first floor, 3-car garage with 8'0 front door, open floor plan with 42 upgraded Cabinets, large kitchen island, extended patio with screen enclosure, security cameras around the house, fruit trees, upgraded tiles throughout the house except in bedrooms, ceiling fans, and motion-activated security lighting, ensuring comfort and peace of mind. Designed for entertaining, built-in wine closet, grand open spaces dressed in custom touches, turn moments into memories. Enclave is the perfect gated community with natural gas connected and grounds care done by HOA, access to excellent schools, community events, fantastic dining options & an array of new local shops. Village Walk is a 24hour guard gated master-planned maintenance free community in highly desirable Lake Nona, which is located minutes from the airport, Orlandos premiere dining & the new medical city, USTA National Campus and new KPMG facility. The HOA covers 1 gigabit internet, cable, landscape maintenance, and access to an array of amenities. Play a match on the clay tennis courts, take a refreshing dip in the recreation and lap pools, or stay in shape at the fitness center. The quality resort-style amenities at this community are exceptional, from the moment you arrive at the 24 hrs gate guarded entrance, the impeccable private roads, the up-scale 26, 000 square foot town center, state of the art fitness center, Resort style swimming pool, heated lap pool, 6 lighted clay tennis courts, gas station, post office, Salon, Spa, Photo studio, basketball court, lakeside gazebo and miles of walking /biking paths and pedestrian bridges, library/business center, card room, on-site-lifestyle & activity director, multi-purpose ballroom, to mention some of the amenities. Priced to sell! Schedule a showing TODAY!"

with st.form("property_form"):
    ad_text = st.text_area("Please enter the ad you want to optimize", height=300, value=dt)
    zip = st.text_input("What is property zipcode?", max_chars=5, value="34785")
    beds = st.number_input("How many beds?", max_value=9, min_value=1, value=3)
    baths = st.number_input("How many baths?", max_value=9, min_value=1, value=2)
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