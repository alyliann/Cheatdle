import os
import math
import json
import numpy as np
import pandas as pd
import datetime as dt
import streamlit as st
import itertools as it
from datetime import datetime
from scipy.stats import entropy
import pickle
import altair as alt
import plotly.express as px
from dotenv import load_dotenv
import gc
import logging

st.set_page_config(
    page_title="Cheatdle",
    page_icon="🟩"
)

st.logo('captures/cheatdle.png')

st.header("🎯 Score Predictor")
# Load datasets
st.markdown(
"""
Enter any **5-letter word**, and we'll predict the average number of guesses it'll take everyone to get it! 💭
We'll also show worldwide statistics to see how your word stacks up.
""")
word = st.text_input("Enter a 5-letter word:", max_chars=5, key="forest").lower()
if word:
    # Validate the word
    if not word.isalpha() or len(word) != 5:
        st.error("Please enter a valid 5-letter word.")
    else:
        st.success(f"Making inference...")

        @st.cache_data
        def load_data():
            words = pd.read_csv("data/words_freq.csv")
            tweets = pd.read_csv("data/tweets.zip")
            tweets["score"] = tweets["tweet_text"].str[11]
            tweets["score"] = pd.to_numeric(tweets['score'], errors='coerce')
            tweets.rename(columns={"wordle_id": "day"}, inplace=True)
            words.dropna(inplace=True)
            words["day"] = pd.to_numeric(words['day'], errors='coerce')
            freqs = pd.read_csv("data/letter-frequencies.csv")
            freqs = freqs[["Letter", "English"]]
            freqs = freqs["English"].tolist()
            df = pd.merge(words, tweets, on='day')
            df.drop(columns=['tweet_id'], inplace=True)
            averages = df.groupby("word", as_index=False)['score'].mean()

            percents = [0.08, 4.61, 24.68, 37.27, 24.86, 7.98, 2.65]
            labels = ["1st", "2nd", "3rd", "4th", "5th", "6th", "Loss"]
            chart_data = pd.DataFrame(
                {
                    "Tries": labels,
                    "Percentage": percents,
                }
            )
            countries = pd.read_csv("data/countries.csv")
            global_cities = pd.read_csv("data/top10_global_cities.csv")
            us_cities = pd.read_csv("data/top10_us_cities.csv")
            states = pd.read_csv("data/states.csv")
            return df, freqs, averages, chart_data, countries, global_cities, us_cities, states
        df, freqs, averages, chart_data, countries, global_cities, us_cities, states = load_data()

        @st.cache_resource
        def load_model():
            filename = 'data/wordle_prediction.pkl'
            model = pickle.load(open(filename, 'rb'))
            return model
        model = load_model()


        # For any given word:
        #    1. Put the word in lower case
        #    2. Extract each letter in the word and make it it's own column
        #    3. Convert to ASCII number using ord() function
        #    4. subtract 97 to simplify char to number representation (a = 0, b = 1, c = 2, ...)
        #    5. Use number representation as index in frequency array

        def predict_score(word):
            if (not word.isalpha() or len(word) != 5):
                raise Exception(
                    "Invalid word format. Please enter a five letter word using only alphabetic characters.")
            df = pd.DataFrame()
            df["word"] = [word]
            df["letter_1"] = df["word"].str.lower().str[0].apply(ord) - 97
            df["letter_2"] = df["word"].str.lower().str[1].apply(ord) - 97
            df["letter_3"] = df["word"].str.lower().str[2].apply(ord) - 97
            df["letter_4"] = df["word"].str.lower().str[3].apply(ord) - 97
            df["letter_5"] = df["word"].str.lower().str[4].apply(ord) - 97
            df["freq"] =    freqs[df["letter_1"][0]] + \
                            freqs[df["letter_2"][0]] + \
                            freqs[df["letter_3"][0]] + \
                            freqs[df["letter_4"][0]] + \
                            freqs[df["letter_5"][0]]
            df.drop(columns=["word"], inplace=True)
            return model.predict(df)
        
        prediction = predict_score(word)
        # If word isn't found in tweet data, None is returned for the average score
        average = None
        if word in averages["word"].values:
            average = averages[averages["word"] == word]["score"].item()
        st.subheader(f"Results for '{word}':")
        col1, col2= st.columns(2)
        with col1:
            st.subheader("🌳")
            st.markdown("**Predicted average score via random forests:**")
            st.subheader("{:0.2f}".format(prediction[0]))
        with col2:
            # Print average score according to tweet data if the word exists in it
            st.subheader("𝕏")
            if average == None:
                st.markdown(("**No data found for this word in tweet data.**"))
            else:
                st.markdown("**Average score via tweet data:**")
                st.subheader("\t\t\t{:0.2f}".format(average))
        # 3.83 is the average number of turns in Wordle
        if prediction > 3.83:
            st.subheader("🤔 Your word is hard to guess!")
            st.markdown("The average Wordle score is **3.83**. Looks like you chose a tough one!")
        else:
            st.subheader("🥳 Streak savior!")
            st.markdown("The average Wordle score is **3.83**. Looks like the average person should be able to figure this one out.")
        st.markdown("**Refer to the chart below to see the percentage breakdown for the results of every Wordle game!**")

        c = alt.Chart(chart_data).mark_bar().encode(x='Tries', y='Percentage')
        st.altair_chart(c, use_container_width=True) 
        st.subheader("🌎 Your word vs. the world")

        def get_bounds(scores, names, prediction):
            if prediction > max(scores):
                return None, float('inf')
            elif prediction < min(scores):
                return float('-inf'), None
            
            idx = np.argsort(scores)
            names = np.array(names)[idx]
            scores.sort()
            higher = float('inf')
            lower = float('-inf')
            for i in range(len(scores)):
                if scores[i] > prediction and scores[i] < higher:
                    higher = i
                if scores[i] < prediction and scores[i] > lower:
                    lower = i
            return higher, lower

        st.markdown("### Global ranking")
        st.markdown("The below chart shows a map of the world organized by the **average scores of each country**.")
        names = countries["Country"].tolist()
        scores = countries["Score"].tolist()
        higher, lower = get_bounds(scores, names, prediction)
        if higher == None:
            st.markdown("The predicted score of your word is **higher** than all of the countries around the world.  \n Broadly speaking, your word may be difficult to guess around the world!  \n")
        elif lower == None:
            st.markdown("The predicted score of your word is **lower** than all of the countries around the world.  \n Broadly speaking, your word may be easy to guess around the world! \n")
        else:
            st.markdown(f"The predicted score of your word is **higher than {names[lower]}'s score ({scores[lower]})** and **lower than {names[higher]}'s score ({scores[higher]})**.  \n")
        fig = px.choropleth(countries, locations="Code", color="Score", color_continuous_scale="Viridis", hover_name="Country", range_color=(3, 4))
        st.plotly_chart(fig)
        st.markdown("### Global city ranking")
        st.markdown("The below chart shows the **10 cities worldwide with the best scores**.")
        scores = global_cities["Score"].tolist()
        names = global_cities["City"].tolist()
        higher, lower = get_bounds(scores, names, prediction)
        if higher == None:
            st.markdown("The predicted score of your word is **higher** than all of the scores of the top 10 global cities.  \n Maybe you can stump them!  \n")
        elif lower == None:
            st.markdown("The predicted score of your word is **lower** than all of the scores of the top 10 global cities.  \n How easily they can guess your word?  \n")
        else:
            st.markdown(f"The predicted score of your word is **higher than {names[lower]}'s score ({scores[lower]})** and **lower than {names[higher]}'s score ({scores[higher]})**.  \n")
        c = alt.Chart(global_cities).mark_bar().encode(x=alt.X('Score:Q', scale=alt.Scale(domain=(3.5, 3.72), clamp=True)), y=alt.Y('City:O', axis=alt.Axis(labelLimit=200)).sort('x'))
        st.altair_chart(c.properties(height = 500), use_container_width=True) 
        st.markdown("### United States state ranking")
        st.markdown("The below chart shows a map of the United States organized by the **average scores of each state**.")
        names = states["State"].tolist()
        scores = states["Score"].tolist()
        higher, lower = get_bounds(scores, names, prediction)
        if higher == None:
            st.markdown("The predicted score of your word is **higher** than all of the scores of each U.S. state.  \n Your word might be tough for the average American!  \n")
        elif lower == None:
            st.markdown("The predicted score of your word is **lower** than all of the scores of each U.S. state.  \n Can the average American guess your word easily?  \n")
        else:
            st.markdown(f"The predicted score of your word is **higher than {names[lower]}'s score ({scores[lower]})** and **lower than {names[higher]}'s score ({scores[higher]})**.  \n")
        fig = px.choropleth(states, locations="Abbreviation", locationmode="USA-states", color="Score", scope="usa", hover_name="State", color_continuous_scale="Viridis", range_color=(3, 4),)
        st.plotly_chart(fig)
        st.markdown("### United States city ranking")
        st.markdown("The below chart shows the **10 cities in the United States with the best scores**.")
        names = us_cities["City"].tolist()
        scores = us_cities["Score"].tolist()
        higher, lower = get_bounds(scores, names, prediction)
        if higher == None:
            st.markdown("The predicted score of your word is **higher** than all of the scores of the top 10 U.S. cities.  \n Maybe you can stump them!  \n")
        elif lower == None:
            st.markdown("The predicted score of your word is **lower** than all of the scores of the top 10 U.S. cities.  \n Wonder how easily they can guess your word?  \n")
        else:
            st.markdown(f"The predicted score of your word is **higher than {names[lower]}'s score ({scores[lower]})** and **lower than {names[higher]}'s score ({scores[higher]})**.  \n")
        c = alt.Chart(us_cities).mark_bar().encode(x=alt.X('Score:Q', scale=alt.Scale(domain=(3.5, 3.67), clamp=True)), y=alt.Y('City:O').sort('x'))
        st.altair_chart(c.properties(height = 500), use_container_width=True) 