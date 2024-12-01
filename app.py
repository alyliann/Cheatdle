import os
import json
import math
import pygame
import random
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import itertools as it
from textblob import TextBlob
from scipy.stats import entropy
import pickle
import altair as alt
import plotly.express as px
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

st.set_page_config(
    page_title="Cheatdle",
    page_icon="🟩"
)

st.image('captures/cheatdle.png', width=300)


# Begin streamlit code:

sentiment, forest, rag = st.tabs(["Sentiment", "Forest", "RAG"])

with sentiment:
    st.header("🚀 Sentiment Analysis")
    st.markdown(
        """
        Enter any **5-letter Wordle word**, and we'll analyze how people on Twitter felt about it! 🎉  
        We'll also visualize sentiment trends and provide deeper insights into the sentiment distribution.
        """
    )

    @st.cache_resource
    def load_data(file_path, chunk_size=10000):
        return pd.read_csv(file_path, chunksize=chunk_size)

    # Load datasets
    try:
        words_freq = pd.read_csv("data/words_freq.csv")
        tweets = load_data("data/tweets.zip")
    except FileNotFoundError as e:
        st.error(f"Error: {e}. Ensure the file paths are correct.")
        st.stop()

    # Input Word
    word = st.text_input("Enter a 5-letter Wordle word:", max_chars=5, key="sentiment").lower()

    if word:
        # Validate the word
        if not word.isalpha() or len(word) != 5:
            st.error("Please enter a valid 5-letter word.")
        else:
            # Check if word exists in dataset
            word_entry = words_freq[words_freq["word"].str.lower() == word]

            if word_entry.empty:
                st.error(f"The word '{word}' was not found in the dataset.")
            else:
                # Get Wordle day and filter tweets
                wordle_day = int(word_entry.iloc[0]["day"])
                wordle_tweets = tweets[tweets["wordle_id"] == wordle_day]

                if wordle_tweets.empty:
                    st.error(f"No tweets found for Wordle #{wordle_day}.")
                else:
                    st.success(f"Analyzing tweets for Wordle #{wordle_day}...")

                    # Sentiment Analysis
                    sentiments = {"positive": 0, "neutral": 0, "negative": 0}
                    polarity_scores = []

                    for _, row in wordle_tweets.iterrows():
                        text = row["tweet_text"]
                        # Skip grid-only tweets
                        if text.count('\n') <= 1 and text.startswith("Wordle"):
                            continue

                        cleaned_text = ' '.join([
                            line for line in text.split('\n')
                            if not line.strip().startswith(('Wordle', '⬛', '⬜', '🟨', '🟩'))
                        ])

                        if cleaned_text.strip():
                            analysis = TextBlob(cleaned_text)
                            polarity = analysis.sentiment.polarity
                            polarity_scores.append(polarity)

                            if polarity > 0:
                                sentiments["positive"] += 1
                            elif polarity < 0:
                                sentiments["negative"] += 1
                            else:
                                sentiments["neutral"] += 1

                    total = sum(sentiments.values())

                    # Results Display
                    if total == 0:
                        st.warning("No valid tweets found for analysis.")
                    else:
                        avg_sentiment = sum(polarity_scores) / len(polarity_scores)
                        sentiment_label = "😊 Positive" if avg_sentiment > 0 else "😐 Neutral" if avg_sentiment == 0 else "😟 Negative"

                        st.subheader(f"Results for '{word}' (Wordle #{wordle_day}):")
                        st.markdown(f"**Total Tweets Analyzed:** {total}")
                        st.markdown(f"**Average Sentiment:** {sentiment_label} ({avg_sentiment:.3f})")

                        # Sentiment Breakdown with Metrics
                        st.markdown("### Sentiment Breakdown")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Positive 😊", sentiments["positive"])
                        col2.metric("Neutral 😐", sentiments["neutral"])
                        col3.metric("Negative 😟", sentiments["negative"])

                        # Sentiment Polarity Distribution
                        st.markdown("### Sentiment Polarity Distribution")
                        polarity_data = pd.DataFrame({"Polarity": polarity_scores})
                        fig = px.histogram(
                            polarity_data,
                            x="Polarity",
                            nbins=20,
                            title="Polarity Score Distribution",
                        )
                        fig.update_layout(
                            bargap=0.2,
                            xaxis_title="Polarity",
                            yaxis_title="Tweet Count",
                        )
                        st.plotly_chart(fig, use_container_width=True)

with forest:
    st.header("🎯 Score Predictor")
    # Load datasets
    try:
        tweets = pd.read_csv("data/tweets.zip", nrows=5000)
        words = pd.read_csv("data/words_freq.csv")
    except FileNotFoundError as e:
        st.error(f"Error: {e}. Ensure the file paths are correct.")
        st.stop()
    st.markdown(
    """
    Enter any **5-letter Wordle word**, and we'll predict the average number of guesses it'll take someone to guess it!
    """)
    word = st.text_input("Enter a 5-letter Wordle word:", max_chars=5, key="forest").lower()
    if word:
        # Validate the word
        if not word.isalpha() or len(word) != 5:
            st.error("Please enter a valid 5-letter word.")
        else:
            st.success(f"Running random forest...")
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
            @st.cache_resource
            def load_model():
                # Load the prediction model once
                return pickle.load(open("data/wordle_prediction.pkl", "rb"))

            model = load_model()
            # For any given word:
            #    1. Put the word in lower case
            #    2. Extract each letter in the word and make it it's own column
            #    3. Convert to ASCII number using ord() function
            #    4. subtract 96 to simplify char to number representation (a = 1, b = 2, c = 3, ...)
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
            averages = df.groupby("word", as_index=False)['score'].mean()
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
            percents = [0.08, 4.61, 24.68, 37.27, 24.86, 7.98, 2.65]
            labels = ["1st", "2nd", "3rd", "4th", "5th", "6th", "Loss"]
            chart_data = pd.DataFrame(
                {
                    "Tries": labels,
                    "Percentage": percents,
                }
            )
            c = alt.Chart(chart_data).mark_bar().encode(x='Tries', y='Percentage')
            st.altair_chart(c, use_container_width=True) 
            st.subheader("🌎 Your word vs. the world")
            countries = pd.read_csv("data/countries.csv")
            global_cities = pd.read_csv("data/top10_global_cities.csv")
            us_cities = pd.read_csv("data/top10_us_cities.csv")
            states = pd.read_csv("data/states.csv")
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
                st.markdown("The predicted score of your word is **higher** than all of the scores of each of every U.S. state.  \n Your word might be tough for the average American!  \n")
            elif lower == None:
                st.markdown("The predicted score of your word is **lower** than all of the scores of each of every U.S. state.  \n Can the average American guess your word easily?  \n")
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



with rag:
    # Load environment variables
    load_dotenv()
    api_key = st.secrets["OPENAI_API_KEY"]

    # Set environment variable to handle tokenizer warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Check for API key before proceeding
    if not api_key:
        st.error(
            "OpenAI API key not found! Please set OPENAI_API_KEY in your .env file")
        st.write("1. Create a .env file in your project directory")
        st.write(
            "2. Add your OpenAI API key like this: OPENAI_API_KEY=sk-your_api_key_here")
        st.write(
            "3. Make sure the .env file is in the same directory as your Python script")
        st.stop()

    @st.cache_resource
    def initialize_qa_chain():
        try:
            # Get the absolute path to the PDF relative to the script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            pdfpath = os.path.join(
                script_dir, "data/CTP Project Design Doc (3).pdf")

            # Check if PDF exists
            if not os.path.exists(pdfpath):
                st.error(f"PDF file not found at: {pdfpath}")
                st.stop()

            st.write(f"Loading PDF from: {pdfpath}")

            # Load PDF
            loader = PyPDFLoader(pdfpath)
            pages = loader.load()

            # Create embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'}
            )

            # Create vector store
            vectorstore = FAISS.from_documents(pages, embeddings)

            # Initialize ChatOpenAI with explicit API key
            llm = ChatOpenAI(
                temperature=0.7,
                api_key=api_key,
                model="gpt-3.5-turbo",
                max_tokens=100,
            )

            # Create the QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(),
            )

            return qa_chain

        except Exception as e:
            st.error(f"Error in initialize_qa_chain: {str(e)}")
            raise e

    # Page title
    st.title("📈 Ask about our Wordle Final Project")

    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize QA chain
    try:
        with st.spinner("Loading PDF and initializing QA system..."):
            qa_chain = initialize_qa_chain()
        st.success("QA system initialized successfully!")
    except Exception as e:
        st.error(f"Error initializing QA chain: {str(e)}")
        st.stop()

    # Placeholder for chat messages
    chat_placeholder = st.empty()

    # Render chat history dynamically
    with chat_placeholder.container():
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input with debouncing
    if "last_prompt" not in st.session_state:
        st.session_state.last_prompt = ""

    prompt = st.chat_input("Ask a question about the Wordle Final Project")

    if prompt and prompt != st.session_state.last_prompt:
        st.session_state.last_prompt = prompt

        # Display user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Refresh chat history
        with chat_placeholder.container():
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Process the input and get a response
        try:
            with st.spinner("Searching document for answer..."):
                response = qa_chain.invoke(prompt)

            result = response["result"]

            # Display assistant response
            st.session_state.messages.append(
                {"role": "assistant", "content": result})

            # Refresh chat history again
            with chat_placeholder.container():
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

        except Exception as e:
            st.error(f"Error processing question: {str(e)}")
