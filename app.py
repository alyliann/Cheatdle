
import os
import json
import math
import pygame
import random
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import datetime as dt
from datetime import datetime
import itertools as it
from textblob import TextBlob
from scipy.stats import entropy
import pickle
import altair as alt
import plotly.express as px
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings  # Updated import
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

st.set_page_config(
    page_title="Cheatdle",
    page_icon="🟩"
)

st.logo('captures/cheatdle.png')

wordle, sentiment, forest, rag = st.tabs(["Game", "Sentiment", "Forest", "RAG"])

# Begin 3Blue1Brown-sampled code below:

MISPLACED = np.uint8(1)
EXACT = np.uint8(2)

SHORT_WORD_LIST_FILE = "data/valid-wordle-words.txt"  # allowed guesses
LONG_WORD_LIST_FILE = "data/wordle-answers.txt"  # possible answers
WORD_FREQ_FILE = "data/freq_map.json"
PATTERN_MATRIX_FILE = "data/pattern_matrix.npy"
ENT_SCORE_PAIRS_FILE = "data/ent_score_pairs.json"

PATTERN_GRID_DATA = dict()
CHUNK_LENGTH = 13000


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def get_word_list(short=False):
    result = []
    file = SHORT_WORD_LIST_FILE if short else LONG_WORD_LIST_FILE
    with open(file) as fp:
        result.extend([word.strip().upper() for word in fp.readlines()])
    return result


def get_word_frequencies(regenerate=False):
    if os.path.exists('data/freq_map.json') or regenerate:
        with open('data/freq_map.json') as fp:
            result = json.load(fp)
        return result
    # Otherwise, regenerate
    freq_map = dict()
    with open(WORD_FREQ_FILE) as fp:
        for line in fp.readlines():
            pieces = line.split(' ')
            word = pieces[0].upper()
            freqs = [
                float(piece.strip())
                for piece in pieces[1:]
            ]
            freq_map[word] = np.mean(freqs[-5:])
    with open(WORD_FREQ_FILE, 'w') as fp:
        json.dump(freq_map, fp)
    return freq_map


def get_frequency_based_priors(n_common=3000, width_under_sigmoid=10):
    freq_map = get_word_frequencies()
    words = np.array(list(freq_map.keys()))
    freqs = np.array([freq_map[w] for w in words])
    arg_sort = freqs.argsort()
    sorted_words = words[arg_sort]

    # We want to imagine taking this sorted list, and putting it on a number
    # line so that it's length is 10, situating it so that the n_common most common
    # words are positive, then applying a sigmoid
    x_width = width_under_sigmoid
    c = x_width * (-0.5 + n_common / len(words))
    xs = np.linspace(c - x_width / 2, c + x_width / 2, len(words))
    priors = dict()
    for word, x in zip(sorted_words, xs):
        priors[word] = sigmoid(x)
    return priors


def get_true_wordle_prior():
    words = get_word_list()
    short_words = get_word_list(short=True)
    return dict(
        (w, int(w in short_words))
        for w in words
    )


def get_possible_words(guess, pattern, word_list):
    all_patterns = get_pattern_matrix([guess], word_list).flatten()
    return list(np.array(word_list)[all_patterns == pattern])


def get_weights(words, priors):
    frequencies = np.array([priors[word] for word in words])
    total = frequencies.sum()
    if total == 0:
        return np.zeros(frequencies.shape)
    return frequencies / total


def words_to_int_arrays(words):
    return np.array([[ord(c)for c in w] for w in words], dtype=np.uint8)


def generate_pattern_matrix(words1, words2):
    # Number of letters/words
    nl = len(words1[0])
    nw1 = len(words1)  # Number of words
    nw2 = len(words2)  # Number of words

    # Convert word lists to integer arrays
    word_arr1, word_arr2 = map(words_to_int_arrays, (words1, words2))

    # equality_grid keeps track of all equalities between all pairs
    # of letters in words. Specifically, equality_grid[a, b, i, j]
    # is true when words[i][a] == words[b][j]
    equality_grid = np.zeros((nw1, nw2, nl, nl), dtype=bool)
    for i, j in it.product(range(nl), range(nl)):
        equality_grid[:, :, i, j] = np.equal.outer(
            word_arr1[:, i], word_arr2[:, j])

    # full_pattern_matrix[a, b] should represent the 5-color pattern
    # for guess a and answer b, with 0 -> grey, 1 -> yellow, 2 -> green
    full_pattern_matrix = np.zeros((nw1, nw2, nl), dtype=np.uint8)

    # Green pass
    for i in range(nl):
        # matches[a, b] is true when words[a][i] = words[b][i]
        matches = equality_grid[:, :, i, i].flatten()
        full_pattern_matrix[:, :, i].flat[matches] = EXACT

        for k in range(nl):
            # If it's a match, mark all elements associated with
            # that letter, both from the guess and answer, as covered.
            # That way, it won't trigger the yellow pass.
            equality_grid[:, :, k, i].flat[matches] = False
            equality_grid[:, :, i, k].flat[matches] = False

    # Yellow pass
    for i, j in it.product(range(nl), range(nl)):
        matches = equality_grid[:, :, i, j].flatten()
        full_pattern_matrix[:, :, i].flat[matches] = MISPLACED
        for k in range(nl):
            # Similar to above, we want to mark this letter
            # as taken care of, both for answer and guess
            equality_grid[:, :, k, j].flat[matches] = False
            equality_grid[:, :, i, k].flat[matches] = False

    # Rather than representing a color pattern as a lists of integers,
    # store it as a single integer, whose ternary representations corresponds
    # to that list of integers.
    pattern_matrix = np.dot(
        full_pattern_matrix,
        (3**np.arange(nl)).astype(np.uint8)
    )

    return pattern_matrix


def get_pattern(guess, answer):
    if PATTERN_GRID_DATA:
        saved_words = PATTERN_GRID_DATA['words_to_index']
        if guess in saved_words and answer in saved_words:
            return get_pattern_matrix([guess], [answer])[0, 0]
    return generate_pattern_matrix([guess], [answer])[0, 0]


def generate_pattern_matrix_in_blocks(many_words1, many_words2, block_length=CHUNK_LENGTH):
    block_matrix = None
    for words1 in chunks(many_words1, block_length):
        row = None

        for words2 in chunks(many_words2, block_length):
            block = generate_pattern_matrix(words1, words2)

            if row is None:
                row = block
            else:
                row = np.hstack((row, block))

        if block_matrix is None:
            block_matrix = row
        else:
            block_matrix = np.vstack((block_matrix, row))

    return block_matrix


def generate_full_pattern_matrix():
    words = get_word_list()
    pattern_matrix = generate_pattern_matrix_in_blocks(words, words)
    # Save to file
    np.save('data/pattern_matrix.npy', pattern_matrix)
    return pattern_matrix


def get_pattern_matrix(words1, words2):
    if not PATTERN_GRID_DATA:
        if not os.path.exists(PATTERN_MATRIX_FILE):
            print("Generating pattern matrix. This takes a minute, but",
                  "the result will be saved to file so that it only",
                  "needs to be computed once.")
            generate_full_pattern_matrix()
        PATTERN_GRID_DATA['grid'] = np.load(PATTERN_MATRIX_FILE)
        PATTERN_GRID_DATA['words_to_index'] = dict(zip(
            get_word_list(), it.count()
        ))

    full_grid = PATTERN_GRID_DATA['grid']
    words_to_index = PATTERN_GRID_DATA['words_to_index']

    indices1 = [words_to_index[w] for w in words1]
    indices2 = [words_to_index[w] for w in words2]
    return full_grid[np.ix_(indices1, indices2)]


def pattern_to_int_list(pattern):
    result = []
    curr = pattern
    for x in range(5):
        result.append(curr % 3)
        curr = curr // 3
    return result


def get_pattern_distributions(allowed_words, possible_words, weights):
    pattern_matrix = get_pattern_matrix(allowed_words, possible_words)

    n = len(allowed_words)
    distributions = np.zeros((n, 3**5))
    n_range = np.arange(n)
    for j, prob in enumerate(weights):
        distributions[n_range, pattern_matrix[:, j]] += prob
    return distributions


def entropy_of_distributions(distributions, atol=1e-12):
    axis = len(distributions.shape) - 1
    return entropy(distributions, base=2, axis=axis)


def get_entropies(allowed_words, possible_words, weights):
    if weights.sum() == 0:
        return np.zeros(len(allowed_words))
    distributions = get_pattern_distributions(
        allowed_words, possible_words, weights)
    return entropy_of_distributions(distributions)


def optimal_guess(allowed_words, possible_words, priors):
    if len(possible_words) == 1:
        return possible_words[0]
    weights = get_weights(possible_words, priors)
    ents = get_entropies(allowed_words, possible_words, weights)

    top_ent = sorted(ents)[-10:]
    top_i = sorted(np.argsort(ents)[-10:])[::-1]
    top_guesses = []
    for num in top_i:
        top_guesses.append(allowed_words[num])

    st.session_state["suggestions"] = {}
    for i in range(min(10, len(top_guesses))):
        st.session_state["suggestions"][i] = {
            top_guesses[i]: top_ent[i]
        }

    return (allowed_words[np.argmax(ents)])


# Begin guess functions:

def get_next_guess(guesses, patterns, possibilities):
    phash = "".join(
        str(g) + "".join(map(str, pattern_to_int_list(p)))
        for g, p in zip(guesses, patterns)
    )
    if phash not in st.session_state["next_guess_map"]:
        choices = st.session_state["DICT_ANSWERS"]
        if st.session_state["hard_mode"]:
            for guess, pattern in zip(guesses, patterns):
                choices = get_possible_words(guess, pattern, choices)
        st.session_state["next_guess_map"][phash] = optimal_guess(
            choices, possibilities, st.session_state["priors"]
        )


def analyze_guesses(guess, possibilities):
    pattern = get_pattern(guess, st.session_state["answer"])
    st.session_state["patterns"].append(pattern)

    possibilities = get_possible_words(guess, pattern, possibilities)

    get_next_guess(st.session_state["guesses"],
                   st.session_state["patterns"], possibilities)
    return possibilities


# Begin Pygame code below:

def get_stats(data):
    # Turns wordle code suggestions and entropies into organized dict
    stats = {
        'Top picks': [],
        'E[Info.]': []
    }
    for index in data:
        for word, ent in data[index].items():
            stats['Top picks'].insert(0, word.lower())
            stats['E[Info.]'].insert(0, ent)
    return stats


def load_dict(file_name, upper=True):
    # Function to load dictionary
    if upper:
        with open(file_name, 'r') as f:
            words = [line.strip() for line in f.readlines()]
            return [word.upper() for word in words]
    else:
        with open(file_name, 'r') as f:
            words = [line.strip() for line in f.readlines()]
            return [word for word in words]


# Initialize necessary variables for Wordle clone

WIDTH = 600
HEIGHT = 700
MARGIN = 10
T_MARGIN = 100
B_MARGIN = 100
LR_MARGIN = 100

GREY = (70, 70, 80)
GREEN = (6, 214, 160)
YELLOW = (255, 209, 102)

pygame.init()
pygame.font.init()
SQ_SIZE = (WIDTH - 4 * MARGIN - 2 * LR_MARGIN) // 5
FONT = pygame.font.SysFont("free sans bold", SQ_SIZE)
FONT_SMALL = pygame.font.SysFont("free sans bold", SQ_SIZE // 2)

if "guesses" not in st.session_state:
    # Streamlit state initialization
    st.session_state["DICT_GUESSING"] = load_dict('data/wordle-answers.txt')
    st.session_state["DICT_ANSWERS"] = load_dict('data/wordle-answers.txt')
    st.session_state["guesses"] = []
    st.session_state["input"] = ""
    st.session_state["answer"] = random.choice(
        st.session_state["DICT_ANSWERS"])
    st.session_state["answer_date"] = None
    st.session_state["all_wordles"] = None
    st.session_state["unguessed"] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    st.session_state["found"] = ''
    st.session_state["game_over"] = False
    st.session_state["game_won"] = False
    st.session_state["hard_mode"] = False
    st.session_state["priors"] = get_frequency_based_priors()
    st.session_state["next_guess_map"] = {}
    st.session_state["patterns"] = []
    st.session_state["possibilities"] = list(
        filter(lambda w: st.session_state["priors"][w] > 0, st.session_state["DICT_ANSWERS"]))
    # Default guess suggestions:
    st.session_state["suggestions"] = {"0": {"trace": 5.8003640125599665}, "1": {"stare": 5.820775159036701}, "2": {"snare": 5.823403587185409}, "3": {"slate": 5.872115140997043}, "4": {
        "raise": 5.877133130432676}, "5": {"irate": 5.8857096269200975}, "6": {"crate": 5.895912778048746}, "7": {"crane": 5.896998055971093}, "8": {"arose": 5.9015186142727085}, "9": {"arise": 5.91076001137177}}


def determine_unguessed_letters(guesses):
    # Function to determine unguessed letters:
    guessed_letters = "".join(guesses)
    return "".join([letter for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if letter not in guessed_letters])


def determine_color(guess, j):
    # Function to determine color of letters:
    letter = guess[j]
    if letter == st.session_state["answer"][j]:
        return GREEN
    elif letter in st.session_state["answer"]:
        return YELLOW
    else:
        return GREY


def draw_guesses(surface):
    # Function to draw guesses:
    y = T_MARGIN
    for i in range(6):
        x = LR_MARGIN
        for j in range(5):
            square = pygame.Rect(x, y, SQ_SIZE, SQ_SIZE)
            pygame.draw.rect(surface, GREY, square, width=2, border_radius=3)

            if i < len(st.session_state["guesses"]):
                color = determine_color(st.session_state["guesses"][i], j)
                pygame.draw.rect(surface, color, square, border_radius=3)
                letter = FONT.render(
                    st.session_state["guesses"][i][j], False, (255, 255, 255))
                surface.blit(letter, letter.get_rect(
                    center=(x + SQ_SIZE // 2, y + SQ_SIZE // 2)))

            if i == len(st.session_state["guesses"]) and j < len(st.session_state["input"]):
                letter = FONT.render(st.session_state["input"][j], False, GREY)
                surface.blit(letter, letter.get_rect(
                    center=(x + SQ_SIZE // 2, y + SQ_SIZE // 2)))

            x += SQ_SIZE + MARGIN
        y += SQ_SIZE + MARGIN


# Begin Streamlit code:

with wordle:

    def update_answer():
        if st.session_state.date:
            st.session_state["answer_date"] = st.session_state.date.strftime(
                '%Y-%m-%d')  # Format as 'YYYY-mm-dd' string
        else:
            st.session_state["answer_date"] = None
        st.session_state["answer"] = get_wordle_by_date()
        reset_game()


    def update_mode():
        if st.session_state.hard:
            st.session_state["hard_mode"] = True
        else:
            st.session_state["hard_mode"] = False
        reset_game()


    def update_unguessed(guess):
            # Updates string of all unguessed letters
        return "".join([letter for letter in st.session_state["unguessed"] if letter not in guess])


    def update_found(guess):
        # Updates string of all found letters
        found = st.session_state["found"]
        for c in guess:
            if c not in found and c in st.session_state["answer"]:
                found += c
        return "".join(sorted(found))


    def is_valid_hard_mode_guess(guess):
        """
        Validates if a guess follows hard mode rules:
        - Must include all previously found letters
        - Must not include any previously ruled out letters
        """
        # Check if the guess contains all found letters
        for letter in st.session_state["found"]:
            if letter not in guess:
                return False, f"Hard mode: Must use found letter '{letter}'"

        # Check if the guess contains any ruled out letters
        # Only check letters that have been ruled out by previous guesses
        if len(st.session_state["guesses"]) > 0:  # Only check after first guess
            for letter in guess:
                # A letter is ruled out if:
                # 1. It's not in any previously found letters AND
                # 2. It was used in a previous guess (so it's not in unguessed anymore) AND
                # 3. It's not in the answer
                if (letter not in st.session_state["found"] and  # not found
                    # was used before
                    letter not in st.session_state["unguessed"] and
                        letter not in st.session_state["answer"]):  # not in answer
                    return False, f"Hard mode: Cannot use ruled out letter '{letter}'"

        return True, ""

    def render_frame():
        surface = pygame.Surface((WIDTH, HEIGHT))
        surface.fill("white")
        letters = FONT_SMALL.render(st.session_state["unguessed"], False, GREY)
        surface.blit(letters, letters.get_rect(center=(WIDTH // 2, T_MARGIN // 2)))
        draw_guesses(surface)
        return pygame.surfarray.array3d(surface).swapaxes(0, 1)


    def get_wordle_by_date():
        if st.session_state["answer_date"]:
            if not st.session_state["all_wordles"]:
                # initialize all_wordles dict to refer back to throughout session
                if os.path.exists('data/all_wordles.json'):
                    with open('data/all_wordles.json') as fp:
                        # load json of all past wordles and their date of occurence
                        st.session_state["all_wordles"] = json.load(fp)
                        return get_wordle_by_date()
                else:
                    st.error(
                        f'Could not retrieve Wordle for {st.session_state["answer_date"]}, randomized answer instead.')
            else:
                if st.session_state["answer_date"] in st.session_state["all_wordles"]:
                    return st.session_state["all_wordles"][st.session_state["answer_date"]]
                else:
                    # if date not found in dict
                    selected = datetime.strptime(
                        st.session_state["answer_date"], '%Y-%m-%d').date()
                    latest = datetime.strptime(
                        next(iter(st.session_state["all_wordles"])), '%Y-%m-%d').date()
                    if selected > latest:
                        st.error(
                            f"We\'re so sorry! The last time this project was updated was {latest.strftime(' % B % -d, % Y')}. We randomized the answer instead.")
                    else:
                        st.error(
                            f'Could not retrieve Wordle for {st.session_state["answer_date"]}, randomized answer instead.')
        if st.session_state["answer_date"]:
            # if date not found in all_wordles dict
            st.session_state["answer_date"] = None
        # if date not provided or error encountered preventing date-wordle matching:
        return random.choice(st.session_state["DICT_ANSWERS"])


    def rerun():
        # replaces st.rerun(), which triggers a warning in callbacks
        st.write("")  # reloads page

    def reset_game():
        st.session_state["guesses"] = []
        st.session_state["input"] = ""
        st.session_state["answer"] = random.choice(
            st.session_state["DICT_ANSWERS"]) if not st.session_state["answer_date"] else get_wordle_by_date()
        st.session_state["unguessed"] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        st.session_state["found"] = ''
        st.session_state["game_over"] = False
        st.session_state["game_won"] = False
        st.session_state["priors"] = get_frequency_based_priors()
        st.session_state["next_guess_map"] = {}
        st.session_state["patterns"] = []
        st.session_state["possibilities"] = list(
            filter(lambda w: st.session_state["priors"][w] > 0, st.session_state["DICT_ANSWERS"]))
        # Default guess suggestions:
        st.session_state["suggestions"] = {"0": {"trace": 5.8003640125599665}, "1": {"stare": 5.820775159036701}, "2": {"snare": 5.823403587185409}, "3": {"slate": 5.872115140997043}, "4": {
            "raise": 5.877133130432676}, "5": {"irate": 5.8857096269200975}, "6": {"crate": 5.895912778048746}, "7": {"crane": 5.896998055971093}, "8": {"arose": 5.9015186142727085}, "9": {"arise": 5.91076001137177}}

    def input_guess():
        guess = st.session_state.guess.upper()

        if len(guess) != 5:
            st.error("Please enter a 5-letter word.")
            st.session_state.guess = ''
            return

        if guess not in st.session_state["DICT_GUESSING"]:
            st.error("Please enter a valid guess.")
            st.session_state.guess = ''
            return

        if st.session_state["hard_mode"]:
            is_valid, error_msg = is_valid_hard_mode_guess(guess)
            if not is_valid:
                st.error(error_msg)
                st.session_state.guess = ''
                return

        st.session_state["guesses"].append(guess)
        n = len(st.session_state["guesses"])
        st.session_state["unguessed"] = update_unguessed(guess)
        st.session_state["found"] = update_found(guess)

        st.session_state["game_over"] = (
            guess == st.session_state["answer"] or len(st.session_state["guesses"]) == 6)
        st.session_state["game_won"] = guess == st.session_state["answer"]
        st.session_state.guess = ''


    # Streamlit top buttons code:

    [date, empty, mode] = st.columns([0.4, 0.1, 0.5])

    with date:
        st.date_input('Select Wordle (Random if unspecified)', value=None, min_value=dt.date(
            2021, 6, 19), max_value=dt.date.today(), format='MM/DD/YYYY', key='date', on_change=update_answer)

    with mode:
        st.write('')
        st.write('')
        st.checkbox("Hard Mode", key="hard",
                    help="In hard mode, you must use all discovered letters and cannot use ruled out letters", on_change=update_mode)

    # Remaining Streamlit code

    [wordle, empty, stats] = st.columns([0.5, 0.1, 0.4])

    with wordle:
        wordle_type = st.session_state["answer_date"] if st.session_state["answer_date"] else 'Random'
        st.subheader(f"{wordle_type} Wordle")

        frame = render_frame()
        frame_image = Image.fromarray(frame)
        with st.container(border=True, height=400):
            st.image(frame_image)

        [input, restart] = st.columns([0.7, 0.4])

        with input:
            # Input field for guesses
            if not st.session_state["game_over"]:
                st.text_input("Enter your guess:", max_chars=5,
                              key='guess', on_change=input_guess).upper()

        with restart:
            m = st.markdown("""
                <style>
                div.stButton > button:first-child {
                    background-color: #eb4242;
                    border-color: #eb4242;
                    color: #ffffff;
                    margin-top: 0.7rem;
                }
                div.stButton > button:hover {
                    background-color: #c22121;
                    border-color: #c22121;
                    color: #ffffff;
                    }
                </style>""", unsafe_allow_html=True)

            if st.button("Restart Game"):
                reset_game()
                st.rerun()

        if st.session_state["game_over"]:
            if st.session_state["game_won"]:
                st.success(
                    f"Congratulations! Score: {len(st.session_state['guesses'])}/6")
            else:
                st.error(
                    f"Game Over! The correct word was {st.session_state['answer']}")

    with stats:
        st.subheader('Guess Suggestions')

        if len(st.session_state["guesses"]) > 0:
            st.session_state["possibilities"] = analyze_guesses(
                st.session_state["guesses"][-1], st.session_state["possibilities"])

        if not st.session_state["game_over"]:
            if len(st.session_state["possibilities"]) < 3:
                stats = {
                    'Top picks': [],
                    'E[Info.]': []
                }
                for word in st.session_state["possibilities"]:
                    stats['Top picks'].append(word.lower())
                    stats['E[Info.]'].append('')
            else:
                stats = get_stats(st.session_state["suggestions"])
            df = pd.DataFrame(stats)
            st.dataframe(df, width=200, hide_index=True)

    if not st.session_state["game_over"] and len(st.session_state["possibilities"]) >= 3:
        st.divider()
        st.subheader(f'Possible Answers: {len(st.session_state["possibilities"])}')
        if st.checkbox(label="Show Possible Answers"):
            st.write(st.session_state["possibilities"])

with sentiment:
    st.header("🚀 Sentiment Analysis")
    st.markdown(
        """
        Enter any **5-letter Wordle word**, and we'll analyze how people on Twitter felt about it! 🎉  
        We'll also visualize sentiment trends and provide deeper insights into the sentiment distribution.
        """
    )

    # Load datasets
    try:
        words_freq = pd.read_csv("data/words_freq.csv")
        tweets = pd.read_csv("data/tweets.zip")
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
                st.error(f"The word '{word.upper()}' was not found in the dataset.")
            else:
                # Get Wordle day and filter tweets
                wordle_day = int(word_entry.iloc[0]["day"])
                wordle_tweets = tweets[tweets["wordle_id"] == wordle_day]

                if wordle_tweets.empty:
                    st.error(f"No tweets found for Wordle word {word.upper()}.")
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

                        st.subheader(f"Results for '{word.upper()}':")
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
    @st.cache_data
    def load_forest_data():
        tweets = pd.read_csv("data/tweets.zip")
        words = pd.read_csv("data/words_freq.csv")
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
        return freqs, df, averages, countries, chart_data, global_cities, us_cities, states
    freqs, df, averages, countries, chart_data, global_cities, us_cities, states = load_forest_data()

    # Load model
    @st.cache_resource
    def load_model():
        filename = 'data/wordle_prediction.pkl'
        model = pickle.load(open(filename, 'rb'))
        return model
    model = load_model()

    st.markdown(
    """
    Enter any **5-letter Wordle word**, and we'll predict the number of guesses it'll take everyone to get it! 💭
    We'll also show worldwide statistics to see how your word stacks up.
    """)
    word = st.text_input("Enter a 5-letter Wordle word:", max_chars=5, key="forest").lower()
    if word:
        # Validate the word
        if not word.isalpha() or len(word) != 5:
            st.error("Please enter a valid 5-letter word.")
        else:
            st.success(f"Running random forest...")

            # For any given word:
            #    1. Put the word in lower case
            #    2. Extract each letter in the word and make it it's own column
            #    3. Convert to ASCII number using ord() function
            #    4. subtract 97 to simplify char to number representation (a = 0, b = 1, c = 2, ...)
            #    5. get frequency of each character using number representation as index to frequency array 

            @st.cache_data
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
            st.subheader(f"Results for '{word.upper()}':")
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

            @st.cache_data
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
    api_key = os.getenv("OPENAI_API_KEY")

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
                script_dir, "data/CTP Project Design Doc.pdf")

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
    st.title("Ask About Cheatdle")

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

    prompt = st.chat_input("Ask a question about Cheatdle...")

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
