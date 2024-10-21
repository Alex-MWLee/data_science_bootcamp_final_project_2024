import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import re
import base64

from huggingface_hub import InferenceApi
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings

# Initialize the HuggingFace inference API
hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=hf_model, huggingfacehub_api_token="please upload your key here!")

# Function to send a prompt and get the response from the model
def get_llm_response(prompt):
    try:
        response = llm(prompt)  # Call the API
        
        # If the response is just a plain string, return it
        if isinstance(response, str):
            return response
        
        # Handle other unexpected formats
        else:
            return f"Unexpected response format: {response}"
    
    except Exception as e:
        return f"Error occurred: {str(e)}"



def handle_query(query):
    # Process the query using the LLM
    response = get_llm_response(query)
    
    # Detect user intent
    if "recommend similar books" in query.lower():
        # Example of extracting the book name from the user query
        book_name = query.split("recommend similar books of ")[1].strip()
        return recommend(book_name)
    
    elif "recommend by title" in query.lower():
        book_name = query.split("recommend by title of ")[1].strip()
        return recommendByTitle(book_name)
    
    elif "recommend by description" in query.lower():
        book_name = query.split("recommend by description of ")[1].strip()
        return recommendByDesc(book_name)
    
    else:
        # Default LLM response if no specific action is detected
        return response

# Load the datasets
path_6 = "Main_Datasets/v1_Books.csv"
path_7 = "Main_Datasets/v1_Ratings.csv"
path_8 = "Main_Datasets/v1_Users.csv"

df_book_v1 = pd.read_csv(path_6)
df_rating_v1 = pd.read_csv(path_7)
df_user_v1 = pd.read_csv(path_8)

# Merging books with ratings
ratings_with_names = pd.merge(df_book_v1, df_rating_v1, on='ISBN')

# Filter for books with more than 200 ratings
x = ratings_with_names.groupby('User-ID').count()['Book-Rating'] > 200
rated_users = x[x].index
Filtered_users = ratings_with_names[ratings_with_names['User-ID'].isin(rated_users)]

# Filter for books rated by more than 50 users
y = Filtered_users['Book-Title'].value_counts() >= 50
famous_books = y[y].index
Final_ratings = Filtered_users[Filtered_users['Book-Title'].isin(famous_books)]

# Pivot table for collaborative filtering
pt = Final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
pt.fillna(0, inplace=True)

# Cosine similarity matrix for the book recommendations
similarity_score = cosine_similarity(pt)

# Recommendation function that returns title, author, image, and URLs
def recommend(book_name):
    try:
        # Check if the book exists in the pivot table (case-insensitive)
        book_name_lower = book_name.lower()
        if not any(pt.index.str.lower() == book_name_lower):
            return None  # Return None if the book is not found
        
        # Find the index of the book
        index = np.where(pt.index.str.lower() == book_name_lower)[0][0]
        similar_items = sorted(list(enumerate(similarity_score[index])), key=lambda x: x[1], reverse=True)[1:11]
        
        # Create a list of recommended books with title, author, image, and URL
        recommendations = []
        for i in similar_items:
            book_title = pt.index[i[0]]
            book_info = df_book_v1[df_book_v1['Book-Title'] == book_title].iloc[0]
            recommendations.append({
                'title': book_info['Book-Title'],
                'author': book_info['Book-Author'],
                'image': book_info['Image-URL-M'],  # URL of the book's image
                #'ISBN': book_info['ISBN'],            # Exclude ISBN for saving
                'url': f"https://www.goodreads.com/search?q={'+'.join(book_info['Book-Title'].split())}"  # Goodreads search link
            })
        return recommendations
    except IndexError:
        return []

# Preprocessing functions
nltk.download('stopwords')

def _removeNonAscii(s):
    return "".join(i for i in s if ord(i) < 128)

def make_lower_case(text):
    return text.lower()

def remove_stop_words(text):
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    return " ".join(text)

def remove_punctuation(text):
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text)
    return " ".join(text)

def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

# Load and preprocess the second dataset
path = "Data_of_example6/GR-Books.csv"
books = pd.read_csv(path)
books.columns = ['Desc', 'unamed', 'author', 'genre', 'url', 'rating', 'title']
books.drop('unamed', inplace=True, axis=1)

# Apply preprocessing steps to descriptions
books['cleaned_desc'] = books['Desc'].apply(_removeNonAscii)
books['cleaned_desc'] = books['cleaned_desc'].apply(make_lower_case)
books['cleaned_desc'] = books['cleaned_desc'].apply(remove_stop_words)
books['cleaned_desc'] = books['cleaned_desc'].apply(remove_punctuation)
books['cleaned_desc'] = books['cleaned_desc'].apply(remove_html)

# Function to recommend books based on title
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import streamlit as st

# Function to recommend books based on title
def recommendByTitle(title):
    try:
        data = books  # 'books' should already be loaded as a DataFrame with titles, authors, etc.
        data.reset_index(level=0, inplace=True)
        
        # Build an index mapping book titles (lowercased) to their row index in the dataset
        indices = pd.Series(data.index, index=data['title'].str.lower()).drop_duplicates()

        title_lower = title.lower()  # Convert the input title to lowercase for consistency

        # Check if the input title exists in the index
        if title_lower not in indices.index:
            return pd.DataFrame()  # Return an empty DataFrame if the title isn't found

        # Vectorize the titles using TF-IDF
        tfidf = TfidfVectorizer(stop_words='english')  # Creating the TF-IDF vectorizer
        tfidf_matrix = tfidf.fit_transform(data['title'])  # Fit and transform the title column

        # Compute the cosine similarity matrix based on title vectors
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Get the index of the input title from the index mapping
        idx = indices[title_lower]

        # Get pairwise similarity scores for the input title
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the books based on similarity scores, in descending order
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the indices of the most similar books (excluding the input title itself)
        sim_scores = sim_scores[1:11]  # Exclude the first result which is the same book

        # Retrieve the recommended book titles based on the indices
        book_indices = [i[0] for i in sim_scores]

        # Create the recommended book DataFrame
        recommendations = data[['title', 'author', 'url']].iloc[book_indices].copy()

        # Add Goodreads URLs for the recommended books
        recommendations['URL'] = recommendations['title'].apply(lambda x: f"https://www.goodreads.com/search?q={'+'.join(x.split())}")

        return recommendations

    except IndexError:
        return pd.DataFrame()  # Return an empty DataFrame if an error occurs


# Function to recommend books based on description
def recommendByDesc(title):
    try:
        data = books
        data.reset_index(level=0, inplace=True)
        indices = pd.Series(data.index, index=data['title'].str.lower())  # Use lowercase for indices

        title_lower = title.lower()  # Convert input title to lowercase
        if title_lower not in indices.index:
            return pd.DataFrame()  # If title not found, return an empty DataFrame

        tf = TfidfVectorizer(analyzer='word', ngram_range=(2, 2), min_df=1, stop_words='english')
        tfidf_matrix = tf.fit_transform(data['cleaned_desc'])

        sg = cosine_similarity(tfidf_matrix, tfidf_matrix)
        idx = indices[title_lower]  # Use lowercase title for indexing

        sig = list(enumerate(sg[idx]))
        sig = sorted(sig, key=lambda x: x[1], reverse=True)
        sig = sig[1:11]

        movie_indices = [i[0] for i in sig]
        rec = data[['title', 'url', 'author']].iloc[movie_indices]

        # Add URLs for recommendations
        rec['URL'] = rec['title'].apply(lambda x: f"https://www.goodreads.com/search?q={'+'.join(x.split())}")  # Change as needed

        return rec
    except IndexError:
        return pd.DataFrame()  # Return an empty DataFrame if error occurs
    
# Function to display recommendations two books per line
def display_recommendations(recommendations):
    if isinstance(recommendations, list):  # For `recommend` function (list format)
        for i in range(0, len(recommendations), 2):
            cols = st.columns(2)  # Create two columns
            for j, rec in enumerate(recommendations[i:i+2]):
                with cols[j]:
                    st.write(f"**{rec['title']}** by {rec['author']}")
                    st.image(rec['image'], width=150)
                    st.markdown(f"[View on Goodreads]({rec['url']})")
                    if st.button(f"Save '{rec['title']}'", key=rec['title']):
                        # Check if the book is already saved
                        already_saved = any(saved_book['title'] == rec['title'] for saved_book in st.session_state.saved_books)
                        if already_saved:
                            st.warning(f"'{rec['title']}' is already saved!")
                        else:
                            st.session_state.saved_books.append({
                                'title': rec['title'],
                                'author': rec['author'],
                                'image': rec['image'],
                                'url': rec['url']
                            })
                            st.success(f"'{rec['title']}' has been saved!")

    else:  # For `recommendByTitle` and `recommendByDesc` (DataFrame format)
        for i in range(0, len(recommendations), 2):
            cols = st.columns(2)  # Create two columns
            for j in range(2):
                if i + j < len(recommendations):  # Check if within bounds
                    row = recommendations.iloc[i + j]
                    with cols[j]:
                        st.write(f"**{row['title']}** by {row['author']}")
                        st.image(row['url'], width=150)  # Assuming 'url' contains the image URL
                        st.markdown(f"[View on Goodreads]({row['URL']})")
                        if st.button(f"Save '{row['title']}'", key=row['title']):
                            # Check if the book is already saved
                            already_saved = any(saved_book['title'] == row['title'] for saved_book in st.session_state.saved_books)
                            if already_saved:
                                st.warning(f"'{row['title']}' is already saved!")
                            else:
                                st.session_state.saved_books.append({
                                    'title': row['title'],
                                    'author': row['author'],
                                    'image': row['url'],
                                    'url': row['URL']
                                })
                                st.success(f"'{row['title']}' has been saved!")

# Part of the philosophy shelf 

# Data Source
path_10 = "goodreads_philosophy_books.csv"
philosophy_books = pd.read_csv(path_10)

# Convert 'Ratings Count' to integer by removing commas and changing data type
philosophy_books['Ratings Count'] = philosophy_books['Ratings Count'].str.replace(',', '').astype(int)

# Function to search by keywords in the 'Title'
def search_by_title(df, keyword):
    result = df[df['Title'].str.contains(keyword, case=False, na=False)].drop_duplicates()
    return result

# Function to search by 'Author'
def search_by_author(df, author_name):
    result = df[df['Author'].str.contains(author_name, case=False, na=False)].drop_duplicates()
    return result

# Part of Streamlit UI Setup
# Function to read and encode the image
def get_base64_of_image(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Get the base64 string of the background image
background_image = get_base64_of_image("Backgroundpic.avif")

# Set up Streamlit UI
st.markdown(
    f"""
    <style>
    .main {{
        background: url("data:image/avif;base64,{background_image}");
        background-size: cover;
        color: white;
    }}
    .sidebar-title-small {{
        font-size: 18px;  /* Adjust this value to make the title smaller */
        font-weight: bold;
        margin-bottom: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Define the title of app
st.title("WuKong Book Lovers Centre")

# Create a sidebar for page selection
page = st.sidebar.selectbox("Select a page", ["Recommendation Options", "Do you prefer to chat with WuKong?", "Philosophy Shelf"])

# Initialize session states for storing saved books
if 'saved_books' not in st.session_state:
    st.session_state.saved_books = []

if page == "Do you prefer to chat with WuKong?":
    st.sidebar.title("WuKong Personal Assistent")
    st.sidebar.markdown('<h3 class="sidebar-title-small">Hello! I am Wukong. How can I help you?</h3>', unsafe_allow_html=True)
    # User input for the chatbot
    user_input = st.text_input("Are you looking for some book recommendations? If yes, you can use this syntax to chat: recommend similar books of 'your favorite book'. ")


# Display chatbot response based on user input
    if user_input:
        chatbot_response = handle_query(user_input)
    
        if isinstance(chatbot_response, str):  # If it's a simple response from LLM
            st.write(chatbot_response)
        elif isinstance(chatbot_response, list):  # If it's a list of recommendations
            display_recommendations(chatbot_response)
        elif isinstance(chatbot_response, pd.DataFrame):  # If recommendations are returned as a DataFrame
            display_recommendations(chatbot_response)



# st.sidebar.title("Saved Books Overview")
    st.sidebar.markdown('<h3 class="sidebar-title-small">Saved Books Overview</h3>', unsafe_allow_html=True)
    if st.session_state.saved_books:
        st.sidebar.write("### Saved Books:")
        for saved_book in st.session_state.saved_books:
            st.sidebar.image(saved_book['image'], width=60)
            st.sidebar.write(f"**Title:** {saved_book['title']}")
            st.sidebar.write(f"**Author:** {saved_book['author']}")
            st.sidebar.markdown(f"[View on Goodreads]({saved_book['url']})")
            st.sidebar.write("---")  # Add a separator between books
    else:
        st.sidebar.write("No books saved yet.")

# Button to download all saved books as CSV
    if st.sidebar.button("Download all saved books as CSV"):
        if st.session_state.saved_books:
            # Convert the saved books list into a DataFrame
            saved_books_df = pd.DataFrame(st.session_state.saved_books)
        
            # Generate CSV data from the DataFrame
            csv = saved_books_df.to_csv(index=False)
        
            # Provide a download button in the sidebar
            st.sidebar.download_button(label="Download CSV", data=csv, file_name='saved_books.csv', mime='text/csv')
        else:
            st.sidebar.warning("No books saved yet.")   

# Page 1: Recommendation Options
if page == "Recommendation Options":
    st.sidebar.title("Recommendation Options")
    option = st.sidebar.radio(
        "Choose a Recommendation Method",
        ('Based on Book Rating', 'Based on Book Title', 'Based on Book Description')
    )

    # Input field for book title
    book_title = st.text_input("Enter one complete name of your favorite book:")

    # Recommendation based on the selected method
    if book_title:
        st.write(f"Recommendations for '{book_title}':")

        if option == 'Based on Book Rating':
            recommendations = recommend(book_title)
        elif option == 'Based on Book Title':
            recommendations = recommendByTitle(book_title)
        else:  # Based on Book Description
            recommendations = recommendByDesc(book_title)

        # Adjusted condition to check for empty DataFrame or None
        if recommendations is None or (isinstance(recommendations, pd.DataFrame) and recommendations.empty):
            st.write("No recommendations found. Please check your inputs or try another title.")
        else:
            display_recommendations(recommendations)  # Call the function to display books in two columns per row

    # Download all saved books as CSV
    # st.sidebar.title("Saved Books Overview")
    st.sidebar.markdown('<h3 class="sidebar-title-small">Saved Books Overview</h3>', unsafe_allow_html=True)
    if st.session_state.saved_books:
        st.sidebar.write("### Saved Books:")
        for saved_book in st.session_state.saved_books:
            st.sidebar.image(saved_book['image'], width=60)
            st.sidebar.write(f"**Title:** {saved_book['title']}")
            st.sidebar.write(f"**Author:** {saved_book['author']}")
            st.sidebar.markdown(f"[View on Goodreads]({saved_book['url']})")
            st.sidebar.write("---")  # Add a separator between books
    else:
        st.sidebar.write("No books saved yet.")

    # Button to download all saved books as CSV
    if st.sidebar.button("Download all saved books as CSV"):
        if st.session_state.saved_books:
            # Convert the saved books list into a DataFrame
            saved_books_df = pd.DataFrame(st.session_state.saved_books)
        
            # Generate CSV data from the DataFrame
            csv = saved_books_df.to_csv(index=False)
        
            # Provide a download button in the sidebar
            st.sidebar.download_button(label="Download CSV", data=csv, file_name='saved_books.csv', mime='text/csv')
        else:
            st.sidebar.warning("No books saved yet.")

# Page 2: Philosophy Shelf Features
elif page == "Philosophy Shelf":
    st.sidebar.title("Philosophy Shelf")
    options = st.sidebar.radio("Select an option", 
                               ['Top 20 Best Rated Books', 
                                'Top 20 Most Rated Books', 
                                'Search by Title', 
                                'Search by Author'])

    # Ranking of the top 20 most rated philosophy books
    if options == 'Top 20 Most Rated Books':
        st.header("Top 20 Most Rated Philosophy Books")
        most_rated_books = philosophy_books.sort_values(by='Ratings Count', ascending=False).drop_duplicates(subset='Ratings Count').head(20)

        # Displaying books in a gallery format
        cols = st.columns(5)  # Set up 5 columns for displaying the books
        for index, (idx, row) in enumerate(most_rated_books.iterrows()):
            with cols[index % 5]:  # This ensures 5 books per row
                # Display the book image
                st.image(row['Image URL'], width=100)

                # Display the average rating
                st.write(f"**Rating:** {row['Average Rating']}")

                # Generate the Goodreads URL
                goodreads_url = f"https://www.goodreads.com/search?q={'+'.join(row['Title'].split())}"

                # Display the "View on Goodreads" link using markdown
                st.markdown(f"[View on Goodreads]({goodreads_url})")

    # Ranking of the top 20 best rated philosophy books
    if options == 'Top 20 Best Rated Books':
        st.header("Top 20 Best Rated Philosophy Books")
        best_rated_books = philosophy_books.sort_values(by='Average Rating', ascending=False).drop_duplicates(subset='Average Rating').head(20)

        # Displaying books in a gallery format
        cols = st.columns(5)  # Set up 5 columns for displaying the books
        for index, (idx, row) in enumerate(best_rated_books.iterrows()):
            with cols[index % 5]:  # This ensures 5 books per row
                # Display the book image
                st.image(row['Image URL'], width=100)

                # Display the average rating
                st.write(f"**Rating:** {row['Average Rating']}")

                # Generate the Goodreads URL
                goodreads_url = f"https://www.goodreads.com/search?q={'+'.join(row['Title'].split())}"

                # Display the "View on Goodreads" link using markdown
                st.markdown(f"[View on Goodreads]({goodreads_url})")

    # Search function by keyword in title
    if options == 'Search by Title':
        st.header("Search Philosophy Books by Title")
        search_keyword = st.text_input("Enter a keyword to search in the title:")
        if search_keyword:
            search_results = search_by_title(philosophy_books, search_keyword)
            if not search_results.empty:
                st.table(search_results[['Title', 'Author', 'Average Rating', 'Ratings Count']])
            else:
                st.write("No results found for the keyword:", search_keyword)

    # Search function by author
    if options == 'Search by Author':
        st.header("Search Philosophy Books by Author")
        author_name = st.text_input("Enter an author name to search:")
        if author_name:
            search_results = search_by_author(philosophy_books, author_name)
            if not search_results.empty:
                st.table(search_results[['Title', 'Author', 'Average Rating', 'Ratings Count']])
            else:
                st.write("No results found for the author:", author_name)
