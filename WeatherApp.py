import customtkinter as ctk
import requests
from PIL import Image, ImageTk
import re
import ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.docstore import InMemoryDocstore
import faiss
from langchain.llms import BaseLLM
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Fetch the API key and model name from environment variables
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")  # Use the environment variable for the API key
model_name = os.getenv("MODEL_NAME")  # Use the environment variable for the model name

# Debugging statements to check if the environment variables are loaded correctly
print(f"WEATHER_API_KEY: {WEATHER_API_KEY}")
print(f"MODEL_NAME: {model_name}")

# Create a custom LangChain LLM wrapper for Ollama
class OllamaLLM(BaseLLM):
    def __init__(self, model_name):
        super().__init__()  # Initialize the base class
        self._model_name = model_name  # Use _model_name to avoid conflict with BaseLLM attributes

    def _generate(self, prompt: str, stop: list = None) -> str:
        # Using Ollama's chat method
        response = ollama.chat(model=self._model_name, messages=[{"role": "user", "content": prompt}])
        print(f"Ollama response: {response}")  # Debugging statement
        return response.message.content if response.message else "Sorry, I couldn't generate a response."

    @property
    def _llm_type(self) -> str:
        return "ollama"

# Initialize LangChain with your local Ollama model
llama_model = OllamaLLM(model_name)

# Initialize FAISS vector store with SentenceTransformers
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = embedding_model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(dimension)
docstore = InMemoryDocstore()
index_to_docstore_id = {}

vector_store = FAISS(embedding_model, index, docstore, index_to_docstore_id)

# Function to get weather data from OpenWeatherMap using requests
def get_weather(city):
    """Fetches current weather data from OpenWeatherMap."""
    print(f"Fetching weather data for city: {city}")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    weather_data = response.json()
    
    # Print the raw weather data for debugging
    print(f"Raw weather data: {weather_data}")
    
    # Extracting necessary data
    temp = weather_data["main"]["temp"]  # Temperature in Celsius
    weather = weather_data["weather"][0]["description"]  # Weather description
    wind_speed_kmh = round(weather_data["wind"]["speed"] * 3.6, 1)  # Wind speed in km/h
    
    # Return the data as a dictionary
    return {"temp": temp, "weather": weather, "wind_speed_kmh": wind_speed_kmh}

# Function to extract city and time from the user's input
def extract_city_and_time(user_input):
    """Extracts the city and the time reference (today/tomorrow/next day) from the user's input."""
    print(f"Extracting city and time from user input: {user_input}")
    
    # Define patterns for city and time
    city_pattern = r"(?:in|for|about|of)\s+([a-zA-Z\s]+)"
    time_patterns = [
        r"\btomorrow\b",  # Looking for 'tomorrow'
        r"\bnext\s+day\b",  # Looking for 'next day'
        r"\btoday\b"  # Looking for 'today' (optional if user wants to specify)
    ]
    
    # Extract time first to avoid confusion with city names
    time_match = None
    for pattern in time_patterns:
        time_match = re.search(pattern, user_input, re.IGNORECASE)
        if time_match:
            break
    
    # Remove the time reference from the input to avoid confusion with city names
    if time_match:
        user_input = re.sub(time_match.group(0), '', user_input, flags=re.IGNORECASE).strip()
    
    # Extract city
    city_match = re.search(city_pattern, user_input, re.IGNORECASE)
    
    if city_match:
        city = city_match.group(1).strip()
        time = time_match.group(0).lower() if time_match else "today"  # Default to "today" if no time is specified
        print(f"Extracted city: {city}, time: {time}")
        return city, time
    else:
        # If no city pattern is found, assume the entire input is the city name
        city = user_input.strip()
        time = time_match.group(0).lower() if time_match else "today"  # Default to "today" if no time is specified
        print(f"Extracted city: {city}, time: {time}")
        return city, time

# Function to check if the user is asking about weather
def is_weather_query(user_input):
    """Checks if the user is asking about the weather."""
    print(f"Checking if user input is a weather query: {user_input}")
    weather_keywords = ["weather", "temperature", "forecast", "condition", "climate"]
    return any(keyword in user_input.lower() for keyword in weather_keywords)

# Function to generate a response from Llama with RAG
def generate_weather_rag(city, time):
    """Generates a weather response using Retrieval Augmented Generation (RAG)."""
    print(f"Generating weather response for city: {city}, time: {time}")
    # Fetch weather data from the OpenWeatherMap API
    weather_data = get_weather(city)
    
    if not weather_data:
        return f"Sorry, I couldn't find the weather data for {city}. Please try again."
    
    # Use the weather data as a 'retrieved' document for the RAG system
    document = f"The weather in {city} for {time} is as follows: Temperature: {weather_data['temp']}°C, " \
               f"Conditions: {weather_data['weather']}, Wind Speed: {weather_data['wind_speed_kmh']} km/h."
    
    # Use the Llama model to generate the response
    input_text = f"Here is the weather data: {document}\n" \
                 f"Generate a friendly, natural, human-like weather update response based on this information."
    
    # Generate the response using LangChain and Ollama model
    response = llama_model._generate(input_text)
    print(f"Generated response: {response}")
    
    return response

# Function to handle user input and generate the weather forecast
def chat(event=None):
    """Handles user input and chatbot response."""
    user_input = user_entry.get()
    if not user_input:
        return

    chat_box.configure(state=ctk.NORMAL)
    chat_box.insert(ctk.END, f"You: {user_input}\n\n", "user")

    # Update prompt based on user input
    if is_weather_query(user_input):
        city, time = extract_city_and_time(user_input)
        if not city or city.lower() == "weather":
            chat_box.insert(ctk.END, "AI: Please specify a city for the weather.\n\n", "ai")
            chat_box.configure(state=ctk.DISABLED)
            chat_box.yview(ctk.END)
            user_entry.delete(0, ctk.END)
            return
        # Generate the weather response using RAG
        weather_summary = generate_weather_rag(city, time)
        chat_box.insert(ctk.END, f"AI: {weather_summary}\n\n", "ai")
    else:
        # Generate a response using the Llama model
        input_text = "The user asked something other than weather. Respond that you can only provide weather updates."
        response = llama_model._generate(input_text)
        chat_box.insert(ctk.END, f"AI: {response}\n\n", "ai")

    chat_box.configure(state=ctk.DISABLED)
    chat_box.yview(ctk.END)
    user_entry.delete(0, ctk.END)

def reset_chat():
    """Clears the chat box and user entry field."""
    chat_box.configure(state=ctk.NORMAL)
    chat_box.delete(1.0, ctk.END)
    chat_box.configure(state=ctk.DISABLED)
    user_entry.delete(0, ctk.END)

# GUI Setup
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("AI Weather Chatbot")
root.geometry("450x600")

# Title label
title_label = ctk.CTkLabel(root, text="AI Weather Chatbot", font=("Helvetica", 18))
title_label.pack(pady=10)

# Chat box with scrollbar
chat_frame = ctk.CTkFrame(root)
chat_frame.pack(padx=20, pady=10, fill="both", expand=True)

chat_box = ctk.CTkTextbox(chat_frame, height=20, width=50, wrap=ctk.WORD,
                          font=("Helvetica", 14), padx=10, pady=10)
chat_box.pack(side="left", fill="both", expand=True)
chat_box.configure(state=ctk.DISABLED)
chat_box.tag_config("user", foreground="lightblue", justify="right")
chat_box.tag_config("ai", foreground="lightgreen", justify="left")

scrollbar = ctk.CTkScrollbar(chat_frame, command=chat_box.yview)
scrollbar.pack(side="right", fill="y")
chat_box.configure(yscrollcommand=scrollbar.set)

# User entry with placeholder text
user_entry = ctk.CTkEntry(root, width=300, font=("Helvetica", 14), justify="left",
                          placeholder_text="Type your message here...")
user_entry.pack(pady=10, padx=20, fill="x")

# Load the send button image
send_image = Image.open("send.png")
send_image = send_image.resize((30, 30), Image.LANCZOS)
send_icon = ImageTk.PhotoImage(send_image)

# Button frame
button_frame = ctk.CTkFrame(root, fg_color="transparent")
button_frame.pack(pady=10, padx=20, fill="x")

button_font = ("Helvetica", 12)
button_width = 120
button_height = 40

# Smaller transparent send button with black border and matching hover color
send_button = ctk.CTkButton(button_frame, image=send_icon, command=chat, text="", width=40, height=40, fg_color="transparent", border_color="black", hover_color="black")
send_button.pack(side=ctk.RIGHT, padx=(0, 10))

# Reset button with darker grey hover color
reset_button = ctk.CTkButton(button_frame, text="Reset Chat", command=reset_chat, font=button_font, width=button_width, height=button_height, fg_color="black", border_color="black", hover_color="#333333")
reset_button.pack(side=ctk.LEFT, padx=(10, 0))

root.bind('<Return>', chat)
root.mainloop()