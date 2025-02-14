# 🌦️ AI Weather Chatbot

![Python](https://img.shields.io/badge/python-3.8%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green)

This is a simple AI-powered weather chatbot using `customtkinter` for the GUI, OpenWeatherMap API for weather data, and `Ollama` for language processing.

## 📖 Table of Contents
- [🚀 Features](#-features)
- [⚙️ Installation](#️-installation)
- [🛠 Usage](#-usage)
- [🤖 Download Ollama Model](#-download-ollama-model)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)

## 🚀 Features
- Fetches real-time weather data from OpenWeatherMap API.
- Uses Retrieval Augmented Generation (RAG) with `FAISS` and `Ollama`.
- GUI built with `customtkinter` for a modern interface.
- Provides natural-sounding weather responses.

## ⚙️ Installation

### Prerequisites
Make sure you have the following installed:
- Python 3.8+
- Git
- Required dependencies

### Clone Repository
```sh
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

### Setup `.env` File
Create a `.env` file in the root directory and add the following:
```ini
WEATHER_API_KEY=your_openweather_api_key
OLLAMA_MODEL=llama3.2:3b-instruct-fp16
```
Replace `your_openweather_api_key` with your actual OpenWeatherMap API key.

### Run the Application
```sh
python WeatherApp.py
```

## 🛠 Usage
- Enter a city name in the chatbox and ask for the weather.
- The chatbot will respond with weather details using OpenWeatherMap API.
- If a non-weather-related query is entered, the bot will redirect to weather-related responses.

## 🤖 Download Ollama Model
To download the required Ollama model, run the following command:
```sh
ollama pull llama3.2:3b-instruct-fp16
```
Make sure `Ollama` is installed on your system before running the command. You can install it from [Ollama's official website](https://ollama.com/).

## 🤝 Contributing
Feel free to fork this project and submit pull requests.

## 📜 License
This project is open-source under the MIT License.
