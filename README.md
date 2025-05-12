# Panasonic Products RAG Chatbot

## Overview
This repository hosts a Retrieval-Augmented Generation (RAG) chatbot built with Chainlit, designed to query a dataset of **168 Panasonic products** across **9 categories**. The dataset is detailed in the *Panasonic Products: Complete Guide and Search Reference.pdf*. The main application (`app.py`) enables users to interact with product information using a local RAG pipeline.

## Dataset Details
The dataset includes 168 products with attributes like Product ID, Category, Title, Price (INR), Image URL, Key Features, Specifications (JSON), and Product URL. Key dataset files:
- `data/FullyFinalPanasonicProducts_WithHotKeywords.csv`: Primary dataset with search keywords.
- `data/CleanedPanasonic_Products.csv`: Cleaned dataset version.
- `data/d2c_panasonic_chatbot_dataset.csv`: Chatbot-specific dataset.

## Setup and Running the Application
### Prerequisites
- Python 3.8+
- Git
- Chainlit
- Virtual environment

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/sxi9/D2C_panasonic_chatbot.git
   cd D2C_panasonic_chatbot