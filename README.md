# Panasonic Products RAG Application

## Overview
This repository hosts a Retrieval-Augmented Generation (RAG) application built with Chainlit, designed to query and analyze a dataset of **168 Panasonic products** across **9 categories**. The dataset is detailed in the *Panasonic Products: Complete Guide and Search Reference.pdf*. The main application (`chat3.py`) enables users to interact with product information using a local RAG pipeline.

## Dataset Details
The dataset includes 168 products with attributes like Product ID, Category, Title, Price (INR), Image URL, Key Features, Specifications (JSON), and Product URL. Key dataset files:
- `FullyFinalPanasonicProducts_WithHotKeywords.csv`: Primary dataset with search keywords.
- `CleanedPanasonic_Products.csv`: Cleaned dataset version.
- `all_items.csv`: Comprehensive dataset.
- `d2c_panasonic_chatbot_dataset.csv`: Chatbot-specific dataset.

### Product Categories and Distribution
| Category           | Product Count | Percentage |
|--------------------|---------------|------------|
| AC                | 38            | 22.6%      |
| Cameras           | 33            | 19.6%      |
| Refrigerators     | 20            | 11.9%      |
| Small Appliances  | 17            | 10.1%      |
| Televisions       | 16            | 9.5%       |
| Washing Machines  | 16            | 9.5%       |
| Accessories       | 12            | 7.1%       |
| Audio             | 8             | 4.8%       |
| Microwave Ovens   | 8             | 4.8%       |

### Price Ranges by Category
| Category           | Min Price (₹) | Max Price (₹) | Average Price (₹) | Product Count |
|--------------------|---------------|---------------|-------------------|---------------|
| Cameras           | 17,247        | 249,156       | 124,793           | 33            |
| Televisions       | 15,990        | 239,990       | 72,928            | 16            |
| AC                | 30,990        | 67,990        | 44,740            | 38            |
| Refrigerators     | 27,990        | 71,990        | 39,991            | 20            |
| Washing Machines  | 8,990         | 32,990        | 18,284            | 16            |
| Audio             | 8,066         | 32,990        | 16,492            | 8             |
| Microwave Ovens   | 5,890         | 17,790        | 11,160            | 8             |
| Small Appliances  | 1,090         | 12,750        | 5,726             | 17            |
| Accessories       | 153           | 1,600         | 708               | 12            |

## Setup and Running the Application
### Prerequisites
- Python 3.8+
- Git
- Chainlit
- Virtual environment

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/uzirox76/localRAG.git
   cd localRAG
