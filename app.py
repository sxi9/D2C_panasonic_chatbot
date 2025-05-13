import os
import re
import time
import logging
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
import chainlit as cl
from chainlit.types import ThreadDict
from functools import lru_cache
from fuzzywuzzy import fuzz

os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./appp.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

CSV_FILE_PATH = "FullyFinalPanasonicProducts_WithHotKeywords.csv"
PDF_FILE_PATH = "Panasonic Products_ Complete Guide and Search Reference.pdf"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
K_RETRIEVAL = 100
MAX_PRODUCTS_TO_SHOW = 10
PERSIST_DIRECTORY = "./chroma_db"

logger.info("Initializing Ollama embeddings and LLM")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = OllamaLLM(
    model="gemma2:2b",
    temperature=0.7,
    num_predict=200
)

query_cache = {}

def process_csv(file_path: str) -> List[Dict[str, Any]]:
    """Process CSV file and convert to list of documents."""
    start_time = time.perf_counter()
    logger.info(f"Starting CSV processing: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        documents = []
        ac_count = 0
        
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            for key, value in row_dict.items():
                if pd.isna(value):
                    row_dict[key] = ""
            
            product_text = f"Product ID: {row_dict.get('product_id', '')}\n"
            product_text += f"Category: {row_dict.get('Category', '')}\n"
            product_text += f"Title: {row_dict.get('Title', '')}\n"
            product_text += f"Price: {row_dict.get('Price', '')}\n"
            
            if 'Key Features' in row_dict and row_dict['Key Features']:
                product_text += f"Key Features: {row_dict['Key Features']}\n"
            
            if 'Specifications' in row_dict and row_dict['Specifications']:
                product_text += f"Specifications: {row_dict['Specifications']}\n"
            
            if 'Size' in row_dict and row_dict['Size']:
                product_text += f"Size: {row_dict['Size']}\n"
            
            if 'Price_Range' in row_dict and row_dict['Price_Range']:
                product_text += f"Price Range: {row_dict['Price_Range']}\n"
            
            if 'Energy_Rating' in row_dict and row_dict['Energy_Rating']:
                product_text += f"Energy Rating: {row_dict['Energy_Rating']}\n"
            
            if 'Smart_Features' in row_dict and row_dict['Smart_Features']:
                product_text += f"Smart Features: {row_dict['Smart_Features']}\n"
            
            if 'Model_Year' in row_dict and row_dict['Model_Year']:
                product_text += f"Model Year: {row_dict['Model_Year']}\n"
            
            if 'Special_Features' in row_dict and row_dict['Special_Features']:
                product_text += f"Special Features: {row_dict['Special_Features']}\n"
            
            if 'Color' in row_dict and row_dict['Color']:
                product_text += f"Color: {row_dict['Color']}\n"
            
            if 'Type' in row_dict and row_dict['Type']:
                product_text += f"Type: {row_dict['Type']}\n"
            
            if 'Warranty' in row_dict and row_dict['Warranty']:
                product_text += f"Warranty: {row_dict['Warranty']}\n"
            
            if 'Room_Size_Sqft' in row_dict and row_dict['Room_Size_Sqft']:
                product_text += f"Room Size: {row_dict['Room_Size_Sqft']}\n"
            
            if 'Hot_Search_Keywords' in row_dict and row_dict['Hot_Search_Keywords']:
                product_text += f"Hot Search Keywords: {row_dict['Hot_Search_Keywords']}\n"
            
            product_url = ""
            if 'Product URL' in row_dict and row_dict['Product URL']:
                product_url = row_dict['Product URL']
                product_text += f"Product URL: {product_url}\n"
            
            image_url = ""
            if 'Image URL' in row_dict and row_dict['Image URL']:
                image_url = row_dict['Image URL']
                product_text += f"Image URL: {image_url}\n"
            
            metadata = {
                "source": "product_catalog",
                "product_id": str(row_dict.get('product_id', '')),
                "category": str(row_dict.get('Category', '')),
                "title": str(row_dict.get('Title', '')),
                "price": str(row_dict.get('Price', '')),
                "size": str(row_dict.get('Size', '')),
                "price_range": str(row_dict.get('Price_Range', '')),
                "energy_rating": str(row_dict.get('Energy_Rating', '')),
                "smart_features": str(row_dict.get('Smart_Features', '')),
                "model_year": str(row_dict.get('Model_Year', '')),
                "special_features": str(row_dict.get('Special_Features', '')),
                "color": str(row_dict.get('Color', '')),
                "type": str(row_dict.get('Type', '')),
                "warranty": str(row_dict.get('Warranty', '')),
                "room_size_sqft": str(row_dict.get('Room_Size_Sqft', '')),
                "hot_search_keywords": str(row_dict.get('Hot_Search_Keywords', '')),
                "image_url": image_url,
                "product_url": product_url
            }
            
            documents.append({"content": product_text, "metadata": metadata})
            if metadata["category"].lower() == "ac":
                ac_count += 1
        
        logger.info(f"Loaded {len(documents)} documents, including {ac_count} AC products")
        end_time = time.perf_counter()
        logger.info(f"Completed CSV processing in {end_time - start_time:.2f} seconds")
        return documents
    except Exception as e:
        logger.error(f"Error processing CSV {file_path}: {str(e)}")
        return []

def process_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Process PDF file and convert to list of documents."""
    start_time = time.perf_counter()
    logger.info(f"Starting PDF processing: {file_path}")
    
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        documents = []
        for page in pages:
            documents.append({
                "content": page.page_content,
                "metadata": {
                    "source": "pdf_manual",
                    "page": page.metadata.get("page", 0)
                }
            })
        
        end_time = time.perf_counter()
        logger.info(f"Completed PDF processing in {end_time - start_time:.2f} seconds")
        return documents
    except Exception as e:
        logger.error(f"Error processing PDF {file_path}: {str(e)}")
        return []

def setup_retrieval_chain(documents: List[Dict[str, Any]]) -> Optional[ConversationalRetrievalChain]:
    """Set up the retrieval chain with documents."""
    start_time = time.perf_counter()
    logger.info("Starting retrieval chain setup")
    
    try:
        if os.path.exists(PERSIST_DIRECTORY):
            logger.info("Loading existing vector store")
            vectorstore = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=embeddings,
                collection_name="panasonic_products"
            )
        else:
            text_splitter = CharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separator="\n\n"
            )
            
            texts = []
            metadatas = []
            for doc in documents:
                chunks = text_splitter.split_text(doc["content"])
                for chunk in chunks:
                    texts.append(chunk)
                    metadatas.append(doc["metadata"])
            
            logger.info("Creating vector store with batch embedding")
            vectorstore_start = time.perf_counter()
            vectorstore = Chroma(
                collection_name="panasonic_products",
                embedding_function=embeddings,
                persist_directory=PERSIST_DIRECTORY
            )
            batch_size = 50
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                vectorstore.add_texts(texts=batch_texts, metadatas=batch_metadatas)
            vectorstore_end = time.perf_counter()
            logger.info(f"Vector store created in {vectorstore_end - vectorstore_start:.2f} seconds")
        
        prompt_template = """You are a Panasonic product assistant. Your role is to recommend ONLY Panasonic products from the provided context, which contains details from the Panasonic product catalog. Do NOT suggest non-Panasonic products (e.g., LG, Samsung, Xiaomi). Use the context to identify products matching the user's query, focusing on the specified category (e.g., AC, TV, Audio System), price, and other specified features. First, explain your reasoning in a concise "Reasoning" section: describe how you interpreted the query, which criteria (e.g., category, price, features) you focused on, and why you selected the products you did. If no products match the exact criteria, suggest the closest matching products and explain why they are recommended. Then, provide the answer in a separate "Answer" section. Format responses with product cards including title, price, features, image URL, and product URL. Be concise, professional, and use emojis. 😊

        Context: {context}

        Question: {question}

        **Reasoning**  
        [Explain your reasoning here based on the query.]

        **Answer**  
        [Provide the final response here, e.g., product recommendations or a message if no products are found.]"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        logger.info("Setting up conversational retrieval chain")
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_kwargs={
                    "k": K_RETRIEVAL
                }
            ),
            combine_docs_chain_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        end_time = time.perf_counter()
        logger.info(f"Completed retrieval chain setup in {end_time - start_time:.2f} seconds")
        return chain
    except Exception as e:
        logger.error(f"Error setting up retrieval chain: {str(e)}")
        return None

def is_greeting_query(query: str) -> bool:
    """Determine if the query is a greeting."""
    greetings = [
        "hi", "hello", "hey", "greetings", "good morning", "good afternoon",
        "good evening", "howdy", "yo"
    ]
    query = query.lower().strip()
    return any(greeting == query for greeting in greetings)

def is_price_query(query: str) -> bool:
    """Determine if the query is price-based."""
    price_indicators = [
        "price", "cost", "budget", "under", "below", "less than",
        "between", "around", "about", "above", "more than",
        "₹", "rs", "inr", "k"
    ]
    query = query.lower()
    return any(indicator in query for indicator in price_indicators)

def is_compare_query(query: str) -> bool:
    """Determine if the query is a comparison request."""
    compare_indicators = ["compare", "comparison", "vs", "versus"]
    query = query.lower()
    return any(indicator in query for indicator in compare_indicators)

def is_select_query(query: str) -> bool:
    """Determine if the query is a selection request."""
    query = query.lower()
    return query.startswith("select ")

def is_done_response(query: str) -> bool:
    """Determine if the user has confirmed they are done."""
    query = query.lower().strip()
    return query in ["yes", "done", "yep", "yeah", "ok", "okay"]

def extract_price_range(query: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract price range from query."""
    price_range_map = {
        "<20k": (0, 20000),
        "20k-40k": (20000, 40000),
        "40k-60k": (40000, 60000),
        "60k-100k": (60000, 100000),
        "100k-200k": (100000, 200000),
        "≥200k": (200000, float('inf'))
    }
    
    min_price_patterns = [
        r'between [\₹]?(\d+)[k]?',
        r'(\d+)[k]?\s?(?:to|and|-)',
        r'from [\₹]?(\d+)[k]?'
    ]
    
    max_price_patterns = [
        r'under [\₹]?(\d+)[k]?',
        r'below [\₹]?(\d+)[k]?',
        r'less than [\₹]?(\d+)[k]?',
        r'budget [\₹]?(\d+)[k]?',
        r'between [\₹]?\d+[k]?\s?(?:to|and|-)\s?[\₹]?(\d+)[k]?',
        r'(?:to|and|-)\s?[\₹]?(\d+)[k]?',
        r'around [\₹]?(\d+)[k]?',
        r'about [\₹]?(\d+)[k]?'
    ]
    
    query = query.lower()
    
    for range_key, (min_p, max_p) in price_range_map.items():
        if range_key.lower() in query:
            return min_p, max_p
    
    min_price = None
    for pattern in min_price_patterns:
        match = re.search(pattern, query)
        if match:
            value = match.group(1)
            if 'k' in query.lower()[match.start():match.end()]:
                min_price = int(value) * 1000
            else:
                min_price = int(value)
            break
    
    max_price = None
    for pattern in max_price_patterns:
        match = re.search(pattern, query)
        if match:
            value = match.group(1)
            if 'k' in query.lower()[match.start():match.end()]:
                max_price = int(value) * 1000
            else:
                max_price = int(value)
            break
    
    if max_price and not min_price and "around" in query:
        min_price = max(0, max_price - 5000)
        max_price = max_price + 5000
    
    return min_price, max_price

def extract_category(query: str) -> Optional[str]:
    """Extract product category from query."""
    categories = {
        "AC": ["ac", "air conditioner", "air conditioning", "cooler", "cooling"],
        "TV": ["tv", "television", "smart tv", "led tv", "oled"],
        "Refrigerator": ["refrigerator", "fridge", "freezer"],
        "Washing Machine": ["washing machine", "washer", "laundry"],
        "Microwave": ["microwave", "oven"],
        "Water Purifier": ["water purifier", "water filter", "purifier"],
        "Vacuum Cleaner": ["vacuum", "vacuum cleaner", "cleaner"],
        "Audio System": ["audio", "speaker", "sound system", "home theater", "soundbar", "party speaker"],
        "Cameras": ["camera", "camcorder", "mirrorless", "lens"],
        "Small Appliances": ["trimmer", "hair dryer", "shaver", "grooming", "vacuum cleaner"],
        "Accessories": ["accessory", "cover", "stand", "bracket", "detergent"]
    }
    
    query = query.lower()
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in query:
                return category
    
    return None

def extract_features(query: str, category: str) -> List[str]:
    """Extract product features from query, specific to the category."""
    ac_features = {
        "inverter": ["inverter", "power saving", "smart inverter"],
        "smart control": ["smart control", "smart", "wifi", "app control", "remote control", "miraie", "google assistant", "alexa"],
        "air purification": ["air purifier", "purification", "filter", "filtration", "nanoe", "pm 0.1", "ag clean"],
        "energy efficiency": ["energy", "efficient", "star rating", "power saving", "econavi"],
        "capacity": ["ton", "capacity"],
        "prime convertible": ["prime convertible", "convertible"],
        "hot and cold": ["hot and cold", "all weather"],
        "high ambient operation": ["55 deg", "high ambient", "extreme heat"],
        "dehumidification": ["dehumidification", "humidity control", "monsoon"]
    }
    
    tv_features = {
        "smart control": ["smart control", "smart", "wifi", "app control", "google assistant", "alexa"],
        "resolution": ["4k", "ultra hd", "full hd", "hd ready"],
        "screen size": ["inch", "\"", "screen size", "display size"],
        "hdr": ["hdr", "hdr10", "dolby vision", "hdr10+"],
        "dolby": ["dolby", "dolby vision", "dolby atmos"],
        "oled": ["oled"],
        "led": ["led"],
        "gaming": ["game mode", "gaming", "low latency", "120hz", "ps5", "ps 5", "playstation"]
    }
    
    audio_features = {
        "portable": ["portable", "wireless", "bluetooth", "party speaker"],
        "soundbar": ["soundbar", "home theater", "5.1ch"],
        "connectivity": ["bluetooth", "usb", "aux", "multi-connectivity"]
    }
    
    features = ac_features if category == "AC" else tv_features if category == "TV" else audio_features if category == "Audio System" else {}
    
    found_features = []
    query = query.lower()
    
    for feature, keywords in features.items():
        for keyword in keywords:
            if keyword in query:
                found_features.append(feature)
                break
    
    logger.info(f"Extracted features for category {category}: {found_features}")
    return found_features

def extract_room_size(query: str) -> Optional[int]:
    """Extract room size in square feet from user response."""
    query = query.lower()
    sq_ft_match = re.search(r'(\d+)\s*(sq\s*ft|square\s*feet|sft|sqft|ft²)', query)
    if sq_ft_match:
        return int(sq_ft_match.group(1))
    number_match = re.search(r'(\d+)', query)
    if number_match:
        return int(number_match.group(1))
    return None

def extract_location(query: str) -> Tuple[Optional[str], bool]:
    """Extract location and pollution concern from user response using the LLM."""
    logger.info(f"Extracting location from query: {query}")
    
    # Prompt to extract location
    location_prompt = f"""You are a location extraction assistant. Your task is to identify any geographical location (city, state, country, region, etc.) mentioned in the user's query. If a location is found, return only the location name. If no location is mentioned, return "None". Do not include explanations or additional text.

    Query: "{query}"

    Location: """
    
    try:
        location_response = llm.invoke(location_prompt)
        location = location_response.strip()
        if location.lower() == "none":
            logger.info(f"No location found in query: {query}")
            return None, False
        logger.info(f"Location extracted by LLM: {location}")
        
        # Prompt to determine pollution concern
        pollution_prompt = f"""You are an environmental analysis assistant. Given a location, determine if it is likely to have high air pollution. Consider factors like whether it is a major urban area, a capital city, an industrial region, or has known air quality issues (e.g., Delhi, Beijing). Rural or less industrialized areas (e.g., a small village) are unlikely to have high pollution. Respond with "Yes" if the location is likely to have high pollution, or "No" if it is not. Do not include explanations or additional text.

        Location: "{location}"

        Pollution Concern: """
        
        pollution_response = llm.invoke(pollution_prompt)
        pollution_concern = pollution_response.strip().lower() == "yes"
        logger.info(f"Pollution concern for {location}: {pollution_concern}")
        
        return location, pollution_concern
    except Exception as e:
        logger.error(f"Error using LLM to extract location: {str(e)}")
        return None, False
    
async def format_product_info(source_doc, msg: cl.Message, product_index: int = None) -> str:
    """Format product information as a Markdown product card with an optional numbered index."""
    metadata = source_doc.metadata
    
    title = metadata.get("title", "Product")
    price = metadata.get("price", "")
    category = metadata.get("category", "")
    product_id = metadata.get("product_id", "")
    size = metadata.get("size", "")
    energy_rating = metadata.get("energy_rating", "")
    smart_features = metadata.get("smart_features", "")
    image_url = metadata.get("image_url", "")
    product_url = metadata.get("product_url", "")
    
    if category.lower() in ["tv", "televisions"]:
        size_match = re.search(r'(\d+)\s*(?:cm|inches|\")', title.lower())
        if size_match:
            size_value = size_match.group(1)
            unit = "cm" if "cm" in title.lower() else "inches"
            size = f"{size_value} {unit}"
    
    content = source_doc.page_content
    key_features = ""
    if "Key Features:" in content:
        features_text = content.split("Key Features:")[1].split("\n")[0]
        key_features = features_text.strip()
    
    emoji = "🔥"
    if category.lower() == "ac" or "air conditioner" in category.lower():
        emoji = "❄️"
    elif "tv" in category.lower() or "television" in category.lower():
        emoji = "📺"
    elif "refrigerator" in category.lower() or "fridge" in category.lower():
        emoji = "🧊"
    elif "washing machine" in category.lower():
        emoji = "🧺"
    elif "microwave" in category.lower() or "oven" in category.lower():
        emoji = "🍳"
    elif "audio" in category.lower() or "speaker" in category.lower():
        emoji = "🔊"
    elif "cameras" in category.lower() or "camera" in category.lower():
        emoji = "📸"
    elif "small appliances" in category.lower():
        emoji = "🔌"
    elif "accessories" in category.lower():
        emoji = "🛠️"
    
    product_info = f"\n"
    if product_index is not None:
        product_info += f"{product_index}. "
    product_info += f"{emoji} **{title}**\n"
    
    if image_url and image_url.startswith("http"):
        product_info += f"![{title}]({image_url})\n"
    else:
        product_info += "🖼️ *Image not available*\n"
    
    product_info += f"💰 **Price**: {price}\n"
    if size:
        product_info += f"📏 **Size**: {size}\n"
    if energy_rating:
        product_info += f"⭐ **Energy Rating**: {energy_rating}\n"
    if smart_features:
        product_info += f"📱 **Smart Features**: {smart_features}\n"
    if key_features:
        product_info += f"✨ **Features**: {key_features}\n"
    product_info += f"🔍 **Product ID**: {product_id}\n"
    
    if product_url and product_url.startswith("http"):
        product_info += f"🛒 **[Buy Now]({product_url})**\n"
    else:
        product_info += "🛒 *Purchase link not available*\n"
    
    product_info += "\n---\n"
    
    for char in product_info:
        await msg.stream_token(char)
    
    return product_info

def update_selected_products(indices: List[int], displayed_docs: List[Dict[str, Any]], select: bool):
    """Update the list of selected product IDs based on numbered indices."""
    selected_product_ids = cl.user_session.get("selected_product_ids", [])
    current_products = {i: doc.metadata["product_id"] for i, doc in enumerate(displayed_docs, 1)}
    
    for index in indices:
        if index in current_products:
            product_id = current_products[index]
            if select:
                if product_id not in selected_product_ids:
                    selected_product_ids.append(product_id)
            else:
                if product_id in selected_product_ids:
                    selected_product_ids.remove(product_id)
    
    cl.user_session.set("selected_product_ids", selected_product_ids)
    logger.debug(f"Updated selected product IDs: {selected_product_ids}")

def compare_products(selected_product_ids: List[str], documents: List[Dict[str, Any]]) -> str:
    """Generate a detailed comparison table for selected products."""
    if len(selected_product_ids) < 2:
        return "Please select at least two products to compare. Type 'select <number>' (e.g., 'select 1') to choose products, then 'compare'. 😊"
    
    selected_docs = []
    for doc in documents:
        if doc["metadata"]["product_id"] in selected_product_ids:
            selected_docs.append(doc)
    
    if len(selected_docs) != len(selected_product_ids):
        return "Some selected products could not be found. Please try again. 😊"
    
    # Define attributes to compare based on category
    category = selected_docs[0]["metadata"]["category"].lower()
    if category == "ac":
        attributes = [
            "Title", "Price", "Size", "Energy Rating", "Smart Features",
            "Air Purification", "Dehumidification", "Swing", "Warranty", "Product URL"
        ]
    elif category in ["tv", "televisions"]:
        attributes = [
            "Title", "Price", "Size", "Energy Rating", "Smart Features",
            "Resolution", "HDR Support", "Key Features", "Product URL"
        ]
    elif category == "audio":
        attributes = [
            "Title", "Price", "Type", "Connectivity", "Portability",
            "Key Features", "Warranty", "Product URL"
        ]
    else:
        attributes = [
            "Title", "Price", "Category", "Size", "Energy Rating",
            "Smart Features", "Key Features", "Product URL"
        ]
    
    table = "| Attribute | " + " | ".join([doc["metadata"]["title"] for doc in selected_docs]) + " |\n"
    table += "|---|" + "---|" * len(selected_docs) + "\n"
    
    for attr in attributes:
        row = f"| **{attr}** |"
        for doc in selected_docs:
            metadata = doc["metadata"]
            content = doc["content"]
            value = ""
            
            if attr == "Title":
                value = metadata.get("title", "N/A")
            elif attr == "Price":
                value = metadata.get("price", "N/A")
            elif attr == "Category":
                value = metadata.get("category", "N/A")
            elif attr == "Size":
                value = metadata.get("size", "N/A")
                if metadata.get("category", "").lower() in ["tv", "televisions"]:
                    size_match = re.search(r'(\d+)\s*(?:cm|inches|\")', metadata.get("title", "").lower())
                    if size_match:
                        size_value = size_match.group(1)
                        unit = "cm" if "cm" in metadata.get("title", "").lower() else "inches"
                        value = f"{size_value} {unit}"
            elif attr == "Energy Rating":
                value = metadata.get("energy_rating", "N/A")
            elif attr == "Smart Features":
                value = metadata.get("smart_features", "N/A")
            elif attr == "Key Features":
                if "Key Features:" in content:
                    features_text = content.split("Key Features:")[1].split("\n")[0]
                    value = features_text.strip()
                else:
                    value = "N/A"
            elif attr == "Product URL":
                url = metadata.get("product_url", "")
                value = f"[Link]({url})" if url.startswith("http") else "N/A"
            elif attr == "Warranty":
                value = metadata.get("warranty", "N/A")
            elif attr == "Air Purification":
                if "pm 0.1" in content.lower() or "nanoe" in content.lower() or "air purification" in content.lower():
                    value = "Yes (PM 0.1 Filter, Nanoe-G)" if "nanoe" in content.lower() else "Yes (PM 0.1 Filter)"
                    if "ag clean" in content.lower():
                        value += ", Ag Clean Plus"
                else:
                    value = "No"
            elif attr == "Dehumidification":
                if "humidity control" in content.lower() or "monsoon" in content.lower():
                    value = "Excellent"
                elif "nanoe" in content.lower() or "pm 0.1" in content.lower():
                    value = "Very Effective"
                else:
                    value = "Good"
            elif attr == "Swing":
                if "4 way" in content.lower():
                    value = "4-Way"
                elif "2 way" in content.lower():
                    value = "2-Way"
                else:
                    value = "N/A"
            elif attr == "Resolution":
                if "4k" in content.lower() or "ultra hd" in content.lower():
                    value = "4K Ultra HD"
                elif "full hd" in content.lower():
                    value = "Full HD"
                elif "hd ready" in content.lower():
                    value = "HD Ready"
                else:
                    value = "N/A"
            elif attr == "HDR Support":
                if "hdr10+" in content.lower():
                    value = "HDR10+"
                elif "dolby vision" in content.lower():
                    value = "Dolby Vision"
                elif "hdr" in content.lower():
                    value = "HDR"
                else:
                    value = "No"
            elif attr == "Type":
                value = metadata.get("type", "N/A")
            elif attr == "Connectivity":
                connectivity = []
                if "bluetooth" in content.lower():
                    connectivity.append("Bluetooth")
                if "usb" in content.lower():
                    connectivity.append("USB")
                if "aux" in content.lower():
                    connectivity.append("AUX")
                value = ", ".join(connectivity) if connectivity else "N/A"
            elif attr == "Portability":
                if "portable" in content.lower() or "wireless" in content.lower() or "party speaker" in metadata.get("title", "").lower():
                    value = "Yes"
                else:
                    value = "No"
            
            row += f" {value} |"
        table += row + "\n"
    
    return f"### Product Comparison\n\n{table}\n\nWould you like more details about any of these products? 😊"

@cl.on_chat_start
async def on_chat_start():
    """Initialize the retrieval chain when the chat starts."""
    logger.info("Chat session started")
    start_time = time.perf_counter()
    
    documents = []
    
    if os.path.exists(CSV_FILE_PATH):
        csv_docs = process_csv(CSV_FILE_PATH)
        documents.extend(csv_docs)
    
    if not documents:
        logger.warning("No documents found to process")
        await cl.Message(content="No documents found to process. Please check the file paths.").send()
        return
    
    chain = setup_retrieval_chain(documents)
    
    if chain:
        cl.user_session.set("chain", chain)
        cl.user_session.set("chat_history", [])
        cl.user_session.set("documents", documents)
        cl.user_session.set("selected_product_ids", [])
        cl.user_session.set("displayed_docs", [])
        cl.user_session.set("query_state", {})
        end_time = time.perf_counter()
        logger.info(f"Chat initialization completed in {end_time - start_time:.2f} seconds")
        await cl.Message(content="Welcome to the Panasonic Product Assistant! 😊 How can I help you find the perfect product today?").send()
    else:
        logger.error("Failed to initialize retrieval chain")
        await cl.Message(content="Failed to initialize the system. Please try again later.").send()

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages and stream responses with conversational flow."""
    chain = cl.user_session.get("chain")
    documents = cl.user_session.get("documents")
    
    if not chain or not documents:
        logger.warning("Retrieval chain or documents not initialized")
        await cl.Message(content="System is still initializing. Please wait a moment and try again.").send()
        return
    
    start_time = time.perf_counter()
    query = message.content.strip()
    logger.info(f"Processing query: {query}")
    
    msg = cl.Message(content="")
    await msg.send()
    
    query_state = cl.user_session.get("query_state", {})
    
    if is_greeting_query(query):
        reasoning = "🤔 **Reasoning**\nI recognized your query as a greeting ('hi', 'hello', etc.), so I'm providing a friendly welcome message to assist you with Panasonic products.\n\n"
        response = "Hello! 😊 How can I assist you with Panasonic products today?"
        
        for char in reasoning:
            await msg.stream_token(char)
        for char in response:
            await msg.stream_token(char)
        
        end_time = time.perf_counter()
        response_time = end_time - start_time
        response_time_text = f"\n\n⏱️ **Response Time**: {response_time:.2f} seconds"
        for char in response_time_text:
            await msg.stream_token(char)
        logger.info(f"Total response time for greeting query '{query}': {response_time:.2f} seconds")
        
        await msg.update()
        
        chat_history = cl.user_session.get("chat_history", [])
        chat_history.append((query, reasoning + response + response_time_text))
        cl.user_session.set("chat_history", chat_history)
        return
    
    if is_select_query(query):
        reasoning = "🤔 **Reasoning**\nI interpreted your query as a selection command ('select <number>'). I'll map the numbers to the currently displayed products and update your selection for comparison.\n\n"
        
        for char in reasoning:
            await msg.stream_token(char)
        
        indices_str = query.lower().replace("select ", "").strip()
        displayed_docs = cl.user_session.get("displayed_docs", [])
        if not displayed_docs:
            response = "No products are currently displayed to select from. Please search for products first (e.g., 'suggest tv under 60k'). 😊"
        else:
            try:
                indices = [int(x) for x in indices_str.split() if x.isdigit()]
                if not indices:
                    raise ValueError
                max_index = len(displayed_docs)
                invalid_indices = [i for i in indices if i < 1 or i > max_index]
                if invalid_indices:
                    response = f"Invalid selection: {', '.join(map(str, invalid_indices))}. Please use numbers between 1 and {max_index}. 😊"
                else:
                    update_selected_products(indices, displayed_docs, True)
                    selected_product_ids = cl.user_session.get("selected_product_ids", [])
                    current_products = {i: doc.metadata["title"] for i, doc in enumerate(displayed_docs, 1)}
                    selected_titles = [current_products[i] for i in indices if i in current_products]
                    response = f"Selected products: {', '.join(selected_titles)}. Currently selected Product IDs: {', '.join(selected_product_ids)}. Type 'compare' to compare them, or 'select <number>' to add more. 😊"
            except ValueError:
                response = "Invalid selection. Please use 'select 1' or 'select 1 2' with numbers from the list above. 😊"
        
        for char in response:
            await msg.stream_token(char)
        
        end_time = time.perf_counter()
        response_time = end_time - start_time
        response_time_text = f"\n\n⏱️ **Response Time**: {response_time:.2f} seconds"
        for char in response_time_text:
            await msg.stream_token(char)
        
        logger.info(f"Total response time for selection query '{query}': {response_time:.2f} seconds")
        await msg.update()
        
        chat_history = cl.user_session.get("chat_history", [])
        chat_history.append((query, reasoning + response + response_time_text))
        cl.user_session.set("chat_history", chat_history)
        return
    
    if is_compare_query(query):
        reasoning = "🤔 **Reasoning**\nI recognized your query as a comparison request ('compare'). I'll retrieve the products you previously selected and create a detailed comparison table based on their attributes, tailored to the product category for better relevance.\n\n"
        
        for char in reasoning:
            await msg.stream_token(char)
        
        selected_product_ids = cl.user_session.get("selected_product_ids", [])
        comparison_response = compare_products(selected_product_ids, documents)
        for char in comparison_response:
            await msg.stream_token(char)
        
        end_time = time.perf_counter()
        response_time = end_time - start_time
        response_time_text = f"\n\n⏱️ **Response Time**: {response_time:.2f} seconds"
        for char in response_time_text:
            await msg.stream_token(char)
        logger.info(f"Total response time for comparison query '{query}': {response_time:.2f} seconds")
        
        await msg.update()
        
        chat_history = cl.user_session.get("chat_history", [])
        chat_history.append((query, reasoning + comparison_response + response_time_text))
        cl.user_session.set("chat_history", chat_history)
        return
    
    if not query_state.get("awaiting_response"):
        category = extract_category(query)
        features = extract_features(query, category or "")
        query_state["category"] = category
        query_state["features"] = features
        query_state["original_query"] = query
    else:
        category = query_state.get("category")
        features = query_state.get("features", [])

    if query_state.get("awaiting_response"):
        step = query_state.get("step")
        if step == "room_size":
            room_size = extract_room_size(query)
            if room_size:
                query_state["room_size_sqft"] = room_size
                query_state["step"] = "location"
                response = f"Got it! Your room size is {room_size} sq ft. 📏 Now, could you please tell me your location (e.g., Delhi, Mumbai)? 🏙️"
            else:
                response = "I couldn't understand the room size. Could you please specify it in square feet (e.g., '2000 sq ft' or '2000')? 📏"
            for char in response:
                await msg.stream_token(char)
            cl.user_session.set("query_state", query_state)
            await msg.update()
            return
        elif step == "location":
            location, pollution_concern = extract_location(query)  # Unpack the tuple
            if location:  # Check if location is not None
                query_state["location"] = location
                query_state["pollution_concern"] = pollution_concern  # Store pollution concern
                query_state["step"] = "done"
                response = f"Thanks! I recognized your location as {location.capitalize()}. 🏙️ Done? (Say 'yes' to see the products) ✅"
            else:
                response = "I couldn't recognize the location. Could you please specify a city like Delhi or Mumbai? 🏙️"
            for char in response:
                await msg.stream_token(char)
            cl.user_session.set("query_state", query_state)
            await msg.update()
            return
        elif step == "done":
            if is_done_response(query):
                query_state["awaiting_response"] = False
                query_state["step"] = None
            else:
                response = "Please confirm if you're done by saying 'yes'. ✅"
                for char in response:
                    await msg.stream_token(char)
                await msg.update()
                return
    
    is_price_based = is_price_query(query_state.get("original_query", query))
    room_size_intent = None
    location_intent = query_state.get("location")  # Retrieve from query_state
    pollution_concern = query_state.get("pollution_concern", False)  # Retrieve from query_state
    room_size_sqft = query_state.get("room_size_sqft")

    query_lower = query.lower()
    if any(word in query_lower for word in ["big room", "large room", "spacious room", "big hall"]):
        room_size_intent = "large"
    elif any(word in query_lower for word in ["small room", "compact room", "tiny room"]):
        room_size_intent = "small"
    
    sq_ft_match = re.search(r'(\d+)\s*(sq\s*ft|square\s*feet|sft|sqft|ft²)', query_lower)
    if sq_ft_match:
        room_size_sqft = int(sq_ft_match.group(1))
        query_state["room_size_sqft"] = room_size_sqft
    
    # Removed redundant extract_location call
    logger.info(f"Using stored location: {location_intent}, Pollution concern (determined by LLM): {pollution_concern}")

    if category == "AC" and (room_size_sqft is None or location_intent is None):
        query_state["awaiting_response"] = True
        if room_size_sqft is None:
            query_state["step"] = "room_size"
            response = "To recommend the best AC, I need some details. Could you please tell me the size of your room in square feet (e.g., '2000 sq ft')? 📏"
        else:
            query_state["step"] = "location"
            response = f"Got it! Your room size is {room_size_sqft} sq ft. 📏 Now, could you please tell me your location (e.g., Delhi, Mumbai)? 🏙️"
        
        for char in response:
            await msg.stream_token(char)
        cl.user_session.set("query_state", query_state)
        await msg.update()
        return
    
    if room_size_sqft and category == "AC":
        if room_size_sqft <= 120:
            required_tonnage = "1 ton"
        elif room_size_sqft <= 180:
            required_tonnage = "1.5 ton"
        elif room_size_sqft <= 240:
            required_tonnage = "2 ton"
        else:
            required_tonnage = "2 ton or higher"
        if required_tonnage not in features:
            features.append(required_tonnage)

    original_query = query_state.get("original_query", query).lower()
    if category == "AC" and room_size_sqft and location_intent:
        enhanced_query = f"Panasonic AC"
        if features:
            enhanced_query += f" with features {', '.join(features)}"
        if room_size_sqft:
            enhanced_query += f" suitable for {room_size_sqft} sq ft room"
        if location_intent:
            enhanced_query += f" suitable for use in {location_intent}"
        if pollution_concern:
            enhanced_query += f" with air purification features for polluted areas"
        enhanced_query += f" hot search keywords: {original_query}"
        logger.info(f"Enhanced query: {enhanced_query}")
    elif is_price_based:
        min_price, max_price = extract_price_range(query_state.get("original_query", query))
        enhanced_query = f"Panasonic {category or ''}"
        if category:
            enhanced_query += f" in category {category}"
        if features:
            enhanced_query += f" with features {', '.join(features)}"
        if room_size_intent:
            enhanced_query += f" suitable for a {room_size_intent} room"
        if room_size_sqft:
            enhanced_query += f" suitable for {room_size_sqft} sq ft room"
        if location_intent:
            enhanced_query += f" suitable for use in {location_intent}"
        if pollution_concern:
            enhanced_query += f" with air purification features for polluted areas"
        enhanced_query += f" hot search keywords: {original_query}"
        logger.info(f"Enhanced query: {enhanced_query}")
    else:
        enhanced_query = f"Panasonic {category or ''}"
        if category:
            enhanced_query += f" in category {category}"
        if room_size_intent:
            enhanced_query += f" suitable for a {room_size_intent} room"
        if room_size_sqft:
            enhanced_query += f" suitable for {room_size_sqft} sq ft room"
        if location_intent:
            enhanced_query += f" suitable for use in {location_intent}"
        if pollution_concern:
            enhanced_query += f" with air purification features for polluted areas"
        enhanced_query += f" hot search keywords: {original_query}"
        logger.info(f"Enhanced query: {enhanced_query}")
    
    if enhanced_query in query_cache:
        logger.info("Serving response from cache")
        response, source_documents = query_cache[enhanced_query]
        reasoning = "🤔 **Reasoning**\nI found your enhanced query in the cache, meaning I've answered this before. I'll provide the same response to save time.\n\n"
        
        for char in reasoning:
            await msg.stream_token(char)
        for char in response:
            await msg.stream_token(char)
        
        end_time = time.perf_counter()
        response_time = end_time - start_time
        response_time_text = f"\n\n⏱️ **Response Time**: {response_time:.2f} seconds (cached)"
        for char in response_time_text:
            await msg.stream_token(char)
        logger.info(f"Total response time for query '{query}': {response_time:.2f} seconds")
        
        await msg.update()
        return
    
    cb = cl.AsyncLangchainCallbackHandler()
    
    search_kwargs = {"k": K_RETRIEVAL}
    if category:
        if category == "TV":
            search_kwargs["filter"] = {"category": {"$in": ["TV", "Televisions"]}}
        elif category == "AC":
            search_kwargs["filter"] = {"category": "AC"}
        elif category == "Microwave":
            search_kwargs["filter"] = {"category": "Microwave Ovens"}
        elif category == "Audio System":
            search_kwargs["filter"] = {"category": "Audio"}
    
    chain.retriever.search_kwargs = search_kwargs
    
    query_start = time.perf_counter()
    res = await chain.ainvoke(
        {"question": enhanced_query, "chat_history": cl.user_session.get("chat_history", [])},
        callbacks=[cb]
    )
    query_end = time.perf_counter()
    logger.info(f"Query processing completed in {query_end - query_start:.2f} seconds")
    
    source_documents = res["source_documents"]
    logger.info(f"Retrieved {len(source_documents)} documents from vector store")
    for i, doc in enumerate(source_documents):
        logger.info(f"Document {i+1}: Category={doc.metadata.get('category', 'N/A')}, Title={doc.metadata.get('title', 'N/A')}")
    
    full_answer = res["answer"]
    reasoning = ""
    answer = full_answer
    
    if "**Reasoning**" in full_answer and "**Answer**" in full_answer:
        reasoning = full_answer.split("**Reasoning**")[1].split("**Answer**")[0].strip()
        reasoning = f"🤔 **Reasoning**\n{reasoning}\n\n"
        answer = full_answer.split("**Answer**")[1].strip()
    else:
        reasoning_parts = []
        if category:
            reasoning_parts.append(f"You’re looking for a Panasonic {category} to suit your needs.")
        else:
            reasoning_parts.append(f"You’re looking for a Panasonic product to suit your needs.")
        
        if room_size_sqft and category == "AC":
            required_tonnage_display = "2 Ton or higher" if room_size_sqft > 240 else "1.5 Ton" if room_size_sqft > 120 else "1 Ton"
            reasoning_parts.append(f"Since you specified a {room_size_sqft} sq ft room, a single {required_tonnage_display} AC is insufficient. For effective cooling, you may need 5-6 tons total (1 ton per 350-400 sq ft), which could mean multiple 2-ton units or a higher-capacity system. I recommend consulting a professional for a proper setup.")
        elif room_size_intent == "large":
            reasoning_parts.append("Since you mentioned a big room, I focused on products suitable for larger spaces.")
        elif room_size_intent == "small":
            reasoning_parts.append("Since you mentioned a small room, I looked for products suitable for compact spaces.")
        
        if location_intent and category == "AC":
            if location_intent.lower() == "bangalore":
                reasoning_parts.append(f"For Bengaluru, with its moderate temperatures (20°C-35°C) and high humidity during monsoons, I prioritized ACs with strong dehumidification, energy efficiency, and air purification to handle occasional urban pollution.")
            else:
                reasoning_parts.append(f"For {location_intent.capitalize()}, where the climate can be hot and humid, I prioritized ACs with strong cooling capabilities and high ambient temperature operation (up to 55°C).")
            if pollution_concern:
                reasoning_parts.append(f"Given {location_intent.capitalize()}'s high pollution levels, I also considered models with advanced air purification features like PM 0.1 filters or nanoe-G technology.")
            else:
                reasoning_parts.append(f"I couldn’t determine the pollution level for {location_intent.capitalize()}. If air quality is a concern, look for ACs with PM 0.1 filters or nanoe-G technology.")
        
        if features:
            feature_descriptions = []
            for feature in features:
                if feature == "inverter":
                    feature_descriptions.append("inverter technology for energy efficiency")
                elif feature == "smart control":
                    feature_descriptions.append("smart control features like Wi-Fi and voice assistant compatibility")
                elif feature == "air purification":
                    feature_descriptions.append("air purification for cleaner air")
                elif feature == "hot and cold":
                    feature_descriptions.append("hot and cold functionality for all-weather use")
                elif feature == "gaming":
                    feature_descriptions.append("gaming features like low latency and 120Hz refresh rate")
                elif feature == "portable":
                    feature_descriptions.append("portability for easy mobility")
                elif feature == "capacity" and "2 ton" in query_lower:
                    feature_descriptions.append("2-ton capacity as requested")
                elif feature == "dehumidification":
                    feature_descriptions.append("strong dehumidification for humid climates")
            if feature_descriptions:
                reasoning_parts.append(f"Based on your query, I prioritized products with {', '.join(feature_descriptions)}.")
        
        if is_price_based:
            min_price, max_price = extract_price_range(query_state.get("original_query", query))
            if min_price and max_price:
                reasoning_parts.append(f"Your budget is between ₹{min_price} and ₹{max_price}, so I filtered products within this range.")
            elif min_price:
                reasoning_parts.append(f"Your minimum budget is ₹{min_price}, so I looked for products above this price.")
            elif max_price:
                reasoning_parts.append(f"Your maximum budget is ₹{max_price}, so I focused on products below this price.")
        
        if source_documents:
            matching_docs = [doc for doc in source_documents if doc.metadata.get("source") == "product_catalog"]
            if matching_docs:
                doc = matching_docs[0]
                metadata = doc.metadata
                content = doc.page_content.lower()
                reasons = []
                if category == "AC":
                    if "2 ton" in metadata.get("size", "").lower():
                        reasons.append("a 2-ton capacity as requested")
                    if "inverter" in content:
                        reasons.append("inverter technology for energy savings")
                    if "pm 0.1" in content or "nanoe" in content:
                        reasons.append("air purification for better air quality")
                    if "humidity control" in content or "monsoon" in content:
                        reasons.append("strong dehumidification for Bengaluru’s climate")
                elif category == "TV":
                    if "game mode" in content or "120hz" in content:
                        reasons.append("gaming features like low latency")
                    if "4k" in content:
                        reasons.append("4K resolution for sharp visuals")
                if reasons:
                    reasoning_parts.append(f"I selected these products because they offer {', '.join(reasons)}.")
            else:
                reasoning_parts.append("I couldn’t find an exact match, so I selected the closest available products based on your criteria.")
        else:
            reasoning_parts.append("I searched the catalog but couldn’t find a perfect match. I’ll suggest the closest alternatives.")
        
        reasoning = f"🤔 **Reasoning**\n{' '.join(reasoning_parts)}\n\n"
    
    for char in reasoning:
        await msg.stream_token(char)
    
    response_parts = []
    
    if is_price_based or (category == "AC" and room_size_sqft and location_intent):
        min_price, max_price = extract_price_range(query_state.get("original_query", query))
        if max_price is None:
            max_price = 60000
        product_texts = []
        displayed_docs = []
        
        category_docs = [(doc, 0) for doc in source_documents if doc.metadata.get("source") == "product_catalog"]
        logger.info(f"After category filter (via retriever): {len(category_docs)} documents")
        
        price_filtered_docs = []
        for doc, _ in category_docs:
            price_str = doc.metadata.get("price", "").replace("₹", "").replace(",", "").strip()
            try:
                price = int(float(price_str))
                price_filtered_docs.append((doc, price))
                logger.info(f"Extracted price for {doc.metadata.get('title')}: ₹{price}")
            except (ValueError, TypeError):
                logger.warning(f"Invalid price format for {doc.metadata.get('title')}: {price_str}")
                continue
        logger.info(f"After price extraction: {len(price_filtered_docs)} documents with valid prices")
        
        within_budget_docs = [(doc, price) for doc, price in price_filtered_docs if (min_price is None or price >= min_price) and (max_price is None or price <= max_price)]
        
        if not within_budget_docs:
            above_budget_docs = [(doc, price) for doc, price in price_filtered_docs if min_price is None or price >= min_price]
            above_budget_docs.sort(key=lambda x: x[1])
            filtered_docs = above_budget_docs[:MAX_PRODUCTS_TO_SHOW]
        else:
            filtered_docs = within_budget_docs
        logger.info(f"After price filter (within ₹{min_price}-₹{max_price} or closest above): {len(filtered_docs)} documents")
        
        if category == "AC" and (room_size_intent == "large" or room_size_sqft):
            required_tonnage_filter = "2 ton" if room_size_intent == "large" or (room_size_sqft and room_size_sqft > 180) else "1.5 ton" if room_size_sqft > 120 else "1 ton"
            tonnage_filtered_docs = []
            for doc, price in filtered_docs:
                size = doc.metadata.get("size", "").lower().replace(" ", "")
                if required_tonnage_filter.replace(" ", "") in size or "2tonorhigher" in size:
                    tonnage_filtered_docs.append((doc, price))
            filtered_docs = tonnage_filtered_docs if tonnage_filtered_docs else filtered_docs
            logger.info(f"After tonnage filter (required: {required_tonnage_filter}): {len(filtered_docs)} documents")
        
        if category == "AC" and pollution_concern:
            purification_filtered_docs = []
            for doc, price in filtered_docs:
                content = doc.page_content.lower()
                if "pm 0.1" in content or "nanoe" in content or "air purification" in content:
                    purification_filtered_docs.append((doc, price))
            filtered_docs = purification_filtered_docs if purification_filtered_docs else filtered_docs
            logger.info(f"After pollution filter: {len(filtered_docs)} documents")
        
        if category == "AC" and location_intent == "bangalore":
            dehumidification_filtered_docs = []
            for doc, price in filtered_docs:
                content = doc.page_content.lower()
                if "humidity control" in content or "monsoon" in content or "nanoe" in content or "pm 0.1" in content:
                    dehumidification_filtered_docs.append((doc, price))
            filtered_docs = dehumidification_filtered_docs if dehumidification_filtered_docs else filtered_docs
            logger.info(f"After dehumidification filter for Bengaluru: {len(filtered_docs)} documents")
        
        if category == "AC":
            climate_filtered_docs = []
            for doc, price in filtered_docs:
                content = doc.page_content.lower()
                if "55 deg" in content or "high ambient" in content or "extreme heat" in content:
                    climate_filtered_docs.append((doc, price))
            filtered_docs = climate_filtered_docs if climate_filtered_docs else filtered_docs
            logger.info(f"After climate filter: {len(filtered_docs)} documents")
        
        if category == "AC" and not filtered_docs:
            tonnage_filtered_docs = []
            for doc, price in price_filtered_docs:
                size = doc.metadata.get("size", "").lower().replace(" ", "")
                if required_tonnage_filter.replace(" ", "") in size or "2tonorhigher" in size:
                    tonnage_filtered_docs.append((doc, price))
            filtered_docs = tonnage_filtered_docs
            logger.info(f"Relaxed climate filter, reverted to tonnage filter: {len(filtered_docs)} documents")
        
        if category == "Audio System" and "portable" in features:
            portable_filtered_docs = []
            for doc, price in filtered_docs:
                content = doc.page_content.lower()
                title = doc.metadata.get("title", "").lower()
                if ("bluetooth" in content or "wireless" in content or "party speaker" in title) and not ("soundbar" in title or "5.1ch" in title):
                    portable_filtered_docs.append((doc, price))
            filtered_docs = portable_filtered_docs if portable_filtered_docs else filtered_docs
            logger.info(f"After portable filter: {len(filtered_docs)} documents")
        
        if category == "TV" and "gaming" in features:
            gaming_filtered_docs = []
            for doc, price in filtered_docs:
                content = doc.page_content.lower()
                title = doc.metadata.get("title", "").lower()
                if "game mode" in title or "low latency" in content or "120hz" in content:
                    gaming_filtered_docs.append((doc, price))
            filtered_docs = gaming_filtered_docs if gaming_filtered_docs else filtered_docs
            logger.info(f"After gaming filter: {len(filtered_docs)} documents")
        
        filtered_docs.sort(key=lambda x: x[1])
        filtered_docs = filtered_docs[:MAX_PRODUCTS_TO_SHOW]
        logger.info(f"Final filtered documents for display: {len(filtered_docs)} documents")
        for i, (doc, price) in enumerate(filtered_docs, 1):
            logger.info(f"Final doc {i}: Title={doc.metadata.get('title')}, Price={price}, Size={doc.metadata.get('size')}")
        
        total_matching_products = len(filtered_docs)
        
        for index, (source_doc, price) in enumerate(filtered_docs, 1):
            product_info = await format_product_info(source_doc, msg, product_index=index)
            product_texts.append(product_info)
            displayed_docs.append(source_doc)
        
        cl.user_session.set("displayed_docs", displayed_docs)
        
        if product_texts:
            products_count = len(product_texts)
            
            if min_price and max_price:
                if all(price <= max_price for _, price in filtered_docs):
                    price_range_text = f"Here are {products_count} Panasonic {category} products between ₹{min_price} and ₹{max_price}:\n\n"
                else:
                    price_range_text = f"No exact matches found between ₹{min_price} and ₹{max_price}. Here are {products_count} Panasonic {category} products slightly above your budget that meet your other criteria:\n\n"
            elif min_price:
                price_range_text = f"Here are {products_count} Panasonic {category} products above ₹{min_price}:\n\n"
            elif max_price:
                if all(price <= max_price for _, price in filtered_docs):
                    price_range_text = f"Here are {products_count} Panasonic {category} products under ₹{max_price}:\n\n"
                else:
                    price_range_text = f"No exact matches found under ₹{max_price}. Here are {products_count} Panasonic {category} products slightly above your budget:\n\n"
            else:
                price_range_text = f"Here are {products_count} Panasonic {category} products matching your criteria:\n\n"
            
            response_parts.append(price_range_text + "\n".join(product_texts))
            
            instruction_text = "\n\nTo compare products, type 'select <number>' (e.g., 'select 1') or 'select 1 2' for multiple, then type 'compare'. 😊\n"
            selected_product_ids = cl.user_session.get("selected_product_ids", [])
            if selected_product_ids:
                instruction_text += f"Currently selected Product IDs: {', '.join(selected_product_ids)}\n"
            response_parts.append(instruction_text)
            for char in instruction_text:
                await msg.stream_token(char)
            
            if category and products_count > 0:
                if category.lower() == "ac":
                    tip = "\n\n💡 **Tip**: Choose an AC based on room size (e.g., 1.5 ton for 150-190 sq.ft.) and look for inverter models for energy savings."
                elif category.lower() == "tv":
                    tip = "\n\n💡 **Tip**: For optimal viewing, choose a screen size where viewing distance is 1.5-2.5 times the diagonal size (e.g., 43 inches for 5-7 feet)."
                elif category.lower() == "refrigerator":
                    tip = "\n\n💡 **Tip**: A 250-300L fridge suits a family of 2-3, while 350-450L is ideal for 4-5 members."
                elif category.lower() == "cameras":
                    tip = "\n\n💡 **Tip**: Consider full-frame mirrorless cameras for professional photography or compact lenses for travel."
                elif category.lower() == "microwave":
                    tip = "\n\n💡 **Tip**: Convection microwaves are great for baking and grilling, while solo models are best for basic reheating."
                elif category.lower() == "audio system":
                    tip = "\n\n💡 **Tip**: Look for Bluetooth-enabled portable speakers for easy mobility, or soundbars for enhanced home theater audio."
                elif category.lower() == "small appliances":
                    tip = "\n\n💡 **Tip**: Look for cordless and waterproof options for convenience in grooming or cleaning appliances."
                else:
                    tip = ""
                response_parts.append(tip)
                for char in tip:
                    await msg.stream_token(char)
            
            if total_matching_products > products_count:
                more_products_text = f"\n\nThere are {total_matching_products - products_count} more products matching your criteria. Would you like to refine your search?"
                response_parts.append(more_products_text)
                for char in more_products_text:
                    await msg.stream_token(char)
            
            follow_up_text = "\n\nIs there anything specific you'd like to know about any of these products?"
            response_parts.append(follow_up_text)
            for char in follow_up_text:
                await msg.stream_token(char)
        else:
            category_docs = []
            price_ranges = []
            for doc in documents:
                if category == "TV" and doc["metadata"]["source"] == "product_catalog" and doc["metadata"]["category"].lower() in ["tv", "televisions"]:
                    category_docs.append(doc)
                elif category == "AC" and doc["metadata"]["source"] == "product_catalog" and doc["metadata"]["category"].lower() == "ac":
                    category_docs.append(doc)
                elif category == "Microwave" and doc["metadata"]["source"] == "product_catalog" and doc["metadata"]["category"].lower() == "microwave ovens":
                    category_docs.append(doc)
                elif category == "Audio System" and doc["metadata"]["source"] == "product_catalog" and doc["metadata"]["category"].lower() == "audio":
                    category_docs.append(doc)
                price_str = doc["metadata"].get("price", "").replace("₹", "").replace(",", "").strip()
                try:
                    price = int(float(price_str))
                    price_ranges.append(price)
                except (ValueError, TypeError):
                    continue
            
            no_products_text = f"No Panasonic {category} products found for your criteria."
            if category == "AC" and room_size_sqft and room_size_sqft > 240:
                no_products_text += f" For a {room_size_sqft} sq ft room, you need at least a 2 Ton AC, which typically costs above ₹50,000 due to higher capacity and advanced features."
            if category_docs:
                if price_ranges:
                    min_available_price = min(price_ranges)
                    max_available_price = max(price_ranges)
                    no_products_text += f" However, I found Panasonic {category} products in the price range of ₹{min_available_price} to ₹{max_available_price}. Would you like to try this range instead? 😊"
                else:
                    no_products_text += f" I found Panasonic {category} products, but their prices are not available. Would you like to try a different category? 😊"
            else:
                no_products_text += f" I couldn't find any Panasonic {category} products in the catalog. Would you like to try a different category? 😊"
            
            response_parts.append(no_products_text)
            for char in no_products_text:
                await msg.stream_token(char)
    else:
        product_texts = []
        displayed_docs = []
        
        if source_documents:
            product_count = 0
            for source_idx, source_doc in enumerate(source_documents):
                if source_doc.metadata.get("source") == "product_catalog":
                    if category == "AC" and (room_size_intent == "large" or room_size_sqft):
                        required_tonnage_filter = "2 ton" if room_size_intent == "large" or (room_size_sqft and room_size_sqft > 180) else "1.5 ton" if room_size_sqft > 120 else "1 ton"
                        size = source_doc.metadata.get("size", "").lower().replace(" ", "")
                        if required_tonnage_filter.replace(" ", "") not in size and "2tonorhigher" not in size:
                            continue
                    
                    if category == "AC" and pollution_concern:
                        content = source_doc.page_content.lower()
                        if not ("pm 0.1" in content or "nanoe" in content or "air purification" in content):
                            continue
                    
                    if category == "AC" and location_intent == "bangalore":
                        content = source_doc.page_content.lower()
                        if not ("humidity control" in content or "monsoon" in content or "nanoe" in content or "pm 0.1" in content):
                            continue
                    
                    if category == "Audio System" and "portable" in features:
                        content = source_doc.page_content.lower()
                        title = source_doc.metadata.get("title", "").lower()
                        if not ("bluetooth" in content or "wireless" in content or "party speaker" in title) or ("soundbar" in title or "5.1ch" in title):
                            continue
                    
                    if category == "TV" and "gaming" in features:
                        content = source_doc.page_content.lower()
                        title = source_doc.metadata.get("title", "").lower()
                        if not ("game mode" in title or "low latency" in content or "120hz" in content):
                            continue
                    
                    if product_count < MAX_PRODUCTS_TO_SHOW:
                        product_info = await format_product_info(source_doc, msg, product_index=product_count + 1)
                        product_texts.append(product_info)
                        displayed_docs.append(source_doc)
                        product_count += 1
        
        cl.user_session.set("displayed_docs", displayed_docs)
        
        if product_texts:
            intro_text = f"Here are {len(product_texts)} Panasonic {category} products that might interest you:\n\n"
            response_parts.append(intro_text + "\n".join(product_texts))
            
            instruction_text = "\n\nTo compare products, type 'select <number>' (e.g., 'select 1') or 'select 1 2' for multiple, then type 'compare'. 😊\n"
            selected_product_ids = cl.user_session.get("selected_product_ids", [])
            if selected_product_ids:
                instruction_text += f"Currently selected Product IDs: {', '.join(selected_product_ids)}\n"
            response_parts.append(instruction_text)
            for char in instruction_text:
                await msg.stream_token(char)
            
            if category == "AC":
                tip = "\n\n💡 **Tip**: Choose an AC based on room size (e.g., 1.5 ton for 150-190 sq.ft.) and look for inverter models for energy savings."
                response_parts.append(tip)
                for char in tip:
                    await msg.stream_token(char)
            elif category == "TV":
                tip = "\n\n💡 **Tip**: For optimal viewing, choose a screen size where viewing distance is 1.5-2.5 times the diagonal size (e.g., 43 inches for 5-7 feet)."
                response_parts.append(tip)
                for char in tip:
                    await msg.stream_token(char)
            elif category == "Microwave":
                tip = "\n\n💡 **Tip**: Convection microwaves are great for baking and grilling, while solo models are best for basic reheating."
                response_parts.append(tip)
                for char in tip:
                    await msg.stream_token(char)
            elif category == "Audio System":
                tip = "\n\n💡 **Tip**: Look for Bluetooth-enabled portable speakers for easy mobility, or soundbars for enhanced home theater audio."
                response_parts.append(tip)
                for char in tip:
                    await msg.stream_token(char)
            
            follow_up_text = "\n\nWould you like to refine your search with a price range or specific features?"
            response_parts.append(follow_up_text)
            for char in follow_up_text:
                await msg.stream_token(char)
        else:
            category_docs = []
            if category:
                for doc in documents:
                    if category == "TV" and doc["metadata"]["source"] == "product_catalog" and doc["metadata"]["category"].lower() in ["tv", "televisions"]:
                        category_docs.append(doc)
                    if category == "AC" and doc["metadata"]["source"] == "product_catalog" and doc["metadata"]["category"].lower() == "ac":
                        category_docs.append(doc)
                    if category == "Microwave" and doc["metadata"]["source"] == "product_catalog" and doc["metadata"]["category"].lower() == "microwave ovens":
                        category_docs.append(doc)
                    if category == "Audio System" and doc["metadata"]["source"] == "product_catalog" and doc["metadata"]["category"].lower() == "audio":
                        category_docs.append(doc)
            
            no_products_text = f"No Panasonic {category} products found for your query."
            if category == "AC" and room_size_sqft and room_size_sqft > 240:
                no_products_text += f" For a {room_size_sqft} sq ft room, you need at least a 2 Ton AC, which typically costs above ₹50,000 due to higher capacity and advanced features."
            if category_docs:
                no_products_text += f" However, I found some Panasonic {category} products. Would you like to specify a price range or additional features? 😊"
            else:
                no_products_text += f" I couldn't find any Panasonic {category} products in the catalog. Would you like to try a different category? 😊"
            
            response_parts.append(no_products_text)
            for char in no_products_text:
                await msg.stream_token(char)
    
    query_cache[enhanced_query] = ("".join(response_parts), source_documents)
    
    end_time = time.perf_counter()
    response_time = end_time - start_time
    response_time_text = f"\n\n⏱️ **Response Time**: {response_time:.2f} seconds"
    for char in response_time_text:
        await msg.stream_token(char)
    logger.info(f"Total response time for query '{query}': {response_time:.2f} seconds")
    
    await msg.update()
    
    chat_history = cl.user_session.get("chat_history", [])
    chat_history.append((query, "".join(response_parts) + response_time_text))
    cl.user_session.set("chat_history", chat_history)
    cl.user_session.set("query_state", query_state)
