import logging
from flask import current_app, jsonify, request
import json
import requests
from difflib import get_close_matches
# from app.services.openai_service import generate_response
import re
from openai import OpenAI
import os
import numpy as np
from dotenv import load_dotenv
from openai import AuthenticationError, APIError, RateLimitError
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple
import chromadb
from chromadb.utils import embedding_functions

# Initialize ChromaDB client (will be set up properly in setup function)
chroma_client = None
product_collection = None
load_dotenv()
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Global variables to cache embeddings
PRODUCT_EMBEDDINGS = None
PRODUCT_DATA = None
EMBEDDING_MODEL = "text-embedding-3-small"  # or "text-embedding-ada-002"

def log_http_response(response):
    logging.info(f"Status: {response.status_code}")
    logging.info(f"Content-type: {response.headers.get('content-type')}")
    logging.info(f"Body: {response.text}")

# Stopwords in Spanish + English
STOPWORDS = {
    # Spanish
    "de", "la", "el", "los", "las", "y", "o", "en", "un", "una", "unos", "unas",
    "para", "por", "con", "del", "al", "mi", "tu", "su", "que", "quiero", "dame",
    "me", "necesito", "hola", "buenos", "dias", "tarde", "noche", "gracias",
    # English
    "the", "a", "an", "of", "to", "in", "on", "for", "with", "at", "by", "from",
    "is", "are", "am", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "can", "could", "should", "would", "may", "might",
    "will", "shall", "hi", "hello", "please", "help", "need", "about"
}

def initialize_embeddings():
    """
    Initialize or load product embeddings on startup.
    This should be called when your Flask app starts.
    """
    global PRODUCT_EMBEDDINGS, PRODUCT_DATA
    
    embeddings_file = "data/product_embeddings.pkl"
    
    # Try to load existing embeddings
    if os.path.exists(embeddings_file):
        try:
            with open(embeddings_file, "rb") as f:
                cached_data = pickle.load(f)
                PRODUCT_EMBEDDINGS = cached_data["embeddings"]
                PRODUCT_DATA = cached_data["products"]
                logging.info(f"Loaded {len(PRODUCT_DATA)} product embeddings from cache")
                return
        except Exception as e:
            logging.warning(f"Failed to load cached embeddings: {e}")
    
    # Generate new embeddings
    logging.info("Generating new product embeddings...")
    generate_and_cache_embeddings()

def generate_and_cache_embeddings():
    """
    Generate embeddings for all products and cache them.
    """
    global PRODUCT_EMBEDDINGS, PRODUCT_DATA
    
    try:
        # Load product data
        with open("data/maka_products.json", "r", encoding="utf-8") as f:
            products = json.load(f)
        
        if not products:
            logging.warning("No products found to embed")
            return
        
        client = OpenAI(api_key=current_app.config['OPENAI_API_KEY'])
        
        # Prepare product data and texts for embedding
        product_list = []
        texts_to_embed = []
        
        for product_name, product_data in products.items():
            # Create rich text representation for embedding
            embed_text = create_product_embed_text(product_name, product_data)
            
            product_list.append({
                "name": product_name,
                "data": product_data,
                "embed_text": embed_text
            })
            texts_to_embed.append(embed_text)
        
        # Generate embeddings in batches
        batch_size = 100  # OpenAI's recommended batch size
        all_embeddings = []
        
        for i in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[i:i + batch_size]
            
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
            logging.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts_to_embed) + batch_size - 1)//batch_size}")
        
        # Store in global variables
        PRODUCT_EMBEDDINGS = np.array(all_embeddings)
        PRODUCT_DATA = product_list
        
        # Cache the embeddings
        cache_data = {
            "embeddings": PRODUCT_EMBEDDINGS,
            "products": PRODUCT_DATA,
            "model": EMBEDDING_MODEL,
            "generated_at": np.datetime64('now').astype(str)
        }
        
        with open("data/product_embeddings.pkl", "wb") as f:
            pickle.dump(cache_data, f)
        
        logging.info(f"Generated and cached embeddings for {len(PRODUCT_DATA)} products")
        
    except Exception as e:
        logging.error(f"Failed to generate embeddings: {e}")
        # Fallback to empty data
        PRODUCT_EMBEDDINGS = np.array([])
        PRODUCT_DATA = []
def create_product_embed_text(product_name: str, product_data: Dict) -> str:
    """
    Create a rich text representation of a product for embedding.
    """
    parts = [
        f"Product: {product_name}",
        f"Description: {product_data.get('description', '')}",
        f"Category: {product_data.get('category', '')}",
        f"Features: {', '.join(product_data.get('features', []))}",
        f"Size: {product_data.get('size', '')}",
        f"Price: {product_data.get('price', '')}",
    ]
    
    # Filter out empty parts
    meaningful_parts = [part for part in parts if not part.endswith(': ')]
    return " | ".join(meaningful_parts)

def semantic_search_products(query: str, top_k: int = 5, similarity_threshold: float = 0.3) -> List[Dict]:
    """
    Perform semantic search on products using embeddings.
    
    Args:
        query: User's search query
        top_k: Number of top results to return
        similarity_threshold: Minimum similarity score to include a result
    
    Returns:
        List of matching products with similarity scores
    """
    global PRODUCT_EMBEDDINGS, PRODUCT_DATA
    
    if PRODUCT_EMBEDDINGS is None or len(PRODUCT_EMBEDDINGS) == 0:
        logging.warning("No product embeddings available, falling back to keyword search")
        return keyword_fallback_search(query)
    
    try:
        client = OpenAI(api_key=current_app.config['OPENAI_API_KEY'])
        
        # Get embedding for the query
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query
        )
        query_embedding = np.array(response.data[0].embedding).reshape(1, -1)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, PRODUCT_EMBEDDINGS)[0]
        
        # Get top matches above threshold
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            similarity_score = similarities[idx]
            if similarity_score >= similarity_threshold:
                product_info = PRODUCT_DATA[idx]
                results.append({
                    "product": product_info["data"],
                    "similarity_score": float(similarity_score),
                    "name": product_info["name"]
                })
        
        logging.info(f"Found {len(results)} products with similarity >= {similarity_threshold}")
        return results
        
    except Exception as e:
        logging.error(f"Semantic search failed: {e}")
        return keyword_fallback_search(query)
def keyword_fallback_search(query: str) -> List[Dict]:
    """
    Fallback keyword-based search when embeddings are not available.
    """
    try:
        with open("data/maka_products.json", "r", encoding="utf-8") as f:
            products = json.load(f)
        
        keywords = clean_message(query)
        matched_products = []
        
        for pname, pdata in products.items():
            pname_lower = pname.lower()
            if any(kw in pname_lower for kw in keywords):
                matched_products.append({
                    "product": pdata,
                    "similarity_score": 0.5,  # Default score for keyword matches
                    "name": pname
                })
        
        return matched_products
        
    except Exception as e:
        logging.error(f"Keyword fallback search failed: {e}")
        return []
def setup_llm_config(app):
    """Setup LLM and embedding configuration"""
    # OpenAI Configuration
    app.config['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY')
    app.config['OPENAI_MODEL'] = os.environ.get('OPENAI_MODEL', 'gpt-3.5-turbo')
    
    # Anthropic Configuration  
    app.config['ANTHROPIC_API_KEY'] = os.environ.get('ANTHROPIC_API_KEY')
    app.config['ANTHROPIC_MODEL'] = os.environ.get('ANTHROPIC_MODEL', 'claude-3-sonnet-20240229')
    
    # General LLM Configuration
    app.config['LLM_PROVIDER'] = os.environ.get('LLM_PROVIDER', 'openai')
    app.config['MAX_TOKENS'] = int(os.environ.get('MAX_TOKENS', 500))
    app.config['TEMPERATURE'] = float(os.environ.get('TEMPERATURE', 0.7))
    
    # Semantic Search Configuration
    app.config['SEARCH_TOP_K'] = int(os.environ.get('SEARCH_TOP_K', 3))
    app.config['SIMILARITY_THRESHOLD'] = float(os.environ.get('SIMILARITY_THRESHOLD', 0.3))
    
    # Initialize embeddings
    with app.app_context():
        initialize_embeddings()


# Flask route for manual embedding refresh (optional admin endpoint)
def create_admin_routes(app):
    """Create admin routes for embedding management"""
    
    @app.route('/admin/refresh-embeddings', methods=['POST'])
    def admin_refresh_embeddings():
        """Admin endpoint to refresh embeddings"""
        try:
            result = refresh_embeddings()
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/admin/embedding-stats', methods=['GET'])
    def admin_embedding_stats():
        """Admin endpoint to get embedding statistics"""
        global PRODUCT_EMBEDDINGS, PRODUCT_DATA
        
        stats = {
            "total_products": len(PRODUCT_DATA) if PRODUCT_DATA else 0,
            "embedding_dimensions": PRODUCT_EMBEDDINGS.shape[1] if PRODUCT_EMBEDDINGS is not None and len(PRODUCT_EMBEDDINGS) > 0 else 0,
            "model_used": EMBEDDING_MODEL,
            "cache_file_exists": os.path.exists("data/product_embeddings.pkl")
        }
        
        return jsonify(stats), 200
def clean_message(message_body):
    """Clean and extract keywords from message"""
    # Remove punctuation and convert to lowercase
    cleaned = re.sub(r'[^\w\s]', '', message_body.lower())
    # Split into words and filter out common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
    keywords = [word for word in cleaned.split() if word not in stop_words and len(word) > 2]
    return keywords

def extract_relevant_context(message_body):
    context = {}

    # Load product data
    with open("data/maka_products.json", "r", encoding="utf-8") as f:
        products = json.load(f)

    # Load contact info
    with open("data/maka_contact_info.json", "r", encoding="utf-8") as f:
        contact_info = json.load(f)

    # Clean input into keywords
    keywords = clean_message(message_body)
    print("🔑 Extracted keywords:", keywords)

    matched_products = []
    for pname, pdata in products.items():
        pname_lower = pname.lower()
        # Match if any keyword is in product name
        if any(kw in pname_lower for kw in keywords):
            matched_products.append(pdata)

    if matched_products:
        context["product_info"] = matched_products

    # Contact info detection
    contact_keywords = [
        "contact", "support", "email", "phone", "address", "website",
        "social", "instagram", "youtube", "tiktok", "whatsapp"
    ]
    if any(kw in message_body.lower() for kw in contact_keywords):
        context["contact_info"] = contact_info

    return context

def build_prompt(user_input, context):
    prompt = f"User asked: {user_input}\n\n"

    # Handle multiple product matches
    if "product_info" in context:
        products = context["product_info"]
        for i, product in enumerate(products, 1):
            prompt += (
                f"--- Product {i} ---\n"
                f"Name: {product.get('name')}\n"
                f"Price: {product.get('price')}\n"
                f"Description: {product.get('description')}\n"
                f"Features: {', '.join(product.get('features', []))}\n"
                f"Size: {product.get('size')}\n"
                f"Category: {product.get('category')}\n"
                f"Availability: {product.get('availability')}\n"
                f"URL: {product.get('url')}\n\n"
            )

    # Handle contact info
    if "contact_info" in context:
        contact = context["contact_info"]
        prompt += "Contact Info:\n"
        if contact.get("email"):
            prompt += f"Email: {', '.join(contact['email'])}\n"
        if contact.get("phone"):
            prompt += f"Phone: {', '.join(contact['phone'])}\n"
        if contact.get("address"):
            prompt += f"Address: {', '.join(contact['address'])}\n"
        if contact.get("website"):
            prompt += f"Website: {contact['website']}\n"
        if contact.get("social_media"):
            socials = contact["social_media"]
            prompt += "Social Media:\n"
            for platform, url in socials.items():
                prompt += f"- {platform.capitalize()}: {url}\n"
        prompt += "\n"

    prompt += "Based on the above data, respond to the user's message appropriately."
    print("this is the prompt", prompt)
    return prompt

def get_text_message_input(recipient, text):
    return json.dumps(
        {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": recipient,
            "type": "text",
            "text": {"preview_url": False, "body": text},
        }
    )


def refresh_embeddings():
    """
    Manually refresh product embeddings (useful for admin endpoints)
    """
    logging.info("Manually refreshing product embeddings...")
    generate_and_cache_embeddings()
    return {"status": "success", "message": f"Refreshed embeddings for {len(PRODUCT_DATA)} products"}


def add_new_product_embedding(product_name: str, product_data: Dict):
    """
    Add embedding for a new product without regenerating all embeddings
    """
    global PRODUCT_EMBEDDINGS, PRODUCT_DATA
    
    try:
        client = OpenAI(api_key=current_app.config['OPENAI_API_KEY'])
        
        # Create embed text for new product
        embed_text = create_product_embed_text(product_name, product_data)
        
        # Generate embedding
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=embed_text
        )
        new_embedding = np.array(response.data[0].embedding)
        
        # Add to existing data
        new_product = {
            "name": product_name,
            "data": product_data,
            "embed_text": embed_text
        }
        
        if PRODUCT_EMBEDDINGS is not None and len(PRODUCT_EMBEDDINGS) > 0:
            PRODUCT_EMBEDDINGS = np.vstack([PRODUCT_EMBEDDINGS, new_embedding])
            PRODUCT_DATA.append(new_product)
        else:
            PRODUCT_EMBEDDINGS = new_embedding.reshape(1, -1)
            PRODUCT_DATA = [new_product]
        
        # Update cache
        cache_data = {
            "embeddings": PRODUCT_EMBEDDINGS,
            "products": PRODUCT_DATA,
            "model": EMBEDDING_MODEL,
            "generated_at": np.datetime64('now').astype(str)
        }
        
        with open("data/product_embeddings.pkl", "wb") as f:
            pickle.dump(cache_data, f)
        
        logging.info(f"Added embedding for new product: {product_name}")
        
    except Exception as e:
        logging.error(f"Failed to add new product embedding: {e}")

# def generate_response(response):
#     # Return text in uppercase
#     return response.upper()
import logging
def generate_response(user_input, context_data):
    print("this is the context data", context_data)
    prompt = build_prompt(user_input, context_data)

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for the company's WhatsApp chatbot. Use only the provided context data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    except AuthenticationError:
        logging.error(" Invalid OpenAI API key. Returning admin contact message.")
        return "Lo siento, hubo un error interno. Por favor contacte al administrador."

    except RateLimitError:
        logging.error(" Rate limit exceeded.")
        return "Actualmente estamos recibiendo muchas solicitudes. Inténtelo de nuevo más tarde."

    except APIError as e:
        logging.error(f" OpenAI API error: {e}")
        return "Lo siento, hubo un problema con el servicio. Inténtelo más tarde."

    except Exception as e:
        logging.error(f" Unexpected error: {e}")
        return "Lo siento, hubo un problema procesando su solicitud. Inténtelo más tarde."


def send_message(data):
    """Send message via WhatsApp API"""
    headers = {
        "Content-type": "application/json",
        "Authorization": f"Bearer {current_app.config['ACCESS_TOKEN']}",
    }
    
    url = f"https://graph.facebook.com/{current_app.config['VERSION']}/{current_app.config['PHONE_NUMBER_ID']}/messages"
    
    try:
        response = requests.post(url, data=data, headers=headers, timeout=10)
        response.raise_for_status()
        logging.info("Message sent successfully")
        return response
    except requests.Timeout:
        logging.error("Timeout occurred while sending message")
        return jsonify({"status": "error", "message": "Request timed out"}), 408
    except requests.RequestException as e:
        logging.error(f"Request failed due to: {e}")
        return jsonify({"status": "error", "message": "Failed to send message"}), 500



def process_text_for_whatsapp(text):
    # Remove brackets
    pattern = r"\【.*?\】"
    # Substitute the pattern with an empty string
    text = re.sub(pattern, "", text).strip()

    # Pattern to find double asterisks including the word(s) in between
    pattern = r"\*\*(.*?)\*\*"

    # Replacement pattern with single asterisks
    replacement = r"*\1*"

    # Substitute occurrences of the pattern with the replacement
    whatsapp_style_text = re.sub(pattern, replacement, text)

    return whatsapp_style_text

def process_whatsapp_message(body):
    """Process WhatsApp message with semantic search"""
    print("Processing message body:", body)
    
    wa_id = body["entry"][0]["changes"][0]["value"]["contacts"][0]["wa_id"]
    name = body["entry"][0]["changes"][0]["value"]["contacts"][0]["profile"]["name"]
    
    message = body["entry"][0]["changes"][0]["value"]["messages"][0]
    message_body = message["text"]["body"]
    
    print("Extracting context with semantic search...")
    context_data = extract_relevant_context_with_embeddings(message_body)
    print("Context extracted successfully")
    
    # Generate response using LLM
    response = generate_llm_response(message_body, context_data, wa_id, name)
    
    # Process response for WhatsApp formatting
    response = process_text_for_whatsapp(response)
    
    # Send the response
    data = get_text_message_input(current_app.config["RECIPIENT_WAID"], response)
    send_message(data)
def extract_relevant_context_with_embeddings(message_body: str) -> Dict[str, Any]:
    """
    Extract relevant context using semantic search with embeddings.
    """
    context = {}
    
    # Semantic search for products
    matched_products = semantic_search_products(
        query=message_body,
        top_k=current_app.config.get('SEARCH_TOP_K', 3),
        similarity_threshold=current_app.config.get('SIMILARITY_THRESHOLD', 0.3)
    )
    
    if matched_products:
        # Add products with similarity scores for better ranking
        context["product_info"] = [item["product"] for item in matched_products]
        context["product_matches"] = matched_products  # Include similarity scores
        
        print(f"🔍 Found {len(matched_products)} semantically similar products:")
        for match in matched_products:
            print(f"  - {match['name']} (similarity: {match['similarity_score']:.3f})")
    
    # Contact info detection (keep existing logic)
    contact_keywords = [
        "contact", "support", "email", "phone", "address", "website",
        "social", "instagram", "youtube", "tiktok", "whatsapp", "help"
    ]
    if any(kw in message_body.lower() for kw in contact_keywords):
        try:
            with open("data/maka_contact_info.json", "r", encoding="utf-8") as f:
                contact_info = json.load(f)
                context["contact_info"] = contact_info
        except FileNotFoundError:
            logging.warning("Contact info file not found")
    
    return context

def generate_llm_response(user_input, context, wa_id, name):
    """
    Generate response using LLM with enhanced context from semantic search
    """
    # Build the system prompt
    system_prompt = build_system_prompt()
    
    # Build the user prompt with context
    user_prompt = build_user_prompt_with_scores(user_input, context, name)
    
    # Choose your LLM provider
    llm_provider = current_app.config.get('LLM_PROVIDER', 'openai')
    
    try:
        if llm_provider == 'openai':
            return generate_openai_response(system_prompt, user_prompt)
        elif llm_provider == 'anthropic':
            return generate_anthropic_response(system_prompt, user_prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
    
    except Exception as e:
        logging.error(f"LLM generation failed: {e}")
        return "I apologize, but I'm experiencing technical difficulties. Please try again later or contact our support team."
def build_system_prompt():
    """Build the system prompt for the LLM"""
    return """You are a helpful customer service assistant for Maka, a product/service company. Your role is to:

1. **Product Inquiries**: Help customers find products using intelligent semantic search
2. **Product Recommendations**: Suggest relevant products based on customer needs
3. **Contact Information**: Provide accurate contact details when requested
4. **General Support**: Answer questions professionally and helpfully
5. **Sales Support**: Guide customers through purchasing decisions

Guidelines:
- Be friendly, professional, and concise and simple response.
- Use the provided context data with similarity scores to prioritize the most relevant products
- Products are ranked by semantic similarity - higher scores mean better matches
- If you don't have specific information, guide users to contact support
- Keep responses conversational and suitable for WhatsApp
- Present multiple product options when available
- Always be helpful and solution-oriented"""


def build_user_prompt_with_scores(user_input, context, name):
    """Build the user prompt with context data including similarity scores"""
    prompt = f"Customer '{name}' asked: {user_input}\n\n"
    
    # Add product context with similarity scores
    if "product_info" in context and "product_matches" in context:
        prompt += "RELEVANT PRODUCTS (ranked by relevance):\n"
        products = context["product_info"]
        matches = context["product_matches"]
        
        for i, (product, match) in enumerate(zip(products, matches), 1):
            similarity_score = match["similarity_score"]
            prompt += f"""
Product {i} (Relevance: {similarity_score:.1%}):
- Name: {product.get('name', 'N/A')}
- Price: {product.get('price', 'N/A')}
- Description: {product.get('description', 'N/A')}
- Features: {', '.join(product.get('features', []))}
- Size: {product.get('size', 'N/A')}
- Category: {product.get('category', 'N/A')}
- Availability: {product.get('availability', 'N/A')}
- URL: {product.get('url', 'N/A')}
"""
    
    # Add contact context
    if "contact_info" in context:
        contact = context["contact_info"]
        prompt += "\nCOMPANY CONTACT INFORMATION:\n"
        if contact.get("email"):
            prompt += f"📧 Email: {', '.join(contact['email'])}\n"
        if contact.get("phone"):
            prompt += f"📞 Phone: {', '.join(contact['phone'])}\n"
        if contact.get("address"):
            prompt += f"📍 Address: {', '.join(contact['address'])}\n"
        if contact.get("website"):
            prompt += f"🌐 Website: {contact['website']}\n"
        if contact.get("social_media"):
            prompt += "📱 Social Media:\n"
            for platform, url in contact["social_media"].items():
                prompt += f"  - {platform.capitalize()}: {url}\n"
    
    prompt += "\nPlease respond to the customer's message using the above information. Focus on the most relevant products (higher relevance scores)."
    return prompt
def generate_openai_response(system_prompt, user_prompt):
    """Generate response using OpenAI"""
    client = OpenAI(api_key=current_app.config['OPENAI_API_KEY'])
    
    response = client.chat.completions.create(
        model=current_app.config.get('OPENAI_MODEL', 'gpt-3.5-turbo'),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=current_app.config.get('MAX_TOKENS', 500),
        temperature=current_app.config.get('TEMPERATURE', 0.7),
    )
    
    return response.choices[0].message.content


def generate_anthropic_response(system_prompt, user_prompt):
    """Generate response using Anthropic Claude"""
    from anthropic import Anthropic
    
    client = Anthropic(api_key=current_app.config['ANTHROPIC_API_KEY'])
    
    response = client.messages.create(
        model=current_app.config.get('ANTHROPIC_MODEL', 'claude-3-sonnet-20240229'),
        max_tokens=current_app.config.get('MAX_TOKENS', 500),
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt}
        ]
    )
    
    return response.content[0].text
def is_valid_whatsapp_message(body):
    """
    Check if the incoming webhook event has a valid WhatsApp message structure.
    """
    return (
        body.get("object")
        and body.get("entry")
        and body["entry"][0].get("changes")
        and body["entry"][0]["changes"][0].get("value")
        and body["entry"][0]["changes"][0]["value"].get("messages")
        and body["entry"][0]["changes"][0]["value"]["messages"][0]
    )
