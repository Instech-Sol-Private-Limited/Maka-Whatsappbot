import logging
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, time
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="WhatsApp Chatbot API", version="1.0.0")

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    CHROMADB_PATH = os.getenv("CHROMADB_PATH", "./chroma_db")
    WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
    WHATSAPP_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID")
    WHATSAPP_VERSION = os.getenv("WHATSAPP_VERSION", "v23.0")
    VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "your_verify_token")
    PRODUCTS_JSON_PATH = os.getenv("PRODUCTS_JSON_PATH", "data/maka_products.json")
    CONTACT_JSON_PATH = os.getenv("CONTACT_JSON_PATH", "data/maka_contact_info.json")
    SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "5"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
    MAX_MEMORY_MESSAGES = int(os.getenv("MAX_MEMORY_MESSAGES", "5"))  

    BOT_ACTIVE_WEEKDAYS = os.getenv("BOT_ACTIVE_WEEKDAYS", "Monday,Tuesday,Wednesday,Thursday,Friday")
    BOT_ACTIVE_WEEKDAY_START = os.getenv("BOT_ACTIVE_WEEKDAY_START", "20:00") 
    BOT_ACTIVE_WEEKDAY_END = os.getenv("BOT_ACTIVE_WEEKDAY_END", "06:00")  # 6 AM
    BOT_WEEKEND_24_7 = os.getenv("BOT_WEEKEND_24_7", "true").lower() == "true"
    BOT_INACTIVE_MESSAGE = os.getenv("BOT_INACTIVE_MESSAGE", 
        "Thank you for contacting us! Our chatbot is currently offline. "
        "We're available on weekdays from 8 PM to 6 AM, and 24/7 on weekends. "
        "Please try again during our active hours or contact us at info@maka.com.do for urgent matters."
    )
config = Config()

openai_client = OpenAI(api_key=config.OPENAI_API_KEY)

chroma_client = chromadb.PersistentClient(path=config.CHROMADB_PATH)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=config.OPENAI_API_KEY,
    model_name=config.EMBEDDING_MODEL
)

class WhatsAppMessage(BaseModel):
    object: str
    entry: List[Dict[str, Any]]

class ProductQuery(BaseModel):
    query: str
    top_k: int = Field(default=3, ge=1, le=10)
    threshold: float = Field(default=0.3, ge=0.0, le=1.0)

class EmbeddingRefreshResponse(BaseModel):
    status: str
    message: str
    products_count: int

# NEW: Conversation Memory Manager
class ConversationMemoryManager:
    """Manage conversation memory for multiple users"""
    
    def __init__(self, max_messages: int = 5):
        self.user_memories: Dict[str, ConversationBufferMemory] = {}
        self.max_messages = max_messages
        self.last_interaction: Dict[str, datetime] = {}
        
    def get_memory(self, phone_number: str) -> ConversationBufferMemory:
        """Get or create memory for a user"""
        # Clean up old conversations (inactive for more than 24 hours)
        self._cleanup_old_conversations()
        
        if phone_number not in self.user_memories:
            self.user_memories[phone_number] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="output"
            )
            logger.info(f"Created new memory for user: {phone_number}")
        
        self.last_interaction[phone_number] = datetime.now()
        return self.user_memories[phone_number]
    
    def _cleanup_old_conversations(self):
        """Remove conversations inactive for more than 24 hours"""
        now = datetime.now()
        inactive_users = [
            phone for phone, last_time in self.last_interaction.items()
            if (now - last_time).total_seconds() > 86400  # 24 hours
        ]
        
        for phone in inactive_users:
            if phone in self.user_memories:
                del self.user_memories[phone]
                del self.last_interaction[phone]
                logger.info(f"Cleaned up inactive conversation for: {phone}")
    
    def clear_memory(self, phone_number: str):
        """Clear memory for a specific user"""
        if phone_number in self.user_memories:
            del self.user_memories[phone_number]
            if phone_number in self.last_interaction:
                del self.last_interaction[phone_number]
            logger.info(f"Cleared memory for user: {phone_number}")
    
    def get_conversation_count(self, phone_number: str) -> int:
        """Get number of messages in conversation"""
        if phone_number in self.user_memories:
            memory = self.user_memories[phone_number]
            return len(memory.chat_memory.messages)
        return 0

# Global memory manager
memory_manager = ConversationMemoryManager(max_messages=config.MAX_MEMORY_MESSAGES)

class ChromaDBManager:
    def __init__(self):
        self.collection = None
        self.contact_info = None
        
    def initialize_collection(self):
        """Initialize or get ChromaDB collection"""
        try:
            self.collection = chroma_client.get_collection(
                name="products",
                embedding_function=openai_ef
            )
            logger.info(f"Loaded existing collection with {self.collection.count()} items")
        except Exception:
            self.collection = chroma_client.create_collection(
                name="products",
                embedding_function=openai_ef,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Created new ChromaDB collection")
            
        self._load_contact_info()
    
    def _load_contact_info(self):
        """Load contact information"""
        try:
            with open(config.CONTACT_JSON_PATH, "r", encoding="utf-8") as f:
                self.contact_info = json.load(f)
            logger.info("Contact info loaded successfully")
        except FileNotFoundError:
            logger.warning("Contact info file not found")
            self.contact_info = {}
    
    def load_products_to_chromadb(self):
        """Load products from JSON file into ChromaDB"""
        try:
            with open(config.PRODUCTS_JSON_PATH, "r", encoding="utf-8") as f:
                products = json.load(f)
            
            if not products:
                logger.warning("No products found in JSON file")
                return 0
            
            documents = []
            metadatas = []
            ids = []
            
            for idx, (product_name, product_data) in enumerate(products.items()):
                doc_text = self._create_product_text(product_name, product_data)
                documents.append(doc_text)
                
                metadata = {
                    "name": product_name,
                    "price": str(product_data.get("price", "")),
                    "category": product_data.get("category", ""),
                    "availability": product_data.get("availability", ""),
                    "url": product_data.get("url", ""),
                    "description": product_data.get("description", "")[:500],  
                }
                metadatas.append(metadata)
                ids.append(f"product_{idx}")
            
            if self.collection.count() > 0:
                self.collection.delete(ids=self.collection.get()["ids"])
            
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Loaded {len(documents)} products into ChromaDB")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Failed to load products: {e}")
            raise
    
    def _create_product_text(self, product_name: str, product_data: Dict) -> str:
        """Create rich text representation of product"""
        parts = [
            f"Product Name: {product_name}",
            f"Description: {product_data.get('description', 'No description available')}",
            f"Category: {product_data.get('category', 'Uncategorized')}",
            f"Price: {product_data.get('price', 'Price not available')}",
            f"Size: {product_data.get('size', 'Size not specified')}",
            f"Features: {', '.join(product_data.get('features', []))}",
            f"Availability: {product_data.get('availability', 'Unknown')}"
        ]
        return " | ".join(parts)
    
    def search_products(self, query: str, top_k: int = 5, threshold: float = 0.3) -> List[Dict]:
        """Search products using semantic similarity"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            if not results or not results["documents"][0]:
                return []
            
            # Format results
            formatted_results = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                similarity = 1 - distance
                
                if similarity >= threshold:
                    formatted_results.append({
                        "name": metadata.get("name", "Unknown"),
                        "description": metadata.get("description", ""),
                        "price": metadata.get("price", "N/A"),
                        "category": metadata.get("category", "N/A"),
                        "availability": metadata.get("availability", "N/A"),
                        "url": metadata.get("url", ""),
                        "similarity_score": float(similarity),
                        "rank": i + 1
                    })
            
            logger.info(f"Found {len(formatted_results)} products for query: {query}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

db_manager = ChromaDBManager()

class BotActivityChecker:
    """Check if bot should be active based on schedule"""
    
    @staticmethod
    def is_bot_active() -> bool:
        """Check if bot should be active based on current time and day"""
        now = datetime.now()
        current_time = now.time()
        current_day = now.weekday()  
        
        if current_day in [5, 6] and config.BOT_WEEKEND_24_7:
            logger.info(f"Bot is ACTIVE (Weekend 24/7 mode) - Day: {current_day}, Time: {current_time}")
            return True
        
        try:
            start_time = datetime.strptime(config.BOT_ACTIVE_WEEKDAY_START, "%H:%M").time()
            end_time = datetime.strptime(config.BOT_ACTIVE_WEEKDAY_END, "%H:%M").time()
        except ValueError:
            logger.error("Invalid time format in config. Using defaults: 20:00-06:00")
            start_time = time(20, 0)  
            end_time = time(6, 0)     
        
        if current_time >= start_time or current_time <= end_time:
            logger.info(f"Bot is ACTIVE (Weekday hours) - Day: {current_day}, Time: {current_time}")
            return True
        
        logger.info(f"Bot is INACTIVE - Day: {current_day}, Time: {current_time}")
        return False
    
    @staticmethod
    def get_next_active_time() -> str:
        """Get human-readable message about next active time"""
        now = datetime.now()
        current_day = now.weekday()
        
        if current_day < 5:  
            return "We'll be back at 8:00 PM today"
        elif current_day == 5:  
            return "We're available 24/7 this weekend!"
        else:  
            return "We're available 24/7 this weekend!"

bot_checker = BotActivityChecker()

def search_products_tool(query: str) -> str:
    """Search for products based on user query"""
    results = db_manager.search_products(query, top_k=config.SEARCH_TOP_K, threshold=config.SIMILARITY_THRESHOLD)
    
    if not results:
        return "No products found matching your query."
    
    response = "Here are the products I found:\n\n"
    for product in results:
        response += f"**{product['name']}**\n"
        response += f"Price: {product['price']}\n"
        response += f"Category: {product['category']}\n"
        response += f"Description: {product['description'][:200]}...\n"
        response += f"Availability: {product['availability']}\n"
        if product.get('url'):
            response += f"Link: {product['url']}\n"
        response += f"(Relevance: {product['similarity_score']:.0%})\n\n"
    
    return response

def get_contact_info_tool(query: str) -> str:
    """Get company contact information"""
    if not db_manager.contact_info:
        return "Contact information is not available at the moment."
    
    contact = db_manager.contact_info
    response = " **Contact Information**\n\n"
    
    if contact.get("email"):
        response += f" Email: {', '.join(contact['email'])}\n"
    if contact.get("phone"):
        response += f" Phone: {', '.join(contact['phone'])}\n"
    if contact.get("address"):
        response += f" Address: {', '.join(contact['address'])}\n"
    if contact.get("website"):
        response += f" Website: {contact['website']}\n"
    if contact.get("social_media"):
        response += "\n Social Media:\n"
        for platform, url in contact["social_media"].items():
            response += f"  • {platform.capitalize()}: {url}\n"
    # Add location
    if contact.get("location"):
        response += f"\n**Our Location**\n"
        response += f"Coordinates: {contact['location'].get('coordinates', '')}\n"
    if contact['location'].get('google_maps'):
            response += f"🗺️ Google Maps: {contact['location']['google_maps']}\n"
    if contact.get("social_media"):
        response += "\n**Social Media:**\n"
        for platform, url in contact["social_media"].items():
            response += f"  • {platform.capitalize()}: {url}\n"
    return response

def create_langchain_agent() -> AgentExecutor:
    """Create LangChain agent with tools"""
    
    tools = [
        Tool(
            name="search_products",
            func=search_products_tool,
            description="Search for products based on customer queries. Use this when customers ask about products, prices, availability, or features."
        ),
        Tool(
            name="get_contact_info",
            func=get_contact_info_tool,
            description="Get company contact information including email, phone, address, website, and social media. Use this when customers ask how to contact the company or need support."
        )
    ]
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a helpful customer service assistant for Maka company's WhatsApp chatbot.
Your responsibilities:
1. Help customers find products using the search_products tool
2. Provide accurate contact information using get_contact_info tool
3. Answer questions professionally and simple and concisely no emojies. 
4. Be friendly and conversational
5. Always use tools to get accurate information
6. Remember the conversation context and refer to previous messages when relevant

IMPORTANT DELIVERY INFORMATION:
- We deliver everywhere in World
- Delivery cost: RD$200
- FREE delivery on orders above RD$2500
- Cash on Delivery (Pay at delivery) is ONLY available in Santo Domingo
- Our location: https://maps.google.com/?q=18.456389,-69.955917, Santo Domingo, Dominican Republic
                                     
Guidelines:
- Keep responses brief and suitable for WhatsApp (2-3 short paragraphs max) and only provide information what user asked don't add extra info.
- If you don't have information, guide users to contact support
- When showing products, highlight key features and prices
- When discussing delivery, always mention the free delivery threshold 
- For whatsapp number only show number in this format +XXXXXXXXXXX
- If customer asks about payment on delivery, confirm their location is Santo Domingo                                         
- if there is any urls in response then it should be in wrapper not full url to be shown
- Reference previous conversation when the customer asks follow-up questions
- If a customer asks "what about the other one?" or similar, use context from chat history                  
"""),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    llm = ChatOpenAI(
        model=config.OPENAI_MODEL,
        temperature=0.3,
        openai_api_key=config.OPENAI_API_KEY
    )
    
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=3,
        handle_parsing_errors=True
    )
    
    return agent_executor

# Global agent instance
agent_executor = None

# WhatsApp utilities
def send_whatsapp_message(phone_number: str, message: str) -> bool:
    """Send message via WhatsApp Business API"""
    print("this is the whatapp version and whatsapp id", config)
    url = f"https://graph.facebook.com/{config.WHATSAPP_VERSION}/{config.WHATSAPP_PHONE_ID}/messages"
    
    headers = {
        "Authorization": f"Bearer {config.WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": phone_number,
        "type": "text",
        "text": {"body": message}
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        logger.info(f"Message sent to {phone_number}")
        return True
    except Exception as e:
        logger.error(f"Failed to send message: {e}")
        return False

def process_whatsapp_message(body: dict) -> bool:
    """Process incoming WhatsApp message"""
    try:
        # Extract message details
        value = body["entry"][0]["changes"][0]["value"]
        
        if "messages" not in value:
            return False
        
        message = value["messages"][0]
        phone_number = message["from"]
        message_text = message.get("text", {}).get("body", "")
        
        if not message_text:
            return False
        
        logger.info(f"Received message from {phone_number}: {message_text}")

        user_memory = memory_manager.get_memory(phone_number)

        
        # Generate response using LangChain agent
        response = agent_executor.invoke({
            "input": message_text,
            "chat_history": user_memory.chat_memory.messages
        })
        response_text = response.get("output", "I apologize, but I'm having trouble processing your request.")
                # Save conversation to memory
        user_memory.save_context(
            {"input": message_text},
            {"output": response_text}
        )
        
        logger.info(f"Conversation history for {phone_number}: {memory_manager.get_conversation_count(phone_number)} exchanges")

        # Send response
        send_whatsapp_message(phone_number, response_text)
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return False





@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global agent_executor
    
    logger.info("Starting up application...")
    
    db_manager.initialize_collection()
    
    # Check if embeddings already exist
    existing_count = db_manager.collection.count() if db_manager.collection else 0
    
    if existing_count > 0:
        logger.info(f"✓ Found existing embeddings: {existing_count} products already loaded")
        logger.info("Skipping embedding generation. Use /admin/refresh-embeddings to regenerate if needed.")
    else:
        logger.info("No existing embeddings found. Loading products and generating embeddings...")
        try:
            products_count = db_manager.load_products_to_chromadb()
            logger.info(f"✓ Successfully loaded {products_count} products with embeddings")
        except Exception as e:
            logger.error(f"✗ Failed to load products: {e}")
            logger.warning("Application will continue but product search may not work")
    
    agent_executor = create_langchain_agent()
    
    logger.info("✓ Application started successfully with conversation memory enabled")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "WhatsApp Chatbot API",
        "version": "1.0.0"
    }

@app.get("/webhook")
async def verify_webhook(request: Request):
    """Verify WhatsApp webhook"""
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")
    
    if mode == "subscribe" and token == config.VERIFY_TOKEN:
        logger.info("Webhook verified successfully")
        return int(challenge)
    
    raise HTTPException(status_code=403, detail="Verification failed")

# @app.post("/webhook")
# async def webhook(request: Request):
#     """Handle incoming WhatsApp messages"""
#     try:
#         body = await request.json()
        
#         # Check for status updates
#         if (body.get("entry", [{}])[0]
#             .get("changes", [{}])[0]
#             .get("value", {})
#             .get("statuses")):
#             return {"status": "ok"}
        
#         # Process message
#         if process_whatsapp_message(body):
#             return {"status": "ok"}
        
#         return {"status": "no_message"}
        
#     except Exception as e:
#         logger.error(f"Webhook error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/webhook")
async def webhook(request: Request):
    """Handle incoming WhatsApp messages"""
    try:
        body = await request.json()
        
        # Check for status updates
        if (body.get("entry", [{}])[0]
            .get("changes", [{}])[0]
            .get("value", {})
            .get("statuses")):
            return {"status": "ok"}
        
        # Check if bot is active before processing message
        if not bot_checker.is_bot_active():
            logger.info("Bot is currently inactive - staying silent (no message sent)")
            
            # Log the incoming message for monitoring purposes
            try:
                value = body["entry"][0]["changes"][0]["value"]
                if "messages" in value:
                    message = value["messages"][0]
                    phone_number = message["from"]
                    message_text = message.get("text", {}).get("body", "")
                    
                    logger.info(f"Received message during inactive hours from {phone_number}: {message_text}")
                    logger.info("Message logged - no automated response sent (manual handling required)")
            except Exception as e:
                logger.error(f"Error logging inactive message: {e}")
            
            # Return success without sending any message
            return {"status": "inactive_silent"}
        
        # Bot is active - process message normally
        if process_whatsapp_message(body):
            return {"status": "ok"}
        
        return {"status": "no_message"}
        
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/refresh-embeddings")
async def refresh_embeddings():
    """Refresh product embeddings in ChromaDB"""
    try:
        count = db_manager.load_products_to_chromadb()
        return EmbeddingRefreshResponse(
            status="success",
            message="Embeddings refreshed successfully",
            products_count=count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/stats")
async def get_stats():
    """Get collection statistics"""
    return {
        "products_count": db_manager.collection.count() if db_manager.collection else 0,
        "embedding_model": config.EMBEDDING_MODEL,
        "llm_model": config.OPENAI_MODEL,
        "collection_name": db_manager.collection.name if db_manager.collection else None
    }

@app.post("/search")
async def search_products(query: ProductQuery):
    """Search products endpoint"""
    results = db_manager.search_products(
        query.query,
        top_k=query.top_k,
        threshold=query.threshold
    )
    return {"results": results, "count": len(results)}

@app.post("/chat")
async def chat(request: Request):
    """Direct chat endpoint for testing"""
    body = await request.json()
    query = body.get("message", "")
    
    if not query:
        raise HTTPException(status_code=400, detail="Message is required")
    
    try:
        response = agent_executor.invoke({"input": query})
        return {"response": response.get("output", "")}
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)