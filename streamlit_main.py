# --- Core Libraries ---
import os
import sys
import traceback
import streamlit as st
import pandas as pd
import datetime
import random
import json

# --- NEW: OCR Libraries ---
import pytesseract
from PIL import Image # Pillow library for image handling
import io # For handling image bytes

# --- LLM Interaction ---
import openai
from openai import OpenAI
# from google import genai
from google.genai import types
import google.generativeai as genai



# --- Configuration ---
st.set_page_config(
    page_title="AI Inventory Manager",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("📦 AI-Powered Inventory Management")

# --- Custom CSS/JS ---
# For sticky sidebar header and potentially Go-to-Top JS later
st.markdown("""
    <style>
        /* Make sidebar header sticky */
        div[data-testid="stSidebarNav"] ul {
            position: sticky;
            top: 0;
            background: #F0F2F6; /* Match Streamlit's light theme sidebar background */
            z-index: 999;
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        /* Adjust for dark theme */
        [data-theme="dark"] div[data-testid="stSidebarNav"] ul {
            background: #1E1E1E; /* Example dark theme background */
        }
        /* Hide the Streamlit "hamburger" menu since we have sidebar nav */
        /* button[title="View fullscreen"] { display: none; } */ /* Optional: Hide fullscreen */
    </style>
    """, unsafe_allow_html=True)


# --- OpenRouter Configuration ---
# Read the App URL from secrets - YOU MUST SET THIS in your Streamlit Cloud secrets!
# Example: APP_URL="https://your-app-name.streamlit.app"
# Use a generic placeholder if not set in secrets.
OPENROUTER_REFERRER_URL = st.secrets.get("APP_URL", "https://inventory-app-placeholder.streamlit.app/") # Added trailing slash just in case
OPENROUTER_APP_TITLE = "Streamlit Inventory Chatbot" # Can customize this

# --- Initialize LLM Client (OpenRouter) ---
llm_provider = "None"
llm_enabled = False
client = None

try:
    # api_key = st.secrets["GITHUB_TOKEN"] # Must be set in secrets
    # # api_key = st.secrets["OPENROUTER_KEY"]

    # base_url="https://models.github.ai/inference"
    # # base_url = "https://openrouter.ai/api/v1"

    # model = "openai/gpt-4o"
    # # model = "mistralai/mistral-small-3.1-24b-instruct:free"

    # client = OpenAI(
    #     base_url=base_url,
    #     api_key=api_key,
    # )
    # # Quick check to validate credentials and connection during startup
    # chat = client.chat.completions.create(
    #     model=model,
    #     messages=[{"role": "user", "content": "Hello!"}],
    #     max_tokens=5
    # )

    # Gemini API
    # api_key 
    api_key = st.secrets["GEMINI_API_KEY"] # Must be set in secrets
    # model
    model = "gemini-2.0-flash"

    # ----------------------------------- > NEW < -----------------------------------
    gemini_model = genai.GenerativeModel(model)

    # ----------------------------------- > OLD < -----------------------------------

    # client
    # client = genai.Client(api_key= api_key)



    # # response - chat
    # response = client.models.generate_content(
    #     model=model, 
    #     contents="Hello!",
    #     config=types.GenerateContentConfig(
    #         max_output_tokens=5,
    #         temperature=0.1
    #     )
    # )

    llm_provider = model
    llm_enabled = True
    

except KeyError:
    st.sidebar.error(f"`{llm_provider}_API_KEY` not found in Streamlit secrets. Chatbot disabled.", icon="⚠️")
except openai.AuthenticationError:
    st.sidebar.error(f"{llm_provider} Authentication Error: Invalid API Key. Chatbot disabled.", icon="🚨")
except openai.APIConnectionError as e:
    st.sidebar.error(f"{llm_provider} Connection Error. Chatbot disabled. Error: {e}", icon="🚨")
except Exception as e:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    # Get the last frame from the traceback
    last_frame = traceback.extract_tb(exc_traceback)[-1]
    file_name = last_frame.filename
    line_no = last_frame.lineno
    func_name = last_frame.name
    line_content = last_frame.line
    st.sidebar.error(f"Error initializing {llm_provider}: {e}. Chatbot disabled." + str(line_no) + str(line_content), icon="⚠️")
    st.sidebar.error(f"Exception:{e}, line no:{line_no}, file name:{file_name}, function name:{func_name}, line content:{line_content}, last frame:{last_frame}", icon="⚠️")


# Helper function to reset chat history
def reset_chat_history():
    assistant_greeting = f"Hello! Ask me about inventory (via {llm_provider})." if llm_enabled else "Hello! LLM inactive. Ask about inventory."
    st.session_state.messages = [{"role": "assistant", "content": assistant_greeting}]

if "messages" not in st.session_state:
    reset_chat_history()

# --- Session State Initialization ---
# Inventory Data (In-memory dictionary)
if 'inventory' not in st.session_state:
    # Sample Data - replace or load from persistent storage in a real app
    st.session_state.inventory = {
        "ITEM001": {"name": "Laptop", "quantity": 15, "price": 1200.00, "last_updated": datetime.datetime.now()},
        "ITEM002": {"name": "Keyboard", "quantity": 50, "price": 75.00, "last_updated": datetime.datetime.now()},
        "ITEM003": {"name": "Mouse", "quantity": 45, "price": 25.50, "last_updated": datetime.datetime.now()},
        "ITEM004": {"name": "Monitor", "quantity": 10, "price": 300.00, "last_updated": datetime.datetime.now()},
        "ITEM005": {"name": "Webcam", "quantity": 30, "price": 45.00, "last_updated": datetime.datetime.now()},
    }

# Chat History - Ensure it's always a list of dictionaries
if "messages" not in st.session_state:
    assistant_greeting = f"Hello! Ask me anything about the inventory (using {llm_provider})." if llm_enabled else "Hello! Inventory Bot here. LLM connection failed."
    st.session_state.messages = [{"role": "assistant", "content": assistant_greeting}]
# Ensure existing messages are dicts (for dev robustness if code changed)
st.session_state.messages = [msg if isinstance(msg, dict) else {"role": getattr(msg, 'role', 'unknown'), "content": getattr(msg, 'content', None), "tool_calls": getattr(msg, 'tool_calls', None)} for msg in st.session_state.messages]


# --- Helper Functions ---
def get_inventory_df():
    """Converts inventory dict to a formatted Pandas DataFrame for display."""
    if not st.session_state.inventory:
        return pd.DataFrame(columns=["Item ID", "Name", "Quantity", "Price", "Last Updated"])

    data_list = [{'Item ID': k, **v} for k, v in st.session_state.inventory.items()]
    df = pd.DataFrame(data_list)

    # --- Formatting (handle missing columns gracefully) ---
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['Formatted Price'] = df['price'].apply(lambda x: '${:,.2f}'.format(x) if pd.notna(x) else 'N/A')
    else:
        df['Formatted Price'] = 'N/A'

    if 'last_updated' in df.columns:
        df['Formatted Last Updated'] = pd.to_datetime(df['last_updated'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
        df['Formatted Last Updated'] = df['Formatted Last Updated'].fillna('N/A')
    else:
        df['Formatted Last Updated'] = 'N/A'

    # --- Column Selection and Renaming for Display ---
    display_columns_map = {
        "Item ID": "Item ID", "name": "Name", "quantity": "Quantity",
        "Formatted Price": "Price", "Formatted Last Updated": "Last Updated"
    }
    # Select existing original or created formatted columns relevant for display
    cols_to_select = [col for col in ["Item ID", "name", "quantity", "Formatted Price", "Formatted Last Updated"]
                      if col in df.columns or col in ['Formatted Price', 'Formatted Last Updated']]
    df_display = df[cols_to_select]

    # Rename the selected columns to user-friendly names
    cols_to_rename = {k: v for k, v in display_columns_map.items() if k in cols_to_select}
    df_display = df_display.rename(columns=cols_to_rename)

    # Ensure final columns are in the desired order, filling missing ones
    final_display_order = ["Item ID", "Name", "Quantity", "Price", "Last Updated"]
    df_display = df_display.reindex(columns=final_display_order, fill_value='N/A')

    return df_display

def generate_item_id():
    """Generates a simple unique item ID (example implementation)."""
    prefix = "ITEM"
    while True:
        new_id = f"{prefix}{random.randint(100, 999)}"
        # Ensure uniqueness within the current session's inventory
        if new_id not in st.session_state.inventory:
            return new_id

# --- Inventory Functions Accessible by LLM (Tools) ---
# These functions MUST return JSON serializable data, ideally JSON strings.
def get_inventory_summary():
    """
    Provides a summary of the inventory: total distinct items and total quantity.
    Returns a JSON string with the summary details.
    """
    inventory = st.session_state.inventory
    count = len(inventory)
    if count == 0:
        return json.dumps({"status": "empty", "message": "The inventory is currently empty."})
    total_quantity = sum(item.get('quantity', 0) for item in inventory.values())
    return json.dumps({
        "status": "success",
        "distinct_items": count,
        "total_quantity": total_quantity,
        "summary_message": f"Inventory has {count} distinct items with a total quantity of {total_quantity} units."
    })

def get_item_details(item_identifier):
    """
    Retrieves details (quantity, price) for one item by its name or Item ID.
    Args: item_identifier (str): The name (e.g., 'Laptop') or ID (e.g., 'ITEM001'). Case-insensitive for names.
    Returns a JSON string with item details or a 'not_found' status.
    """
    inventory = st.session_state.inventory
    identifier_norm = item_identifier.strip()
    item_id_match = identifier_norm # Assume it might be an ID first

    # Check by ID (case-sensitive as defined in inventory keys)
    if item_id_match in inventory:
        item = inventory[item_id_match]
        return json.dumps({
            "status": "found", "id": item_id_match, "name": item.get('name', 'N/A'),
            "quantity": item.get('quantity', 'N/A'), "price": item.get('price', 'N/A')
        })

    # Check by name (case-insensitive)
    identifier_lower = identifier_norm.lower()
    for item_id, details in inventory.items():
        if details.get('name', '').lower() == identifier_lower:
            return json.dumps({
                "status": "found", "id": item_id, "name": details.get('name', 'N/A'),
                "quantity": details.get('quantity', 'N/A'), "price": details.get('price', 'N/A')
            })

    # Not found by ID or name
    return json.dumps({"status": "not_found", "identifier": item_identifier})

def find_low_stock_items(quantity_threshold=10):
    """
    Finds items with quantity at or below a threshold.
    Args: quantity_threshold (int): Max quantity for low stock. Defaults to 10.
    Returns a JSON string listing low stock items or a 'none_found' status.
    """
    inventory = st.session_state.inventory
    low_stock_items = []
    for item_id, details in inventory.items():
        # Ensure quantity is treated as a number, default to infinity if missing/invalid
        try:
             current_quantity = int(details.get('quantity', float('inf')))
        except (ValueError, TypeError):
             current_quantity = float('inf') # Skip if quantity is not a valid number

        if current_quantity <= quantity_threshold:
            low_stock_items.append({
                "id": item_id,
                "name": details.get('name', 'N/A'),
                "quantity": details.get('quantity') # Report original value
            })

    if not low_stock_items:
        return json.dumps({"status": "none_found", "threshold": quantity_threshold})
    else:
        return json.dumps({
            "status": "found", "threshold": quantity_threshold, "items": low_stock_items
        })

# --- Inventory Functions Accessible by LLM (Tools) ---
# ... (Keep existing functions: get_inventory_summary, get_item_details, find_low_stock_items) ...

# In add_inventory_item function:

def add_inventory_item(item_name, quantity, price):
    """
    Adds a new item. Requires name, POSITIVE quantity, and POSITIVE price.
    Returns JSON string confirming success/failure. Includes new Item ID if successful.
    """
    inventory = st.session_state.inventory
    item_name = item_name.strip()

    # --- Input Validation within the function ---
    if not item_name:
        return json.dumps({"status": "validation_error", "message": "Cannot add item: Item name is missing or empty."})
    try:
        quantity = int(quantity)
        if quantity <= 0: # Require positive
            return json.dumps({"status": "validation_error", "message": f"Cannot add '{item_name}': The quantity must be a positive number (you provided {quantity}). Please provide a valid quantity."})
    except (ValueError, TypeError):
        return json.dumps({"status": "validation_error", "message": f"Cannot add '{item_name}': Invalid quantity value '{quantity}'. Must be a positive whole number."})
    try:
        price = float(price)
        if price <= 0.0: # Require positive
             return json.dumps({"status": "validation_error", "message": f"Cannot add '{item_name}': The price must be a positive number (you provided ${price:.2f}). Please provide a valid price."})
    except (ValueError, TypeError):
         return json.dumps({"status": "validation_error", "message": f"Cannot add '{item_name}': Invalid price value '{price}'. Must be a positive number."})

    # --- Generate ID and Add ---
    try:
        for item_id, details in inventory.items(): # Check duplicates
            if details.get('name','').lower() == item_name.lower():
                 return json.dumps({"status": "validation_error", "message": f"Cannot add '{item_name}': Item already exists (ID: {item_id}). Use 'update' if needed."})
        new_id = generate_item_id()
        inventory[new_id] = {"name": item_name, "quantity": quantity, "price": price, "last_updated": datetime.datetime.now()}
        return json.dumps({"status": "success", "message": f"Added '{item_name}' (ID: {new_id}) qty {quantity}, price ${price:.2f}.", "item_id": new_id, "name": item_name, "quantity": quantity, "price": price})
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Error adding item: {str(e)}"})

def update_inventory_item(item_identifier, new_name=None, new_quantity=None, new_price=None):
    """
    Updates an existing inventory item identified by its name or ID.
    At least one field (new_name, new_quantity, new_price) must be provided for update.
    Args:
        item_identifier (str): The name or ID of the item to update.
        new_name (str, optional): The new name for the item.
        new_quantity (int, optional): The new quantity. Must be non-negative.
        new_price (float, optional): The new price. Must be non-negative.
    Returns a JSON string indicating success or failure.
    """
    inventory = st.session_state.inventory
    identifier_norm = item_identifier.strip()

    # --- Find the item ---
    item_id_to_update = None
    original_item = None

    # Check by ID
    if identifier_norm in inventory:
        item_id_to_update = identifier_norm
        original_item = inventory[item_id_to_update]
    else:
        # Check by name (case-insensitive)
        identifier_lower = identifier_norm.lower()
        for item_id, details in inventory.items():
            if details.get('name', '').lower() == identifier_lower:
                item_id_to_update = item_id
                original_item = details
                break

    if not item_id_to_update or not original_item:
        return json.dumps({"status": "not_found", "message": f"Could not find item '{item_identifier}' to update."})

    # --- Check if any update field was provided ---
    if new_name is None and new_quantity is None and new_price is None:
        return json.dumps({"status": "validation_error", "message": f"No update provided for '{original_item.get('name', item_id_to_update)}'. Please specify a new name, quantity, or price."})

    # --- Validate provided updates ---
    updates_to_apply = {}
    validation_errors = []

    if new_name is not None:
        new_name = new_name.strip()
        if not new_name:
            validation_errors.append("New name cannot be empty.")
        else:
            updates_to_apply["name"] = new_name

    if new_quantity is not None:
        try:
            new_quantity = int(new_quantity)
            if new_quantity < 0:
                validation_errors.append(f"New quantity ({new_quantity}) cannot be negative.")
            else:
                updates_to_apply["quantity"] = new_quantity
        except (ValueError, TypeError):
            validation_errors.append(f"Invalid new quantity value '{new_quantity}'. Must be a whole number.")

    if new_price is not None:
        try:
            new_price = float(new_price)
            if new_price < 0.0:
                validation_errors.append(f"New price (${new_price}) cannot be negative.")
            else:
                updates_to_apply["price"] = new_price
        except (ValueError, TypeError):
            validation_errors.append(f"Invalid new price value '{new_price}'. Must be a number.")

    if validation_errors:
        error_message = f"Cannot update '{original_item.get('name', item_id_to_update)}': " + " ".join(validation_errors)
        return json.dumps({"status": "validation_error", "message": error_message})

    # --- Apply Updates ---
    try:
        updated_item_data = original_item.copy() # Start with existing data
        updated_item_data.update(updates_to_apply) # Overwrite with validated changes
        updated_item_data["last_updated"] = datetime.datetime.now() # Update timestamp

        inventory[item_id_to_update] = updated_item_data # Save back to inventory

        return json.dumps({
            "status": "success",
            "message": f"Successfully updated item '{updated_item_data.get('name')}' (ID: {item_id_to_update}).",
            "item_id": item_id_to_update,
            "updated_fields": list(updates_to_apply.keys()), # List fields that were changed
            "new_data": {k: updated_item_data.get(k) for k in ["name", "quantity", "price"]} # Show current state
        })
    except Exception as e:
         return json.dumps({"status": "error", "message": f"An unexpected error occurred updating item: {str(e)}"})
    

# --- Define Tools for LLM ---
# Map tool names to actual Python functions
available_functions = {
    "get_inventory_summary": get_inventory_summary,
    "get_item_details": get_item_details,
    "find_low_stock_items": find_low_stock_items,
    "add_inventory_item": add_inventory_item, 
    "update_inventory_item": update_inventory_item,
}

# Define tool structure for the LLM API call
# function_declarations = [
#     types.FunctionDeclaration(name="get_inventory_summary", description="Get a summary of the inventory status: total distinct items and total quantity."),
#     types.FunctionDeclaration(name="get_item_details", description="Get details (quantity, price) for a specific item by its name or Item ID.", parameters={"type": "object", "properties": { "item_identifier": {"type": "string", "description": "The name (e.g., 'Laptop', 'Keyboard') or Item ID (e.g., 'ITEM001') of the inventory item."}}, "required":["item_identifier"]}),
#     types.FunctionDeclaration(name="find_low_stock_items", description="Find items in the inventory that are low in stock, based on a quantity threshold.", parameters={"type": "object", "properties": { "quantity_threshold": {"type": "integer", "description": "The quantity threshold. Items with quantity at or below this value are considered low stock. Defaults to 10 if not specified by the user."}}, "required":[]}),
#     types.FunctionDeclaration(name="add_inventory_item", description="Adds a new item to the inventory system. Requires the item's name, quantity, and price.", parameters={"type": "object", "properties": { "item_name": {"type": "string", "description": "The name of the new item to add."}, "quantity": {"type": "integer", "description": "The initial stock quantity for the new item."}, "price": {"type": "number", "description": "The price per unit for the new item."}}, "required":["item_name", "quantity", "price"]}),
#     types.FunctionDeclaration(name="update_inventory_item", description="Updates an existing item in the inventory. Requires the item's current name or ID, and at least one field to update (new name, new quantity, or new price).", parameters={"type": "object", "properties": {"item_identifier": {"type": "string", "description": "The current name or Item ID of the item to be updated."}, "new_name": {"type": "string", "description": "The new name for the item (optional)."}, "new_quantity": {"type": "integer", "description": "The new stock quantity for the item (optional)."}, "new_price": {"type": "number", "description": "The new price per unit for the item (optional)."}}, "required":["item_identifier"]}),
# ]
# Define tool structure for the LLM API call
function_declarations = [
    {
        "name": "get_inventory_summary",
        "description": "Get a summary of the inventory status: total distinct items and total quantity."
    },
    {
        "name": "get_item_details",
        "description": "Get details (quantity, price) for a specific item by its name or Item ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "item_identifier": {
                    "type": "string",
                    "description": "The name (e.g., 'Laptop', 'Keyboard') or Item ID (e.g., 'ITEM001') of the inventory item."
                }
            },
            "required": ["item_identifier"]
        }
    },
    {
        "name": "find_low_stock_items",
        "description": "Find items in the inventory that are low in stock, based on a quantity threshold.",
        "parameters": {
            "type": "object",
            "properties": {
                "quantity_threshold": {
                    "type": "integer",
                    "description": "The quantity threshold. Items with quantity at or below this value are considered low stock. Defaults to 10 if not specified by the user."
                }
            },
            "required": []
        }
    },
    {
        "name": "add_inventory_item",
        "description": "Adds a new item to the inventory system. Requires the item's name, quantity, and price.",
        "parameters": {
            "type": "object",
            "properties": {
                "item_name": {"type": "string", "description": "The name of the new item to add."},
                "quantity": {"type": "integer", "description": "The initial stock quantity for the new item."},
                "price": {"type": "number", "description": "The price per unit for the new item."}
            },
            "required": ["item_name", "quantity", "price"]
        }
    },
    {
        "name": "update_inventory_item",
        "description": "Updates an existing item in the inventory. Requires the item's current name or ID, and at least one field to update (new name, new quantity, or new price).",
        "parameters": {
            "type": "object",
            "properties": {
                "item_identifier": {"type": "string", "description": "The current name or Item ID of the item to be updated."},
                "new_name": {"type": "string", "description": "The new name for the item (optional)."},
                "new_quantity": {"type": "integer", "description": "The new stock quantity for the item (optional)."},
                "new_price": {"type": "number", "description": "The new price per unit for the item (optional)."}
            },
            "required": ["item_identifier"]
        }
    },
]


# --- LLM Interaction Logic ---
# ----------------------------------- > OLD < -----------------------------------
# def run_conversation(user_prompt):
#     """Sends conversation to OpenRouter, handles tool calls, returns final response."""
#     if not client or not llm_enabled:
#         return f"LLM client ({llm_provider}) not available. Cannot process request."

#     # --- Choose Model ---
#     model_name = model # A reliable choice for OpenAI-style function calling

#     # --- Prepare History ---
#     st.session_state.messages.append({"role": "user", "content": user_prompt})
#     messages_for_api = st.session_state.messages

#     try:
#         # Configure the client and tools
#         tools = types.Tool(function_declarations=function_declarations)
#         config = types.GenerateContentConfig(tools=[tools])

#         # Send request with function declarations
#         response = client.models.generate_content(
#             model="gemini-2.0-flash",
#             contents=user_prompt,
#             config=config,
#         )

#         # Check for a function call
#         response_message = response.candidates[0].content # Gemini Content object
#         tool_calls = response_message.parts[0].function_call

#         # --- Handle Tool Calls (if any) ---
#         if tool_calls:
#             # **FIX:** Append the assistant's response as a dictionary
#             assistant_message_dict = {
#                 "role": response_message.role,
#                 "content": response_message.content, # May be None
#                 "tool_calls": response_message.tool_calls # Store the list of tool call objects
#             }
#             st.session_state.messages.append(assistant_message_dict)

#             # Execute tools and collect results
#             tool_results_messages = []
#             for tool_call in tool_calls:
#                 function_name = tool_call.function.name
#                 function_to_call = available_functions.get(function_name)
#                 function_args_str = tool_call.function.arguments
#                 function_response = None # Initialize response variable

#                 if not function_to_call:
#                     function_response = json.dumps({"status": "error", "message": f"Tool '{function_name}' not found or not implemented."})
#                 else:
#                     try:
#                         function_args = json.loads(function_args_str)

#                         # --- Specific Function Argument Handling ---
#                         if function_name == "find_low_stock_items":
#                              threshold = function_args.get("quantity_threshold", 10)
#                              function_response = function_to_call(quantity_threshold=threshold)

#                         elif function_name == "get_item_details":
#                              identifier = function_args.get("item_identifier")
#                              if identifier: function_response = function_to_call(item_identifier=identifier)
#                              else: function_response = json.dumps({"status": "error", "message": "Missing 'item_identifier' argument for get_item_details."})

#                         elif function_name == "get_inventory_summary":
#                              function_response = function_to_call()

#                         elif function_name == "add_inventory_item":
#                             # Pre-call check for KEY PRESENCE from LLM Arguments
#                             # We check if the LLM actually included values for the required fields in its function call request.
#                             name = function_args.get("item_name")
#                             qty = function_args.get("quantity")
#                             prc = function_args.get("price")

#                             missing_keys = []
#                             # Use a stricter check: Is the key present AND is the value not None?
#                             # (Some LLMs might send null explicitly if they don't know)
#                             if function_args.get("item_name") is None: missing_keys.append("item_name")
#                             if function_args.get("quantity") is None: missing_keys.append("quantity")
#                             if function_args.get("price") is None: missing_keys.append("price")

#                             if missing_keys:
#                                 # **CONVERSATIONAL ERROR FEEDBACK for LLM:**
#                                 # Tell the LLM precisely how to respond to the user to get the missing info.
#                                 # Use placeholders like {item_name} that the LLM can fill if the name *was* provided.
#                                 provided_name = name if name else "(unspecified item)" # Use provided name if available
#                                 details_needed = ', '.join(missing_keys).replace("item_name","name").replace("quantity","quantity").replace("price","price")

#                                 # This message is designed to be the core of the LLM's *next conversational turn*
#                                 user_facing_request = f"Okay, I can try to add '{provided_name}', but I need more details. Could you please provide the {details_needed} for this item?"

#                                 function_response = json.dumps({
#                                     "status": "error_user_input_required", # Specific status
#                                     # The 'message' should guide the LLM's response generation directly
#                                     "message": user_facing_request
#                                 })
#                             else:
#                                 # Keys and non-None values were provided by LLM.
#                                 # Proceed to call the function for value validation (e.g., positive checks).
#                                 function_response = function_to_call(item_name=name, quantity=qty, price=prc)
#                         elif function_name == "update_inventory_item":
#                             identifier = function_args.get("item_identifier")
#                             new_name = function_args.get("new_name") # Will be None if not provided
#                             new_qty = function_args.get("new_quantity") # Will be None if not provided
#                             new_prc = function_args.get("new_price") # Will be None if not provided

#                             if not identifier:
#                                 function_response = json.dumps({"status": "error", "message": "Missing 'item_identifier' argument for update_inventory_item."})
#                             elif new_name is None and new_qty is None and new_prc is None:
#                                 # Check if at least one update field was given by LLM
#                                 function_response = json.dumps({"status": "error", "message": "Cannot update item. No new name, quantity, or price was specified. Ask the user what they want to change."})
#                             else:
#                                 # Call update function with provided args (function handles internal validation)
#                                 function_response = function_to_call(
#                                     item_identifier=identifier,
#                                     new_name=new_name,
#                                     new_quantity=new_qty,
#                                     new_price=new_prc
#                                 )

#                         else: # Fallback for any unexpected function name
#                             function_response = json.dumps({"status": "error", "message": f"Function '{function_name}' is recognized but argument handling is not implemented."})

#                     except json.JSONDecodeError: function_response = json.dumps({"status": "error", "message": f"Invalid arguments format from LLM for {function_name}: {function_args_str}"})
#                     except Exception as e: function_response = json.dumps({"status": "error", "message": f"Error preparing to execute {function_name}: {str(e)}"})


#                 # Prepare message for API with tool result (ensure response is not None)
#                 if function_response is None:
#                     function_response = json.dumps({"status": "error", "message": f"Execution failed to produce a result for {function_name}."})

#                 tool_results_messages.append({
#                     "tool_call_id": tool_call.id, "role": "tool", "name": function_name,
#                     "content": function_response, # Function output (JSON string)
#                 })

#             # Add all tool results (dictionaries) to the main message history
#             st.session_state.messages.extend(tool_results_messages)

#             # --- Second API Call: Send tool results back to LLM ---
#             messages_for_second_call = st.session_state.messages
#             second_response = client.chat.completions.create(
#                 model=model_name, messages=messages_for_second_call
#             )
#             final_response_content = second_response.choices[0].message.content
#             # Append final assistant response (dictionary)
#             st.session_state.messages.append({"role": "assistant", "content": final_response_content})
#             return final_response_content

#         # --- Handle Direct Response (No Tool Call) ---
#         else:
#             final_response_content = response_message.content
#             # Append direct assistant response (dictionary)
#             st.session_state.messages.append({"role": "assistant", "content": final_response_content})
#             return final_response_content

#     # --- Error Handling ---
#     except openai.APIError as e:
#         error_msg = f"{llm_provider} API Error ({model_name}): {e}"
#         st.error(error_msg, icon="🚨")
#         st.session_state.messages.append({"role": "assistant", "content": f"Sorry, encountered an API error: {e}"})
#         return f"API Error: {e}"
#     except Exception as e:
#         exc_type, exc_value, exc_traceback = sys.exc_info()
#         # Get the last frame from the traceback
#         last_frame = traceback.extract_tb(exc_traceback)[-1]
#         file_name = last_frame.filename
#         line_no = last_frame.lineno
#         func_name = last_frame.name
#         line_content = last_frame.line
#         print("\n--- Extracted Details (Last Frame) ---")
#         print(f"File: {file_name}")
#         print(f"Function: {func_name}")
#         print(f"Line Number: {line_no}")
#         print(f"Line Content: {line_content}")
#         error_msg = f"Unexpected error during LLM interaction ({model_name}): {e}"
#         st.error(error_msg + str(line_no), icon="🚨")
#         if st.session_state.messages and st.session_state.messages[-1]["role"] == "user": st.session_state.messages.pop()
#         st.session_state.messages.append({"role": "assistant", "content": f"Sorry, an unexpected error occurred: {e}"})
#         return f"Unexpected Error: {e}"

# ----------------------------------- > NEW < -----------------------------------
def run_conversation(user_prompt):
    if not gemini_model or not llm_enabled:
        return f"LLM client ({llm_provider}) not available."

    # Gemini chat history needs specific structure: role, parts (list of content)
    # Convert our session_state.messages to Gemini format
    gemini_history = []
    for msg in st.session_state.messages:
        role = "user" if msg.get("role") == "user" else "model" # Gemini uses "model" for assistant
        parts = []
        if msg.get("content"): # Standard text content
            # Ensure content is in the [{"text": "..."}] structure for Gemini
            if isinstance(msg.get("content"), list) and msg.get("content") and "text" in msg.get("content")[0]:
                parts = msg.get("content")
            else: # If it's just a string, wrap it
                parts = [{"text": str(msg.get("content"))}]

        elif msg.get("tool_calls"): # Assistant's request to call tools (OpenAI format)
            # Convert OpenAI tool_calls to Gemini FunctionCall parts
            for tc in msg.get("tool_calls"):
                parts.append({"function_call": {"name": tc.function.name, "args": json.loads(tc.function.arguments)}})

        elif msg.get("role") == "tool": # Result from a tool call
            # Convert our tool result to Gemini FunctionResponse parts
            parts.append({"function_response": {"name": msg.get("name"), "response": {"result": msg.get("content")}}}) # Gemini expects response in a specific "result" field sometimes, or direct

        if parts: # Only add if there's something to add
            gemini_history.append({"role": role, "parts": parts})


    # Start a new chat session with Gemini (or use an existing one if managing longer convos)
    # For simplicity here, we re-initialize chat for each turn with history.
    # More advanced use might keep the `chat` object alive.
    chat = gemini_model.start_chat(history=gemini_history)

    # Add current user prompt to session state (using our original dict format for display consistency)
    st.session_state.messages.append({"role": "user", "content": [{"text": user_prompt}]}) # Store in Gemini format for history

    try:
        # Configure the client and tools
        tools = types.Tool(function_declarations=function_declarations)
        config = types.GenerateContentConfig(tools=[tools])

        # Send request with function declarations
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=user_prompt,
            config=config,
        )

        # --- Process Gemini's Response ---
        response_message = response.candidates[0].content # Gemini Content object
        final_text_response = None

        # Check for function calls requested by Gemini
        if response_message.parts and response_message.parts[0].function_call:
            fc = response_message.parts[0].function_call
            function_name = fc.name
            function_args = dict(fc.args) # fc.args is a Struct, convert to dict

            # Store Gemini's intent to call function (convert to our display format)
            # This part needs careful mapping if we want to reuse OpenAI-style display
            st.session_state.messages.append({
                "role": "assistant", # Our display role
                "content": None, # Gemini might not have text if it's just a function call
                "gemini_function_call": {"name": function_name, "args": function_args} # Store for display
            })

            # --- Execute the function ---
            function_to_call = available_functions.get(function_name)
            function_response_json_str = None

            # --- Pre-call checks (similar to before) ---
            if not function_to_call:
                function_response_json_str = json.dumps({"status": "error", "message": f"Tool '{function_name}' not found."})
            else:
                # Pre-call check for add_inventory_item's required args
                if function_name == "add_inventory_item":
                    name = function_args.get("item_name")
                    qty = function_args.get("quantity")
                    prc = function_args.get("price")
                    missing_keys = []
                    if name is None: missing_keys.append("item_name")
                    if qty is None: missing_keys.append("quantity")
                    if prc is None: missing_keys.append("price")
                    if missing_keys:
                        details_needed = ', '.join(missing_keys).replace("item_name","name")
                        user_facing_request = f"Okay, I can try to add '{name if name else '(item)'}', but I need more details. Please provide: {details_needed}."
                        function_response_json_str = json.dumps({"status": "error_user_input_required", "message": user_facing_request })
                    else:
                        function_response_json_str = function_to_call(item_name=name, quantity=qty, price=prc)
                # Pre-call check for update_inventory_item
                elif function_name == "update_inventory_item":
                    identifier = function_args.get("item_identifier")
                    new_name = function_args.get("new_name")
                    new_qty = function_args.get("new_quantity")
                    new_prc = function_args.get("new_price")
                    if not identifier: function_response_json_str = json.dumps({"status": "error", "message": "Missing 'item_identifier' for update."})
                    elif new_name is None and new_qty is None and new_prc is None: function_response_json_str = json.dumps({"status": "error", "message": "No update field specified. Ask user what to change."})
                    else: function_response_json_str = function_to_call(item_identifier=identifier, new_name=new_name, new_quantity=new_qty, new_price=new_prc)
                # Other functions
                else:
                    try:
                        function_response_json_str = function_to_call(**function_args) # Call with unpacked args
                    except TypeError as te: # Handles case where function doesn't expect certain args (e.g. summary)
                        if not function_args: # If no args were passed, and it's expected (like summary)
                            function_response_json_str = function_to_call()
                        else:
                            function_response_json_str = json.dumps({"status":"error", "message":f"Argument mismatch for {function_name}: {te}"})
                    except Exception as e_exec:
                        function_response_json_str = json.dumps({"status":"error", "message":f"Error executing {function_name}: {e_exec}"})


            # Send function response back to Gemini
            # Gemini expects the result in a specific field, often 'result' or just the value.
            # Our functions return JSON strings, which is fine.
            st.session_state.messages.append({
                "role": "tool", # Our display role
                "name": function_name,
                "content": function_response_json_str
            })

            # Send the function response back to the model.
            # Gemini API requires sending a list of Parts, one of which is the FunctionResponse.
            response = chat.send_message(
                # Content part should be a genai.types.Part
                [genai.types.Part(function_response={"name": function_name, "response": {"result": function_response_json_str}})],
            )
            # The model should now generate a text response based on the tool's output
            if response.candidates and response.candidates[0].content.parts:
                final_text_response = response.candidates[0].content.parts[0].text
            else:
                final_text_response = "Tool executed, but I didn't get a follow-up text response."

        else: # No function call, direct text response from Gemini
            if response_message.parts:
                final_text_response = response_message.parts[0].text
            else:
                final_text_response = "I received a response, but it had no text content."

        # Store Gemini's final text response (for display)
        st.session_state.messages.append({"role": "assistant", "content": [{"text": final_text_response}]}) # Gemini format for history
        return final_text_response

    except Exception as e:
        error_msg = f"Error with {llm_provider}: {type(e).__name__} - {e}"
        # print(f"Full Gemini Error: {e}") # For detailed local debugging
        st.error(error_msg, icon="🚨")
        # Ensure the last user message isn't duplicated on retry due to error
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user" and st.session_state.messages[-1]["content"][0]["text"] == user_prompt:
             # This check is a bit complex, might need refinement
             pass # Don't pop if it's the current one, only if a previous try failed
        st.session_state.messages.append({"role": "assistant", "content": [{"text": f"Sorry, an error occurred: {e}"}]})
        return f"Error: {e}"


# --- Streamlit UI Layout ---
st.sidebar.header("Actions")

# --- Optional: Add Persistence Section (Example) --- (Keep as before)
# ...


# --- Sidebar ---
with st.sidebar:
    # st.title("⚙️ Controls & Info")
    # st.divider()

    # --- LLM Connection Status ---
    st.subheader("LLM Status")
    if llm_enabled:
        st.success(f"Connected via {llm_provider}", icon="✅")
    else:
        # Check specific errors if client init failed
        error_message = "LLM connection failed. "
        try:
            # Attempt to access secrets again to give specific feedback
            st.secrets["GITHUB_TOKEN"]
            # If key exists, error must be connection/auth
            error_message += f"Check API Key validity or {llm_provider} status."
        except KeyError:
             error_message += f"`{llm_provider}_API_KEY` missing in secrets."
        except Exception as e:
            error_message += f"Unexpected init error: {e}"
        st.error(error_message, icon="⚠️")
    st.caption(f"Referrer URL: {OPENROUTER_REFERRER_URL}")
    st.divider()

    # --- Navigation ---
    st.subheader("Navigation")
    page_options = ["📊 View Inventory", "📝 Manage Items", "💬 Chatbot", "📄 OCR Process"]
    selected_page = st.radio(
        "Go to:",
        page_options,
        label_visibility="collapsed" # Hide the 'Go to:' label itself
    )
    st.divider()

    # --- Chat Controls ---
    st.subheader("Chat Controls")
    if st.button("Clear Chat History", key="clear_chat"):
        reset_chat_history()
        st.success("Chat history cleared.")
        st.rerun() # Rerun to reflect the cleared chat immediately

    # --- Go to Top Button ---
    st.divider()
    st.caption("AI Inventory Manager v1.1")


# --- Main Page Content (Conditional based on Sidebar Selection) ---
if selected_page == "📊 View Inventory":
    st.header("📊 Current Inventory Status")
    st.dataframe(get_inventory_df(), use_container_width=True, hide_index=True)
    if st.button("Refresh View", key="refresh_view"):
        st.rerun()

elif selected_page == "📝 Manage Items":
    st.header("Manage Inventory Items")
    st.info("Add, update, or delete items from the inventory list.", icon="ℹ️")
    col_add, col_manage = st.columns(2)
    # Add Form... (Keep as before)
    with col_add:
        st.subheader("➕ Add New Item")
        with st.form("add_item_form", clear_on_submit=True):
            new_name = st.text_input("Item Name*")
            new_quantity = st.number_input("Quantity*", min_value=0, step=1, value=0)
            new_price = st.number_input("Price (per unit)*", min_value=0.00, step=0.01, value=0.00, format="%.2f")
            submitted_add = st.form_submit_button("Add Item")
            if submitted_add:
                if not new_name or new_quantity is None or new_price is None: st.warning("Please fill in all required fields (*).")
                else:
                    new_id = generate_item_id(); st.session_state.inventory[new_id] = {"name": new_name.strip(), "quantity": int(new_quantity), "price": float(new_price), "last_updated": datetime.datetime.now()}
                    st.success(f"✅ Item '{new_name}' ({new_id}) added successfully!"); st.rerun()
    # Update/Delete Form... (Keep as before)
    with col_manage:
        st.subheader("✏️ Update / 🗑️ Delete Item")
        if not st.session_state.inventory: st.info("Inventory is empty. Add items first.")
        else:
            item_options = [(f"{details.get('name', 'N/A')} ({item_id})", item_id) for item_id, details in st.session_state.inventory.items()]; item_options.sort(); item_options.insert(0, ("-- Select Item --", None))
            selected_option = st.selectbox("Select Item to Manage", options=item_options, format_func=lambda option: option[0], key="manage_select"); selected_id = selected_option[1]
            if selected_id:
                item = st.session_state.inventory.get(selected_id)
                if item:
                    with st.form(f"update_delete_{selected_id}_form"):
                        st.write(f"**Managing:** {item.get('name', 'N/A')} ({selected_id})"); update_name = st.text_input("Item Name*", value=item.get('name', '')); update_quantity = st.number_input("Quantity*", min_value=0, step=1, value=item.get('quantity', 0)); update_price = st.number_input("Price*", min_value=0.00, step=0.01, format="%.2f", value=item.get('price', 0.00))
                        update_col, delete_col = st.columns(2)
                        with update_col: submitted_update = st.form_submit_button("Update Item")
                        with delete_col: submitted_delete = st.form_submit_button("Delete Item", type="primary")
                        if submitted_update:
                            if not update_name or update_quantity is None or update_price is None: st.warning("Please ensure all fields have valid values (*).")
                            else: st.session_state.inventory[selected_id] = {"name": update_name.strip(), "quantity": int(update_quantity), "price": float(update_price), "last_updated": datetime.datetime.now()}; st.success(f"✅ Item '{update_name}' ({selected_id}) updated!"); st.rerun()
                        if submitted_delete:
                            deleted_name = st.session_state.inventory.get(selected_id, {}).get('name', 'Unknown')
                            if selected_id in st.session_state.inventory: del st.session_state.inventory[selected_id]; st.success(f"🗑️ Item '{deleted_name}' ({selected_id}) deleted!"); st.rerun()
                            else: st.warning(f"Item {selected_id} was already deleted."); st.rerun()
                else: st.warning(f"Item {selected_id} no longer seems to exist. Refreshing list.")

elif selected_page == "💬 Chatbot":
    st.header(f"💬 Chat with Inventory Bot ({llm_provider})")

    if not llm_enabled:
        st.warning(f"LLM client ({llm_provider}) failed to initialize. Chatbot functionality is disabled. Check secrets.", icon="⚠️")
    else:
        st.info("Ask questions about inventory status, item details, or low stock.", icon="💡")

        # **FIXED:** Display chat history (robustly handling dictionary structure)
        for i, message in enumerate(st.session_state.messages):
            role = message.get("role", "unknown") # Safely get role
            with st.chat_message(role):
                # --- Tool Result Message ---
                if role == "tool":
                    tool_name = message.get('name', 'unknown_function')
                    tool_content = message.get('content', '{}')
                    st.markdown(f"🛠️ **Function Result (`{tool_name}`)**")
                    try: # Try to pretty-print JSON content
                        parsed_content = json.loads(tool_content)
                        st.json(parsed_content)
                    except json.JSONDecodeError: # If not valid JSON, show as plain text/code
                        st.code(tool_content, language=None)

                # --- Assistant Message Requesting Tool Calls ---
                elif message.get("tool_calls"):
                    # Display any textual content that might accompany the tool call request
                    if message.get("content"):
                        st.markdown(message.get("content"))
                    # Display the requested tool calls
                    calls = message.get("tool_calls", []) # Default to empty list
                    if calls: # Check if list is not empty and has items
                        st.markdown("```tool_code") # Use a custom language for potential styling
                        for tc in calls:
                            # Safely access nested attributes using getattr
                            func = getattr(tc, 'function', object()) # Get function object safely
                            func_name = getattr(func, 'name', 'unknown')
                            func_args = getattr(func, 'arguments', '{}')
                            st.text(f"Function: {func_name}\nArgs: {func_args}")
                        st.markdown("```")

                # --- Regular User or Assistant Text Message ---
                elif message.get("content"):
                    st.markdown(message.get("content"))

                # --- Fallback for messages with unexpected structure ---
                else:
                    st.write(f"*(Message with role '{role}' has no displayable content)*")


        # Accept user input
        if prompt := st.chat_input("Ask about inventory (e.g., 'low stock', 'details of Laptop')..."):
            # Display user message immediately
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display thinking spinner and then the response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("🤔 Thinking..."):
                    # Get LLM response (handles function calling)
                    full_response = run_conversation(prompt)
                    message_placeholder.markdown(full_response or "*Assistant did not generate a response.*")

            # Automatically scroll chat to bottom (often needed after adding messages)
            # This JS is a bit hacky but common in Streamlit for chat
            st.components.v1.html("""
                <script>
                // Find the container holding the chat messages by data-testid
                const chatContainer = window.parent.document.querySelector('div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"]');
                // Alternative selector if the above doesn't work (inspect element to find a reliable parent)
                // const chatContainer = window.parent.document.querySelector('section.main > div.block-container');

                if (chatContainer) {
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
                </script>
            """, height=0) # Height 0 so the component itself isn't visible

            

    # Inside the `elif selected_page == "💬 Chatbot":` block:
    # --- Floating Go To Top Button (Chatbot Page Only - Robust JS Approach) ---

    button_html = """
        <style>
            #scrollTopBtn {
                position: fixed; /* Fixed/sticky position */
                bottom: 20px; /* Place the button 20px from the bottom */
                right: 30px; /* Place the button 30px from the right */
                z-index: 99; /* Make sure it does not overlap other elements */
                border: none; /* Remove borders */
                outline: none; /* Remove outline */
                background-color: #555; /* Set a background color */
                color: white; /* Text color */
                cursor: pointer; /* Add a mouse pointer on hover */
                padding: 15px; /* Some padding */
                border-radius: 10px; /* Rounded corners */
                font-size: 18px; /* Increase font size */
                display: none; /* Hidden by default */
            }

            #scrollTopBtn:hover {
                background-color: #007bff; /* Add a darker background on hover */
            }
        </style>

        <button onclick="scrollToTop()" id="scrollTopBtn" title="Go to top">Top</button>

        <script>
            // Get the button
            var mybutton = document.getElementById("scrollTopBtn");

            // When the user scrolls down 100px from the top of the document, show the button
            window.onscroll = function() {scrollFunction()};

            function scrollFunction() {
                // Use document.documentElement.scrollTop for compatibility
                if (document.body.scrollTop > 100 || document.documentElement.scrollTop > 100) {
                    mybutton.style.display = "block";
                } else {
                    mybutton.style.display = "none";
                }
            }

            // When the user clicks on the button, scroll to the top of the document
            function scrollToTop() {
                // Use smooth scrolling
                window.scrollTo({top: 0, behavior: 'smooth'});
            }
        </script>
        """

    # Inject the HTML/CSS/JS into the Streamlit app
    st.markdown(button_html, unsafe_allow_html=True)


elif selected_page == "📄 OCR Process":
    st.header("📄 Process Invoice/Bill via OCR")
    st.info("Upload an image of an invoice, bill, or stock list to extract text and attempt to add items to inventory.", icon="📷")

    uploaded_file = st.file_uploader("Choose an image file (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)
            st.divider()

            if st.button("🔍 Extract Text from Image", key="extract_text_ocr"):
                with st.spinner("Processing image with OCR..."):
                    try:
                        # Perform OCR
                        extracted_text = pytesseract.image_to_string(image)
                        st.session_state.ocr_text = extracted_text # Store in session state
                        st.success("Text extracted successfully!")
                    except pytesseract.TesseractNotFoundError:
                        st.error("Tesseract OCR engine not found. Please ensure it's installed and in your PATH.")
                        st.error("See: https://tesseract-ocr.github.io/tessdoc/Installation.html")
                        st.session_state.ocr_text = None
                    except Exception as e:
                        st.error(f"An error occurred during OCR: {e}")
                        st.session_state.ocr_text = None

        except Exception as e:
            st.error(f"Error opening or displaying image: {e}")
            st.session_state.ocr_text = None # Clear any previous OCR text on image error

    # --- Display Extracted Text and Parsing Options ---
    if 'ocr_text' in st.session_state and st.session_state.ocr_text:
        st.subheader("Extracted Text:")
        st.text_area("OCR Output", st.session_state.ocr_text, height=300)
        st.divider()
        st.subheader("Process Extracted Text")
        st.write("Use the extracted text to update inventory. This step often requires an LLM to interpret the unstructured text.")

        if st.button("🤖 Attempt to Add/Update Inventory using LLM", key="process_ocr_llm"):
            if not llm_enabled or not client:
                st.warning("LLM client is not available. Cannot process text with LLM.")
            else:
                ocr_prompt = f"""
                The following text was extracted from an image using OCR. It likely represents an invoice, bill, or stock list.
                Your task is to identify items, their quantities, and their prices from this text.
                For each item found, you should call the 'add_inventory_item' function with its name, quantity, and price.
                If an item seems to already exist (e.g., the text implies an update to an existing item's quantity or price),
                you should call the 'update_inventory_item' function, providing the item_identifier and the new quantity or price.
                Prioritize 'add_inventory_item' for new entries. Be careful with parsing; quantities and prices must be numbers.
                If information is ambiguous or missing for an item, skip it rather than guessing. Just provide the items once, currently you are providing multiple times.

                OCR Text:
                ---
                {st.session_state.ocr_text}
                ---

                Based on this text, what inventory actions (add or update) should be taken?
                """
                with st.spinner("🤖 LLM is processing OCR text to identify inventory actions..."):
                    # The run_conversation function will handle the tool calls (add/update)
                    # We are essentially using the LLM as the "user" driving the tool calls
                    # based on the OCR text.
                    llm_interpretation_response = run_conversation(ocr_prompt)

                    # Display LLM's summary of actions or final interpretation
                    st.markdown("#### LLM Processing Summary:")
                    st.info(llm_interpretation_response or "LLM did not provide a summary of actions.")
                    st.success("Inventory update attempt finished. Please check the 'View Inventory' page.")
                    # Rerun to reflect any inventory changes in other views
                    # st.rerun()

            # Define the CSS styles separately for clarity
#             button_css = """
#                 <style>
#                     #stGoToTopBtn {
#  /* Hidden by default */
#                         position: fixed;
#                         bottom: 500px; /* Slightly higher */
#                         right: 500px;
#                         z-index: 1001; /* Ensure high z-index */
#                         border: none;
#                         outline: none;
#                         background-color: #007bff; /* Use a primary color */
#                         color: white;
#                         cursor: pointer;
#                         padding: 12px 16px; /* Slightly larger padding */
#                         border-radius: 50%; /* Make it round */
#                         font-size: 20px; /* Larger icon */
#                         line-height: 1; /* Ensure icon is centered vertically */
#                         box-shadow: 0 2px 5px rgba(0,0,0,0.2); /* Add subtle shadow */
#                         opacity: 0.8;
#                         transition: opacity 0.3s, background-color 0.3s;
#                     }

#                     #stGoToTopBtn:hover {
#                         background-color: #0056b3; /* Darker shade on hover */
#                         opacity: 1;
#                     }
#                 </style>
#                 """

#             # Define the JavaScript to create, manage, and handle the button
#             button_js = """
#                 <script>
#                     // Function to create the button if it doesn't exist
#                     function createGoToTopButton() {
#                         if (document.getElementById("stGoToTopBtn")) {
#                             // Button already exists, ensure event listeners are attached
#                             attachScrollListener();
#                             return;
#                         }

#                         const btn = document.createElement("button");
#                         btn.innerHTML = "⬆️"; // Use just the arrow for a round button
#                         btn.id = "stGoToTopBtn";
#                         btn.title = "Go to top";
#                         btn.onclick = scrollToTopFunction;
#                         document.body.appendChild(btn); // Append to body

#                         console.log("Go To Top button created.");
#                         attachScrollListener(); // Attach listener after creation
#                     }

#                     // Function to scroll to top
#                     function scrollToTopFunction() {
#                         try {
#                             // Target the main scrollable container in Streamlit (more specific)
#                             const mainScrollable = window.parent.document.querySelector('section[data-testid="stAppViewContainer"] > section'); // Common Streamlit structure
#                             if (mainScrollable) {
#                                 mainScrollable.scrollTo({top: 0, behavior: 'smooth'});
#                                 console.log("Scrolled specific container to top.");
#                             } else {
#                                 // Fallback to window scroll if specific container not found
#                                 window.scrollTo({top: 0, behavior: 'smooth'});
#                                 console.log("Scrolled window to top (fallback).");
#                             }
#                         } catch (e) {
#                             console.error("Error finding scrollable element or scrolling:", e);
#                             // Absolute fallback
#                             window.scrollTo(0, 0);
#                         }
#                     }

#                     // Function to show/hide button based on scroll position
#                     function handleScroll() {
#                         const btn = document.getElementById("stGoToTopBtn");
#                         if (!btn) return; // Exit if button doesn't exist

#                         let scrollPos = 0;
#                         try {
#                             // Try to get scroll position from the specific container
#                             const mainScrollable = window.parent.document.querySelector('section[data-testid="stAppViewContainer"] > section');
#                             if (mainScrollable) {
#                                 scrollPos = mainScrollable.scrollTop;
#                             } else {
#                                 // Fallback scroll position check
#                                 scrollPos = document.body.scrollTop || document.documentElement.scrollTop;
#                             }
#                         } catch (e) {
#                             // Fallback if accessing parent/querySelector fails
#                             scrollPos = document.body.scrollTop || document.documentElement.scrollTop;
#                         }


#                         if (scrollPos > 200) { // Show after scrolling a bit more
#                             btn.style.display = "block";
#                         } else {
#                             btn.style.display = "none";
#                         }
#                     }

#                     // Function to attach the scroll listener to the correct element
#                     function attachScrollListener() {
#                         try {
#                             // Attempt to attach listener to the specific scrollable container
#                             const mainScrollable = window.parent.document.querySelector('section[data-testid="stAppViewContainer"] > section');
#                             if (mainScrollable) {
#                                 mainScrollable.onscroll = handleScroll;
#                                 console.log("Attached scroll listener to specific container.");
#                                 // Initial check in case page is already scrolled
#                                 handleScroll();
#                                 return; // Success
#                             }
#                         } catch (e) {
#                             console.warn("Could not attach scroll listener to specific container, using window.", e);
#                         }

#                         // Fallback: Attach listener to the window
#                         window.onscroll = handleScroll;
#                         console.log("Attached scroll listener to window (fallback).");
#                         // Initial check
#                         handleScroll();
#                     }

#                     // --- Execution ---
#                     // Use DOMContentLoaded to ensure the body exists before appending
#                     if (document.readyState === "complete" || document.readyState === "interactive") {
#                         // DOM already loaded
#                         createGoToTopButton();
#                     } else {
#                         // Wait for the DOM to load
#                         document.addEventListener("DOMContentLoaded", createGoToTopButton);
#                     }

#                 </script>
#                 """

            # Embed the CSS and JS into the Streamlit page
            # st.components.v1.html(button_css + button_js, height=500, scrolling=False)
            # st.components.v1.html("THIS IS THE HTML CONTENT TO BE BUTTONED", height=500, scrolling=False)


# Default case (shouldn't happen with radio buttons but good practice)
else:
    st.error("Invalid page selected.")