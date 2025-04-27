# --- Core Libraries ---
import streamlit as st
import pandas as pd
import datetime
import random
import json

# --- LLM Interaction ---
import openai
from openai import OpenAI

# --- Configuration ---
st.set_page_config(
    page_title="AI Inventory Manager",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("📦 AI-Powered Inventory Management")

# --- OpenRouter Configuration ---
# Read the App URL from secrets - YOU MUST SET THIS in your Streamlit Cloud secrets!
# Example: APP_URL="https://your-app-name.streamlit.app"
# Use a generic placeholder if not set in secrets.
# OPENROUTER_REFERRER_URL = st.secrets.get("APP_URL", "https://your-inventory-app.streamlit.app/placeholder")
OPENROUTER_APP_TITLE = "Streamlit Inventory Chatbot" # Can customize this

# --- Initialize LLM Client (OpenRouter) ---
llm_provider = "None"
llm_enabled = False
client = None

try:
    openrouter_key = st.secrets["OPENROUTER_API_KEY"] # Must be set in secrets

    client = OpenAI(
        api_key=openrouter_key,
        base_url="https://openrouter.ai/api/v1",

    )
    # Quick check to validate credentials and connection during startup
    client.models.list()

    llm_provider = "OpenRouter"
    llm_enabled = True
    st.sidebar.success(f"Connected to {llm_provider}", icon="✅")
    # st.sidebar.caption(f"Using URL: {OPENROUTER_REFERRER_URL}") # Show URL being used

except KeyError:
    st.sidebar.error("`OPENROUTER_API_KEY` not found in Streamlit secrets. Chatbot disabled.", icon="⚠️")
except openai.AuthenticationError:
    st.sidebar.error(f"OpenRouter Authentication Error: Invalid API Key. Chatbot disabled.", icon="🚨")
except openai.APIConnectionError as e:
    st.sidebar.error(f"OpenRouter Connection Error. Chatbot disabled. Error: {e}", icon="🚨")
except Exception as e:
    st.sidebar.error(f"Error initializing OpenRouter: {e}. Chatbot disabled.", icon="⚠️")


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

# Chat History
if "messages" not in st.session_state:
    assistant_greeting = f"Hello! Ask me anything about the inventory (using {llm_provider})." if llm_enabled else "Hello! Inventory Bot here. LLM connection failed."
    st.session_state.messages = [{"role": "assistant", "content": assistant_greeting}]

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

# --- Define Tools for LLM ---
# Map tool names to actual Python functions
available_functions = {
    "get_inventory_summary": get_inventory_summary,
    "get_item_details": get_item_details,
    "find_low_stock_items": find_low_stock_items,
}

# Define tool structure for the LLM API call
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_inventory_summary",
            "description": "Get a summary of the inventory status: total distinct items and total quantity.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_item_details",
            "description": "Get details (quantity, price) for a specific item by its name or Item ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_identifier": {
                        "type": "string",
                        "description": "The name (e.g., 'Laptop', 'Keyboard') or Item ID (e.g., 'ITEM001') of the inventory item.",
                    },
                },
                "required": ["item_identifier"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_low_stock_items",
            "description": "Find items in the inventory that are low in stock, based on a quantity threshold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "quantity_threshold": {
                        "type": "integer",
                        "description": "The quantity threshold. Items with quantity at or below this value are considered low stock. Defaults to 10 if not specified by the user.",
                    },
                },
                "required": [], # Threshold is optional
            },
        },
    },
]

# --- LLM Interaction Logic ---
def run_conversation(user_prompt):
    """Sends conversation to OpenRouter, handles tool calls, returns final response."""
    if not client or not llm_enabled:
        return f"LLM client ({llm_provider}) not available. Cannot process request."

    # --- Choose Model ---
    # Ensure the chosen model supports tool/function calling on OpenRouter
    # Examples: "openai/gpt-3.5-turbo", "openai/gpt-4-turbo-preview", "anthropic/claude-3-haiku-20240307"
    # Check OpenRouter docs for model capabilities if unsure.
    model_name = "openai/gpt-3.5-turbo" # A reliable choice for OpenAI-style function calling
    # model_name = "anthropic/claude-3-haiku-20240307" # Good alternative, ensure testing

    # --- Prepare History ---
    # Add user message to the official history first
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # Prepare message list for the API call (can potentially trim history here if needed)
    messages_for_api = st.session_state.messages # Send full history for now

    try:
        # --- First API Call: Get response or tool request ---
        response = client.chat.completions.create(
            model=model_name,
            messages=messages_for_api,
            tools=tools,
            tool_choice="auto", # Let model decide whether to use a tool
        )
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls # Check if the model wants to call functions

        # --- Handle Tool Calls (if any) ---
        if tool_calls:
            # Append the assistant's response wanting to call tools
            st.session_state.messages.append(response_message) # Store the full message object

            # Execute tools and collect results
            tool_results_messages = []
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions.get(function_name)
                function_args_str = tool_call.function.arguments

                if not function_to_call:
                    function_response = json.dumps({"error": f"Tool '{function_name}' not found."})
                else:
                    try:
                        function_args = json.loads(function_args_str)
                        # Call function: Handle cases with and without specific args
                        if function_name == "find_low_stock_items":
                             # Use default if threshold not provided, else use provided value
                             threshold = function_args.get("quantity_threshold", 10) # Safely get arg or use default
                             function_response = function_to_call(quantity_threshold=threshold)
                        elif function_name == "get_item_details":
                             identifier = function_args.get("item_identifier")
                             if identifier:
                                 function_response = function_to_call(item_identifier=identifier)
                             else:
                                 function_response = json.dumps({"error": "Missing 'item_identifier' argument."})
                        elif function_name == "get_inventory_summary":
                             function_response = function_to_call() # No arguments needed
                        else:
                             # Should not happen if tool definition is correct
                             function_response = json.dumps({"error": f"Unhandled arguments for function {function_name}"})

                    except json.JSONDecodeError:
                        function_response = json.dumps({"error": f"Invalid arguments format from LLM for {function_name}: {function_args_str}"})
                    except Exception as e:
                        function_response = json.dumps({"error": f"Error executing {function_name}: {str(e)}"})

                # Prepare message for API with tool result
                tool_results_messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response, # Function output (JSON string)
                })

            # Add all tool results to the main message history
            st.session_state.messages.extend(tool_results_messages)

            # --- Second API Call: Send tool results back to LLM ---
            messages_for_second_call = st.session_state.messages # Send updated history
            second_response = client.chat.completions.create(
                model=model_name,
                messages=messages_for_second_call,
                # No tools needed here, we just want the final text response
            )
            final_response_content = second_response.choices[0].message.content
            # Append the final assistant text response
            st.session_state.messages.append({"role": "assistant", "content": final_response_content})
            return final_response_content

        # --- Handle Direct Response (No Tool Call) ---
        else:
            final_response_content = response_message.content
            # Append the direct assistant response
            st.session_state.messages.append({"role": "assistant", "content": final_response_content})
            return final_response_content

    # --- Error Handling ---
    except openai.APIError as e:
        error_msg = f"{llm_provider} API Error ({model_name}): {e}"
        st.error(error_msg, icon="🚨")
        st.session_state.messages.append({"role": "assistant", "content": f"Sorry, encountered an API error: {e}"})
        return f"API Error: {e}"
    except Exception as e:
        error_msg = f"Unexpected error during LLM interaction ({model_name}): {e}"
        st.error(error_msg, icon="🚨")
        # Attempt removal of last user message on error, if appropriate
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            st.session_state.messages.pop()
        st.session_state.messages.append({"role": "assistant", "content": f"Sorry, an unexpected error occurred: {e}"})
        return f"Unexpected Error: {e}"


# --- Streamlit UI Layout ---
st.sidebar.header("Actions")

# --- Tab Definitions ---
tab1, tab2, tab3 = st.tabs(["📊 View Inventory", "📝 Manage Items", "💬 Chatbot"])

# --- Tab 1: View Inventory ---
with tab1:
    st.header("Current Inventory Status")
    st.dataframe(get_inventory_df(), use_container_width=True, hide_index=True)
    if st.button("Refresh View", key="refresh_view"):
        st.rerun()

# --- Tab 2: Manage Items ---
with tab2:
    st.header("Manage Inventory Items")
    st.info("Add, update, or delete items from the inventory list.", icon="ℹ️")

    col_add, col_manage = st.columns(2)

    # Add New Item Form
    with col_add:
        st.subheader("➕ Add New Item")
        with st.form("add_item_form", clear_on_submit=True):
            new_name = st.text_input("Item Name*")
            new_quantity = st.number_input("Quantity*", min_value=0, step=1, value=0)
            new_price = st.number_input("Price (per unit)*", min_value=0.00, step=0.01, value=0.00, format="%.2f")
            submitted_add = st.form_submit_button("Add Item")

            if submitted_add:
                if not new_name or new_quantity is None or new_price is None:
                    st.warning("Please fill in all required fields (*).")
                else:
                    new_id = generate_item_id()
                    st.session_state.inventory[new_id] = {
                        "name": new_name.strip(),
                        "quantity": int(new_quantity),
                        "price": float(new_price),
                        "last_updated": datetime.datetime.now()
                    }
                    st.success(f"✅ Item '{new_name}' ({new_id}) added successfully!")
                    st.rerun() # Rerun to update views immediately

    # Update/Delete Item Section
    with col_manage:
        st.subheader("✏️ Update / 🗑️ Delete Item")
        if not st.session_state.inventory:
            st.info("Inventory is empty. Add items first.")
        else:
            # Create a list of tuples: (display name, item_id) for the selectbox
            item_options = [
                (f"{details.get('name', 'N/A')} ({item_id})", item_id)
                for item_id, details in st.session_state.inventory.items()
            ]
            item_options.sort() # Sort alphabetically by display name
            item_options.insert(0, ("-- Select Item --", None)) # Placeholder

            # Use format_func to display the name, but return the ID
            selected_option = st.selectbox(
                "Select Item to Manage",
                options=item_options,
                format_func=lambda option: option[0], # Display name part
                key="manage_select"
            )
            selected_id = selected_option[1] # Get the actual item ID

            if selected_id:
                item = st.session_state.inventory.get(selected_id)
                if item: # Check if item still exists (might be deleted in another session/tab)
                    with st.form(f"update_delete_{selected_id}_form"): # Unique form key per item
                        st.write(f"**Managing:** {item.get('name', 'N/A')} ({selected_id})")
                        update_name = st.text_input("Item Name*", value=item.get('name', ''))
                        update_quantity = st.number_input("Quantity*", min_value=0, step=1, value=item.get('quantity', 0))
                        update_price = st.number_input("Price*", min_value=0.00, step=0.01, format="%.2f", value=item.get('price', 0.00))

                        update_col, delete_col = st.columns(2)
                        with update_col:
                            submitted_update = st.form_submit_button("Update Item")
                        with delete_col:
                            submitted_delete = st.form_submit_button("Delete Item", type="primary")

                        if submitted_update:
                            if not update_name or update_quantity is None or update_price is None:
                                st.warning("Please ensure all fields have valid values (*).")
                            else:
                                st.session_state.inventory[selected_id] = {
                                    "name": update_name.strip(),
                                    "quantity": int(update_quantity),
                                    "price": float(update_price),
                                    "last_updated": datetime.datetime.now()
                                }
                                st.success(f"✅ Item '{update_name}' ({selected_id}) updated!")
                                st.rerun()

                        if submitted_delete:
                            deleted_name = st.session_state.inventory.get(selected_id, {}).get('name', 'Unknown')
                            if selected_id in st.session_state.inventory:
                                del st.session_state.inventory[selected_id]
                                st.success(f"🗑️ Item '{deleted_name}' ({selected_id}) deleted!")
                                st.rerun()
                            else:
                                st.warning(f"Item {selected_id} was already deleted.")
                                st.rerun()
                else:
                     st.warning(f"Item {selected_id} no longer seems to exist. Refreshing list.")
                     # Implicitly handled by rerun on next interaction or manual refresh

# --- Tab 3: Chatbot ---
with tab3:
    st.header(f"💬 Chat with Inventory Bot ({llm_provider})")

    if not llm_enabled:
        st.warning(f"LLM client ({llm_provider}) failed to initialize. Chatbot functionality is disabled. Check secrets.", icon="⚠️")
    else:
        st.info("Ask questions about inventory status, item details, or low stock.", icon="💡")

        # Display chat history
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                # Display different types of messages appropriately
                if message["role"] == "tool":
                     st.markdown(f"```json\n{message.get('content', '{}')}\n```", unsafe_allow_html=True) # Show tool JSON result in code block
                elif isinstance(message.get("content"), str):
                     st.markdown(message["content"])
                elif message.get("tool_calls"):
                     # Display assistant's intention to call function(s) more clearly
                     calls = message["tool_calls"]
                     call_descs = [f"`{tc.function.name}`" for tc in calls] # Just show names for brevity
                     st.markdown(f"*(Thinking: Need to use function(s): {', '.join(call_descs)})*")
                else: # Fallback for unexpected message format
                     st.write(message)

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

# --- Optional: Add Persistence Section (Example - requires more setup for cloud) ---
# with st.sidebar:
#     st.header("Persistence (Example)")
#     st.warning("Note: Saving/Loading to CSV is basic and may not work reliably on Streamlit Cloud's ephemeral filesystem without modification (e.g., using cloud storage or db).")
#     if st.button("Save Inventory (CSV)"):
#         try:
#             # Ensure data consistency before saving
#             df_to_save = pd.DataFrame.from_dict(st.session_state.inventory, orient='index')
#             # Convert datetime to string for CSV compatibility
#             if 'last_updated' in df_to_save.columns:
#                  df_to_save['last_updated'] = df_to_save['last_updated'].astype(str)
#             df_to_save.to_csv("inventory_snapshot.csv", index_label="ItemID")
#             st.success("Inventory saved to inventory_snapshot.csv")
#         except Exception as e:
#             st.error(f"Failed to save inventory: {e}")

#     if st.button("Load Inventory (CSV)"):
#         try:
#             df_loaded = pd.read_csv("inventory_snapshot.csv", index_col="ItemID")
#             # Convert back to required dictionary format, handling types
#             loaded_inventory = {}
#             for item_id, row in df_loaded.iterrows():
#                  item_data = row.to_dict()
#                  # Attempt to convert types back carefully
#                  item_data['quantity'] = int(item_data.get('quantity', 0))
#                  item_data['price'] = float(item_data.get('price', 0.0))
#                  try: # Handle datetime conversion errors
#                      item_data['last_updated'] = pd.to_datetime(item_data.get('last_updated')).to_pydatetime()
#                  except:
#                      item_data['last_updated'] = datetime.datetime.now() # Fallback
#                  loaded_inventory[item_id] = item_data

#             st.session_state.inventory = loaded_inventory
#             st.success("Inventory loaded from inventory_snapshot.csv")
#             st.rerun() # Refresh UI
#         except FileNotFoundError:
#             st.error("inventory_snapshot.csv not found.")
#         except Exception as e:
#             st.error(f"Failed to load inventory: {e}")