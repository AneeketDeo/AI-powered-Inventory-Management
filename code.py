import streamlit as st
import pandas as pd
import datetime
import random # For generating sample IDs

# --- Configuration ---
st.set_page_config(page_title="Inventory Manager & Chatbot", layout="wide")
st.title("📦 Inventory Management System with Chatbot")

# --- Session State Initialization ---
# Initialize inventory data (using a dictionary for this example)
# Structure: {item_id: {'name': str, 'quantity': int, 'price': float, 'last_updated': datetime}}
if 'inventory' not in st.session_state:
    # Sample Data
    st.session_state.inventory = {
        "ITEM001": {"name": "Laptop", "quantity": 15, "price": 1200.00, "last_updated": datetime.datetime.now()},
        "ITEM002": {"name": "Keyboard", "quantity": 50, "price": 75.00, "last_updated": datetime.datetime.now()},
        "ITEM003": {"name": "Mouse", "quantity": 45, "price": 25.50, "last_updated": datetime.datetime.now()},
        "ITEM004": {"name": "Monitor", "quantity": 10, "price": 300.00, "last_updated": datetime.datetime.now()},
    }

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me about the inventory."}]

# --- Helper Functions ---
def get_inventory_df():
    """Converts the inventory dictionary to a Pandas DataFrame."""
    if not st.session_state.inventory:
        return pd.DataFrame(columns=["Item ID", "Name", "Quantity", "Price", "Last Updated"])
    # Convert dict to list of dicts with 'Item ID' included
    data_list = [{'Item ID': k, **v} for k, v in st.session_state.inventory.items()]
    df = pd.DataFrame(data_list)
    # Format columns for better display
    df['Price'] = df['Price'].map('${:,.2f}'.format)
    df['Last Updated'] = pd.to_datetime(df['last_updated']).dt.strftime('%Y-%m-%d %H:%M:%S')
    return df[["Item ID", "Name", "Quantity", "Price", "Last Updated"]]

def generate_item_id():
    """Generates a unique item ID."""
    prefix = "ITEM"
    while True:
        new_id = f"{prefix}{random.randint(100, 999)}"
        if new_id not in st.session_state.inventory:
            return new_id

# --- Inventory Management Functions (Chatbot Backend) ---
def chatbot_get_inventory_summary():
    """Provides a summary of the inventory for the chatbot."""
    count = len(st.session_state.inventory)
    total_quantity = sum(item['quantity'] for item in st.session_state.inventory.values())
    return f"There are {count} different items in stock, with a total quantity of {total_quantity} units."

def chatbot_get_item_details(item_name_or_id):
    """Finds details for a specific item by name or ID."""
    item_name_or_id = item_name_or_id.strip().upper()
    # Check by ID first
    if item_name_or_id in st.session_state.inventory:
        item = st.session_state.inventory[item_name_or_id]
        return f"Details for {item_name_or_id} ({item['name']}): Quantity={item['quantity']}, Price=${item['price']:.2f}."

    # Check by name (case-insensitive)
    for item_id, details in st.session_state.inventory.items():
        if details['name'].strip().upper() == item_name_or_id:
            return f"Details for {details['name']} ({item_id}): Quantity={details['quantity']}, Price=${details['price']:.2f}."

    return f"Sorry, I couldn't find an item with the name or ID '{item_name_or_id}'."

def chatbot_find_low_stock(threshold=10):
    """Finds items with quantity below a threshold."""
    low_stock_items = []
    for item_id, details in st.session_state.inventory.items():
        if details['quantity'] <= threshold:
            low_stock_items.append(f"- {details['name']} ({item_id}): {details['quantity']} units")

    if not low_stock_items:
        return f"No items found with stock at or below {threshold} units."
    else:
        return f"Items with low stock (<= {threshold} units):\n" + "\n".join(low_stock_items)

# --- SIMULATED Chatbot Response Logic ---
# In a real app, this would call an LLM (like OpenAI, Anthropic, Gemini)
# potentially using function calling / tool usage.
def get_chatbot_response(user_query):
    """Generates a response based on the user query (Rule-based)."""
    query_lower = user_query.lower()

    if "how many items" in query_lower or "summary" in query_lower or "overview" in query_lower:
        return chatbot_get_inventory_summary()
    elif "details of" in query_lower or "tell me about" in query_lower or "quantity of" in query_lower or "price of" in query_lower:
        # Try to extract item name/ID (simple extraction)
        parts = query_lower.split(" of ")
        if len(parts) > 1:
            item_name_or_id = parts[-1].replace("?","").strip()
            return chatbot_get_item_details(item_name_or_id)
        else:
             return "Please specify which item you want details for, e.g., 'details of Laptop' or 'details of ITEM001'."
    elif "low stock" in query_lower:
        # Check if a specific threshold is mentioned
        try:
             # Very basic number extraction
             words = query_lower.split()
             threshold_idx = words.index("below") if "below" in words else (words.index("under") if "under" in words else -1)
             if threshold_idx != -1 and threshold_idx + 1 < len(words) and words[threshold_idx+1].isdigit():
                 threshold = int(words[threshold_idx+1])
                 return chatbot_find_low_stock(threshold)
             else:
                 return chatbot_find_low_stock() # Use default
        except:
             return chatbot_find_low_stock() # Use default on error
    elif "list all" in query_lower or "show all" in query_lower:
         items = [f"- {details['name']} ({item_id}): {details['quantity']} @ ${details['price']:.2f}" for item_id, details in st.session_state.inventory.items()]
         if not items:
             return "The inventory is currently empty."
         return "Here are all the items in stock:\n" + "\n".join(items)

    elif "hello" in query_lower or "hi" in query_lower:
        return "Hello! How can I help you with the inventory today?"
    else:
        return "Sorry, I didn't quite understand that. You can ask me for:\n- Inventory summary\n- Details of a specific item (e.g., 'details of Laptop')\n- Low stock items (e.g., 'show low stock below 15')\n- List all items"


# --- Streamlit UI Layout ---
tab1, tab2, tab3 = st.tabs(["📊 View Inventory", "📝 Manage Items", "💬 Chatbot"])

# --- Tab 1: View Inventory ---
with tab1:
    st.header("Current Inventory")
    st.dataframe(get_inventory_df(), use_container_width=True, hide_index=True)
    # Optional: Add search/filter here later

# --- Tab 2: Manage Items ---
with tab2:
    st.header("Manage Inventory Items")

    col1, col2 = st.columns(2)

    # Add New Item Form
    with col1:
        st.subheader("➕ Add New Item")
        with st.form("add_item_form", clear_on_submit=True):
            new_name = st.text_input("Item Name*", key="add_name")
            new_quantity = st.number_input("Quantity*", min_value=0, step=1, key="add_qty")
            new_price = st.number_input("Price (per unit)*", min_value=0.0, step=0.01, format="%.2f", key="add_price")
            submitted_add = st.form_submit_button("Add Item")

            if submitted_add:
                if not new_name or new_quantity is None or new_price is None:
                    st.warning("Please fill in all required fields.")
                else:
                    new_id = generate_item_id()
                    st.session_state.inventory[new_id] = {
                        "name": new_name,
                        "quantity": new_quantity,
                        "price": new_price,
                        "last_updated": datetime.datetime.now()
                    }
                    st.success(f"✅ Item '{new_name}' ({new_id}) added successfully!")
                    # No need for rerun, form clear + success message is often enough,
                    # but rerun forces immediate update of table in other tab if viewed next.
                    # st.rerun() # Uncomment if you want immediate refresh everywhere


    # Update/Delete Item Form
    with col2:
        st.subheader("✏️ Update / 🗑️ Delete Item")
        if not st.session_state.inventory:
            st.info("No items available to manage.")
        else:
            item_ids = list(st.session_state.inventory.keys())
            # Prepend a placeholder
            item_ids_with_placeholder = ["-- Select Item ID --"] + item_ids
            selected_id = st.selectbox("Select Item ID to Manage", options=item_ids_with_placeholder, key="manage_select_id")

            if selected_id != "-- Select Item ID --":
                item = st.session_state.inventory[selected_id]

                with st.form("update_delete_form"):
                    st.write(f"**Managing Item:** {item['name']} ({selected_id})")
                    update_name = st.text_input("Item Name", value=item['name'], key="update_name")
                    update_quantity = st.number_input("Quantity", min_value=0, step=1, value=item['quantity'], key="update_qty")
                    update_price = st.number_input("Price", min_value=0.0, step=0.01, format="%.2f", value=item['price'], key="update_price")

                    col_update, col_delete = st.columns(2)
                    with col_update:
                         submitted_update = st.form_submit_button("Update Item")
                    with col_delete:
                        submitted_delete = st.form_submit_button("Delete Item", type="primary") # Make delete more prominent/dangerous


                    if submitted_update:
                        if not update_name or update_quantity is None or update_price is None:
                            st.warning("Please ensure all fields have valid values.")
                        else:
                             st.session_state.inventory[selected_id] = {
                                "name": update_name,
                                "quantity": update_quantity,
                                "price": update_price,
                                "last_updated": datetime.datetime.now()
                            }
                             st.success(f"✅ Item '{update_name}' ({selected_id}) updated successfully!")
                             # Clear selection after update to avoid accidental re-submission? Optional.
                             # st.session_state.manage_select_id = "-- Select Item ID --" # Does not work well with form state
                             st.rerun() # Rerun to reflect changes and reset form state implicitly

                    if submitted_delete:
                         # Optional: Add a confirmation step here
                         deleted_name = st.session_state.inventory[selected_id]['name']
                         del st.session_state.inventory[selected_id]
                         st.success(f"🗑️ Item '{deleted_name}' ({selected_id}) deleted successfully!")
                         # Clear selection
                         # st.session_state.manage_select_id = "-- Select Item ID --" # Does not work well with form state
                         st.rerun() # Rerun to reflect changes and reset form state implicitly


# --- Tab 3: Chatbot ---
with tab3:
    st.header("💬 Chat with Inventory Bot")

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask about your inventory... (e.g., 'low stock', 'details of ITEM001')"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                 # Get response from our rule-based "chatbot"
                 response = get_chatbot_response(prompt)
                 st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


# --- Optional: Add persistence ---
# You could add buttons here to save/load inventory from/to a CSV or JSON file
# Example (Conceptual):
# if st.sidebar.button("Save Inventory to CSV"):
#     df = pd.DataFrame.from_dict(st.session_state.inventory, orient='index')
#     df.to_csv("inventory.csv")
#     st.sidebar.success("Inventory saved!")

# if st.sidebar.button("Load Inventory from CSV"):
#      try:
#          df = pd.read_csv("inventory.csv", index_col=0)
#          # Convert back to required dictionary format, handle potential type issues
#          st.session_state.inventory = df.to_dict(orient='index') # Needs refinement for types/datetime
#          st.sidebar.success("Inventory loaded!")
#          st.rerun()
#      except FileNotFoundError:
#          st.sidebar.error("inventory.csv not found.")