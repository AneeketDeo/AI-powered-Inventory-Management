
---

# AI-powered Inventory Management

## Overview

This project is a Streamlit-based web application for managing inventory with AI-powered features. It provides:
- Inventory viewing and management (add, update, delete items)
- A chatbot interface for querying inventory using natural language (with LLM integration)
- OCR-based extraction of inventory data from images (e.g., invoices, bills)
- In-memory inventory storage (can be extended for persistence)

---

## Features

### 1. Inventory Management
- View all inventory items in a table with details: Item ID, Name, Quantity, Price, Last Updated.
- Add new items with unique IDs, quantity, and price.
- Update or delete existing items.
- Inventory is stored in Streamlit session state as a dictionary.

### 2. Chatbot (AI Assistant)
- Ask questions about inventory (e.g., "Show low stock", "Details of Laptop").
- Uses an LLM (OpenAI-compatible, e.g., GPT-4o) for natural language understanding and function calling.
- Supports tool calls for:
  - Inventory summary
  - Item details lookup
  - Low stock detection
  - Adding/updating items

### 3. OCR Processing
- Upload an image (invoice, bill, etc.) and extract text using Tesseract OCR.
- LLM interprets extracted text to suggest inventory actions (add/update items).

---

## File Structure

- streamlit_main.py: Main Streamlit app with all features (LLM, OCR, inventory management).
- streamlit.py: Simpler version of the app with rule-based chatbot (no LLM/OCR).
- requirements.txt / packages.txt: List of required Python packages.

---

## Key Components

### Inventory Data Structure

```python
{
  "ITEM001": {
    "name": "Laptop",
    "quantity": 15,
    "price": 1200.00,
    "last_updated": datetime.datetime
  },
  ...
}
```

### Main Functions

- `get_inventory_df()`: Converts inventory dict to a formatted DataFrame for display.
- `generate_item_id()`: Generates a unique item ID.
- `add_inventory_item()`: Adds a new item (with validation).
- `update_inventory_item()`: Updates an existing item (by name or ID).
- `get_inventory_summary()`: Returns a summary (total items, total quantity).
- `get_item_details()`: Looks up item details by name or ID.
- `find_low_stock_items()`: Lists items below a quantity threshold.

### LLM Integration

- Uses OpenAI-compatible API for chat and function calling.
- Functions are exposed as "tools" for the LLM to call.
- Handles tool call results and conversational flow.

### OCR Integration

- Uses `pytesseract` and `Pillow` to extract text from uploaded images.
- LLM processes extracted text to identify inventory actions.

---

## Usage

1. **Install dependencies** (from requirements.txt).
2. **Set up Streamlit secrets** for LLM API keys (see code comments for details).
3. **Run the app**:
   ```powershell
   streamlit run streamlit_main.py
   ```
4. **Navigate** using the sidebar:
   - View Inventory
   - Manage Items
   - Chatbot
   - OCR Process

---

## Customization

- Extend inventory persistence (e.g., save/load from CSV or database).
- Add more LLM tools or improve prompt engineering.
- Enhance OCR post-processing for better item extraction.

---
