import fitz  # PyMuPDF
import re
import sqlite3
import os
from datetime import datetime

# Directory to save extracted logos
LOGO_DIR = "extracted_logos"
JOURNAL_LOGO_DIR = os.path.join(LOGO_DIR, "journal")
CLIENT_LOGO_DIR = os.path.join(LOGO_DIR, "client")

# Create the logo directories if they don't exist
if not os.path.exists(LOGO_DIR):
    os.makedirs(LOGO_DIR)
if not os.path.exists(JOURNAL_LOGO_DIR):
    os.makedirs(JOURNAL_LOGO_DIR)
if not os.path.exists(CLIENT_LOGO_DIR):
    os.makedirs(CLIENT_LOGO_DIR)

# Initialize SQLite database based on type (journal or client)
def init_db(db_type):
    if db_type == "journal":
        db_name = "journal_trademarks.db"
    elif db_type == "client":
        db_name = "client_trademarks.db"
    else:
        raise ValueError("Invalid db_type. Must be 'journal' or 'client'.")

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trademarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            journal_no TEXT,
            journal_date TEXT,
            class TEXT,
            application_number TEXT,
            application_date TEXT,
            company_name TEXT,
            address TEXT,
            entity_type TEXT,
            legal_status TEXT,
            address_for_service TEXT,
            usage_details TEXT,
            location_of_use TEXT,
            goods_services TEXT,
            additional_notes TEXT,
            logo_path TEXT,
            logo_placeholder TEXT,
            source_file TEXT,
            proprietor_name TEXT
        )
    ''')
    conn.commit()
    print(f"DEBUG: Initialized database {db_name} for {db_type} data.")
    return conn, cursor

# Function to dynamically find the starting page of trademark data in journal PDF
def find_trademark_start_page(doc):
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").upper()
        
        if "APPLICATIONS ADVERTISED BEFORE REGISTRATION" in text:
            print(f"DEBUG: Found transition header on page {page_num}. Starting extraction from page {page_num + 1}.")
            return page_num + 1
        
        if re.search(r"CLASS \d+", text):
            lines = text.split("\n")
            for i, line in enumerate(lines):
                if re.search(r"CLASS \d+", line) and i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if re.match(r"\d+\s+\d{2}/\d{2}/\d{4}", next_line):
                        print(f"DEBUG: Found Class pattern with structured entry on page {page_num}. Starting extraction from this page.")
                        return page_num
    
    print("DEBUG: No clear starting page found. Starting from page 0.")
    return 0

# Function to extract images (logos) from a page and save them
def extract_logo_from_page(page, application_number, page_num, is_journal):
    images = page.get_images(full=True)
    if not images:
        print(f"DEBUG: No images found for application {application_number} on page {page_num}.")
        return None
    
    doc = page.parent
    for img_index, img in enumerate(images):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        
        # Determine the appropriate subfolder based on is_journal
        logo_subdir = JOURNAL_LOGO_DIR if is_journal else CLIENT_LOGO_DIR
        logo_filename = f"{application_number}_page_{page_num}_img_{img_index}.png"
        logo_path = os.path.join(logo_subdir, logo_filename)
        
        with open(logo_path, "wb") as img_file:
            img_file.write(image_bytes)
        
        print(f"DEBUG: Extracted logo for application {application_number} on page {page_num}. Saved to {logo_path}.")
        return logo_path
    return None

# Function to extract placeholder text where a logo might be absent
def extract_logo_placeholder(text, lines, start_idx):
    placeholder = ""
    for i in range(start_idx, len(lines)):
        line = lines[i].strip()
        if "NO LOGO" in line.upper() or "DEVICE NOT PRESENT" in line.upper() or "TEXT MARK" in line.upper():
            placeholder = line
            break
        if re.match(r"[A-Z\s,.-]+", line) and not re.match(r"USED SINCE|PROPOSED TO BE USED|CLASS \d+", line):
            break
    print(f"DEBUG: Extracted logo placeholder: {placeholder if placeholder else 'None'}")
    return placeholder if placeholder else None

# Function to extract proprietor name from company name
def extract_proprietor_name(company_name):
    trading_as_match = re.match(r"(.+?)\s+TRADING AS\s+.+", company_name, re.I)
    if trading_as_match:
        proprietor = trading_as_match.group(1).strip()
        print(f"DEBUG: Extracted proprietor from 'TRADING AS': {proprietor}")
        return proprietor
    
    if " AND " in company_name.upper():
        proprietors = [name.strip() for name in company_name.split(" AND ")]
        proprietor_str = ", ".join(proprietors)
        print(f"DEBUG: Extracted multiple proprietors: {proprietor_str}")
        return proprietor_str
    
    print(f"DEBUG: Using company_name as proprietor: {company_name}")
    return company_name

# Function to extract data from a PDF (journal or client)
def extract_data_from_pdf(pdf_path, is_journal=True):
    print(f"DEBUG: Processing started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize the appropriate database
    db_type = "journal" if is_journal else "client"
    conn, cursor = init_db(db_type)
    
    doc = fitz.open(pdf_path)
    start_page = find_trademark_start_page(doc) if is_journal else 0
    print(f"DEBUG: Starting extraction from page {start_page} for {pdf_path}.")
    
    current_class = None
    journal_no = None
    journal_date = None
    
    for page_num in range(start_page, len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        print(f"DEBUG: Raw text on page {page_num}:\n{text}\n{'-'*50}")
        lines = text.split("\n")
        
        for line in lines[:5]:
            journal_match = re.match(r"Trade Marks Journal No: (\d+)\s*,\s*(\d{2}/\d{2}/\d{4})", line)
            if journal_match:
                journal_no, journal_date = journal_match.groups()
                print(f"DEBUG: Extracted journal_no: {journal_no}, journal_date: {journal_date}")
                class_match = re.search(r"Class (\d+)", line)
                if class_match:
                    current_class = class_match.group(1)
                    print(f"DEBUG: Extracted class from journal line: {current_class}")
                break
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            print(f"DEBUG: Processing line {i}: {line}")
            
            class_match = re.match(r"Class (\d+)", line)
            if class_match:
                current_class = class_match.group(1)
                print(f"DEBUG: Updated current_class to: {current_class}")
                i += 1
                continue
            
            app_match = re.match(r"(\d+)\s+(\d{2}/\d{2}/\d{4})", line)
            if not app_match:
                i += 1
                continue
            
            application_number, application_date = app_match.groups()
            print(f"DEBUG: Extracted application_number: {application_number}, application_date: {application_date}")
            
            i += 1
            company_name = ""
            address = ""
            entity_type = ""
            legal_status = ""
            
            accumulated_lines = []
            while i < len(lines):
                next_line = lines[i].strip()
                if re.match(r"USED SINCE|PROPOSED TO BE USED|Address for service", next_line, re.I):
                    break
                accumulated_lines.append(next_line)
                i += 1
            
            print(f"DEBUG: Accumulated lines for company details:\n" + "\n".join(accumulated_lines) + "\n" + "-"*50)
            
            j = 0
            while j < len(accumulated_lines):
                line = accumulated_lines[j].strip()
                
                if "trading as" in line.lower():
                    company_name = line
                    legal_status = "PROPRIETOR"
                    print(f"DEBUG: Found 'trading as' in company name: {company_name}")
                    print(f"DEBUG: Inferred legal_status: {legal_status}")
                    j += 1
                    continue
                
                if not company_name:
                    company_name = line
                    if "PVT. LTD." in company_name.upper() or "PRIVATE LIMITED" in company_name.upper():
                        legal_status = "a company organized under the laws of India"
                        print(f"DEBUG: Inferred legal_status from company name: {legal_status}")
                    elif re.match(r"[A-Z\s.]+$", company_name.upper()) and not address:
                        legal_status = "INDIVIDUAL"
                        print(f"DEBUG: Inferred legal_status as INDIVIDUAL: {legal_status}")
                    print(f"DEBUG: Set company_name: {company_name}")
                    j += 1
                    continue
                
                if re.search(r"[A-Z\s]+-\s*\d{3}\s*\d{3}", line) and not address:
                    address = line
                    print(f"DEBUG: Extracted address: {address}")
                    j += 1
                    continue
                
                if re.search(r"MANUFACTURERS?|MERCHANTS?|TRADERS?|DEALERS?", line.upper()):
                    entity_type = line
                    print(f"DEBUG: Extracted entity_type: {entity_type}")
                    j += 1
                    continue
                
                if re.search(r"(a company organized under the laws of India|INDIAN NATIONAL|INDIVIDUAL|PROPRIETOR)", line.lower()):
                    legal_status = line
                    print(f"DEBUG: Extracted legal_status: {legal_status}")
                    j += 1
                    continue
                
                j += 1
            
            # Extract proprietor name
            proprietor_name = extract_proprietor_name(company_name)
            
            address_for_service = ""
            while i < len(lines):
                next_line = lines[i].strip()
                if "Address for service" in next_line:
                    i += 1
                    while i < len(lines):
                        sub_line = lines[i].strip()
                        if re.match(r"USED SINCE|PROPOSED TO BE USED", sub_line, re.I):
                            break
                        address_for_service += " " + sub_line
                        i += 1
                    break
                i += 1
            address_for_service = address_for_service.strip()
            print(f"DEBUG: Extracted address_for_service: {address_for_service}")
            
            usage_details = ""
            while i < len(lines):
                next_line = lines[i].strip()
                usage_match = re.match(r"Used Since :(\d{2}/\d{2}/\d{4})|Proposed to be Used", next_line)
                if usage_match:
                    usage_details = next_line
                    i += 1
                    break
                i += 1
            print(f"DEBUG: Extracted usage_details: {usage_details}")
            
            location_of_use = ""
            while i < len(lines):
                next_line = lines[i].strip().upper()
                if re.match(r"[A-Z]+$", next_line) and next_line not in ["USED SINCE", "PROPOSED TO BE USED"]:
                    location_of_use = next_line
                    i += 1
                    break
                i += 1
            print(f"DEBUG: Extracted location_of_use: {location_of_use}")
            
            goods_services = ""
            while i < len(lines):
                next_line = lines[i].strip()
                if "THIS IS" in next_line.upper() or "REGISTRATION OF THIS TRADE MARK" in next_line.upper() or "No claim is made" in next_line:
                    break
                goods_services += " " + next_line
                i += 1
            goods_services = goods_services.strip()
            print(f"DEBUG: Extracted goods_services: {goods_services}")
            
            additional_notes = ""
            while i < len(lines):
                next_line = lines[i].strip()
                if re.match(r"\d+\s+\d{2}/\d{2}/\d{4}", next_line) or re.match(r"Class \d+", next_line):
                    break
                additional_notes += " " + next_line
                i += 1
            additional_notes = additional_notes.strip()
            print(f"DEBUG: Extracted additional_notes: {additional_notes}")
            
            logo_path = extract_logo_from_page(page, application_number, page_num, is_journal)
            logo_placeholder = None
            if not logo_path:
                logo_placeholder = extract_logo_placeholder(text, lines, i)
            
            cursor.execute('''
                INSERT INTO trademarks (
                    journal_no, journal_date, class, application_number, application_date,
                    company_name, address, entity_type, legal_status, address_for_service,
                    usage_details, location_of_use, goods_services, additional_notes,
                    logo_path, logo_placeholder, source_file, proprietor_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                journal_no, journal_date, current_class, application_number, application_date,
                company_name, address, entity_type, legal_status, address_for_service,
                usage_details, location_of_use, goods_services, additional_notes,
                logo_path, logo_placeholder, pdf_path, proprietor_name
            ))
            print(f"DEBUG: Inserted data for application {application_number} into database.")
            
            i += 1
    
    conn.commit()
    conn.close()
    doc.close()

# Main function to process PDFs
def process_trademarks(journal_pdf_path):
    print(f"Processing journal PDF: {journal_pdf_path}")
    extract_data_from_pdf(journal_pdf_path, is_journal=True)
    
    #print(f"Processing client PDF: {client_pdf_path}")
    #extract_data_from_pdf(client_pdf_path, is_journal=False)

if __name__ == "__main__":
    journal_pdf = "C:\\Users\\sukri\\Downloads\\document.pdf"
    process_trademarks(journal_pdf)