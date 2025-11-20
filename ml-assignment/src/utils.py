# This file is optional.
# You can add any utility functions you need for your implementation here.
import pdfplumber
import re

def scrape_pdf(pdf_path, output_txt):
    """
    Scrapes text from the specific Alice in Wonderland PDF,
    filtering out UI navigation noise.
    """
    
    # List of exact phrases appearing in the UI to ignore
    # Based on the file analysis, these appear on nearly every page
    NOISE_PHRASES = [
        "Frankenstein",
        "Planet PDF",
        "DEBENU",
        "PDF FREEDOM",
        "This eBook was designed and published by Planet PDF"
    ]

    clean_text = []

    print(f"Opening {pdf_path}...")
    
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"Found {total_pages} pages. Processing...")

        for i, page in enumerate(pdf.pages):
            # Extract text from the page
            text = page.extract_text()
            
            if not text:
                continue

            lines = text.split('\n')
            cleaned_lines = []
                
            for line in lines:
                # 1. Check for exact noise phrases
                is_noise = False
                for noise in NOISE_PHRASES:
                    if noise.lower() in line.lower():
                        is_noise = True
                        break
                
                # 3. Filter out standalone page numbers (often just digits)
                # 1. Filter out page numbers like "2 of 345"
                # This regex matches:
                # ^     : Start of the line
                # \d+   : One or more digits
                # \s+   : One or more spaces
                # of    : The literal word "of"
                # \s+   : One or more spaces
                # \d+   : One or more digits
                # $     : End of the line
                if re.match(r'^\d+\s+of\s+\d+$', line.strip()):
                    continue

                if not is_noise:
                    cleaned_lines.append(line)

            # Add page marker (optional, helpful for debugging)
            page_content = "\n".join(cleaned_lines)
            
            # Only add pages that have actual content left
            if page_content.strip():
                #clean_text.append(f"--- Page {i+1} ---")
                clean_text.append(page_content)
                #clean_text.append("\n")

    # Write to file
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(clean_text))

    print(f"Successfully scraped content to {output_txt}")

if __name__ == "__main__":
    # Ensure the PDF file is in the same directory or provide full path
    pdf_filename = "Frankenstein_T.pdf"
    output_filename = "Frankenstein_T_cleaned_text.txt"
    
    try:
        scrape_pdf(pdf_filename, output_filename)
    except FileNotFoundError:
        print(f"Error: Could not find '{pdf_filename}'. Make sure the file is in the correct folder.")