import os
from typing import List, Dict, Any
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class PdfParser:
    """
    A modular class to handle PDF loading, text extraction, and chunking
    for Agentic RAG systems.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initializes the parser with LangChain's RecursiveCharacterTextSplitter.
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def extract_text_by_page(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extracts raw text from the PDF, keeping track of page numbers.
        
        :param file_path: Path to the PDF file.
        :return: A list of dictionaries containing page number and text content.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file at {file_path} does not exist.")

        extracted_pages = []
        
        # Open the PDF document
        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text")  # "text" layout preserves natural reading order
                extracted_pages.append({
                    "page_number": page_num,
                    "text": text.strip()
                })
                
        return extracted_pages

    def split_text_into_chunks(self, pages_data: List[Dict[str, Any]]) -> List[Document]:
        """
        Splits extracted page text into smaller, overlapping chunks suitable for LLM context windows.
        
        :param pages_data: Output from extract_text_by_page.
        :return: A list of LangChain Document objects containing chunk text and metadata.
        """
        langchain_docs = []

        for page in pages_data:
            if not page["text"]:
                continue  # Skip empty pages

            doc = Document(
                page_content = page["text"],
                metadata = {"page_number": page["page_number"]}
            )
            langchain_docs.append(doc)

        #using the splitter to create chunks from the page text
        final_chunks = self.splitter.split_documents(langchain_docs)
        return final_chunks


# ==========================================
# Example Usage / Testing Local Component
# ==========================================
if __name__ == "__main__":
    # Initialize the parser
    parser = PdfParser(chunk_size=500, chunk_overlap=100)
    
    # Replace this with a path to a real PDF for testing
    sample_pdf = "/home/suyogdahal/Desktop/project/Rag_System_Using_HF_LC/temp_uploads/saap_company_budget_rag_test.pdf" 
    
    try:
        print(f"--- Phase 1: Extracting text from {sample_pdf} ---")
        raw_pages = parser.extract_text_by_page(sample_pdf)
        print(f"Successfully processed {len(raw_pages)} pages.\n")
        
        print("--- Phase 2: Splitting text into chunks for Vector DB ---")
        processed_chunks = parser.split_text_into_chunks(raw_pages)
        print(f"Generated {len(processed_chunks)} total chunks.\n")
        
        # Print a sample chunk to inspect structure
        if processed_chunks:
            print("--- Sample Chunk Output ---")
            print(f"Metadata: {processed_chunks[0].metadata}")
            print(f"Content:\n{processed_chunks[0].page_content[:200]}...")
            
    except FileNotFoundError as e:
        print(f"Error: {e}. Please provide a valid PDF file path to run the test script.")