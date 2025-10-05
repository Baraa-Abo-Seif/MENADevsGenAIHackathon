from pathlib import Path
from typing import Dict, Any, List
import logging
import yaml
from document_processor import DocumentProcessor

class TextExtractor:
    def __init__(self, config_path: str): #yaml ignore
        """Initialize the text extractor with configuration."""
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        self.document_processor = DocumentProcessor(config_path)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f) 

    async def extract_text_from_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """Extract text from multiple files and combine the results."""
        results = {
            'files': [],
            'combined_text': '',
            'metadata': {
                'total_files': len(file_paths),
                'successful_extractions': 0,
                'failed_extractions': 0
            }
        }

        for file_path in file_paths:
            try:
                content = self.document_processor.process_document(file_path) #read file dict
                
                text = self._extract_text_from_content(content) #Convert to text
                
                results['files'].append({
                    'filename': Path(file_path).name,
                    'text': text,
                    'status': 'success'
                })
                
                if results['combined_text']: #Merge to Text
                    results['combined_text'] += '\n\n--- New Document ---\n\n'
                results['combined_text'] += text
                
                results['metadata']['successful_extractions'] += 1
                
            except Exception as e: #Deal with mistakes
                self.logger.error(f"Error processing file {file_path}: {str(e)}")
                results['files'].append({
                    'filename': Path(file_path).name,
                    'error': str(e),
                    'status': 'error'
                })
                results['metadata']['failed_extractions'] += 1

        return results

    def _extract_text_from_content(self, content: Dict[str, Any]) -> str:
        """Extract text from processed content based on its structure."""
        if not content:
            return ""

        text_parts = []

        def extract_text_recursive(data: Any) -> None:
            """Recursively extract text from nested structures."""
            if isinstance(data, str):
                text_parts.append(data)
            elif isinstance(data, (int, float)):
                text_parts.append(str(data))
            elif isinstance(data, dict):
                for value in data.values():
                    extract_text_recursive(value)
            elif isinstance(data, list):
                for item in data:
                    extract_text_recursive(item)

        extract_text_recursive(content)
        
        return ' '.join(text_parts)

    def validate_extraction(self, extracted_text: str) -> bool:
        """Validate extracted text for basic quality checks."""
        if not extracted_text or len(extracted_text.strip()) < 10:
            return False
            
        return True