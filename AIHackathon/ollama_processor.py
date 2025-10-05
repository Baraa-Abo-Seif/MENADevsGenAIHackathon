import aiohttp
import logging
from typing import Dict, Any, Optional
import yaml
import json
from pathlib import Path
from datetime import datetime

class OllamaProcessor:
    def __init__(self, config_path: str):
        """Initialize Ollama processor with configuration."""
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            ollama_config = config.get('ollama', {})
            self.base_url = ollama_config.get('base_url', 'http://localhost:11434/api/generate')
            self.model = "llama3.2:1b"  # Use the specific model
            self.max_retries = ollama_config.get('max_retries', 3)
            self.timeout = ollama_config.get('timeout', 60)  # Increased timeout for streaming
            self.chunk_size = ollama_config.get('chunk_size', 1024)
        except Exception as e:
            self.logger.error(f"Error loading config: {str(e)}")
            # Set default values
            self.base_url = 'http://localhost:11434/api/generate'
            self.model = "llama3.2:1b"
            self.max_retries = 3
            self.timeout = 60
            self.chunk_size = 1024

    def _create_prompt(self, text: str) -> str:
        """Create a structured prompt for Ollama."""
        return f"""You are a precise document analyzer. Extract and summarize key information from the following text into clear key-value pairs.
                Follow these rules strictly:
                1. Normalize all dates to ISO 8601 format (YYYY-MM-DD)
                2. Convert all monetary amounts to JOD
                3. Use standardized keys (e.g., Date, Amount, Reference, Entity)
                4. Present each pair on a new line with a colon separator
                5. Keep values concise and specific
                6. Remove any uncertain or ambiguous information
                
                Example format:
                Date: 2025-10-05
                Amount: 150.00 JOD
                Reference: INV-2025-001
                Entity: Jordan Bank Ltd
                Location: Amman, Jordan
                
                Text to analyze:
                {text}
                
                Key Information (only validated facts):"""

    async def _make_request(self, prompt: str, attempt: int = 1) -> Optional[Dict[str, Any]]:
        """Make a request to the Ollama API with retry logic and streaming support."""
        try:
            async with aiohttp.ClientSession() as session: # session HTTP
                async with session.post(
                    self.base_url,
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": True
                    },
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status != 200: #HTTP 200 error
                        error_msg = f"Ollama API request failed with status {response.status}"
                        self.logger.error(error_msg)
                        if attempt < self.max_retries:
                            return await self._make_request(prompt, attempt + 1)
                        return None
                    
                    # Handle streaming response
                    full_response = []
                    async for line in response.content:
                        if not line:
                            continue
                            
                        try:
                            chunk = json.loads(line)
                            if 'response' in chunk:
                                full_response.append(chunk['response'])
                            if chunk.get('done', False):
                                break
                        except json.JSONDecodeError:
                            continue
                    
                    result = ''.join(full_response).strip()
                    return self._parse_ollama_response(result)
                    
        except aiohttp.ClientError as e:
            self.logger.error(f"Ollama API request error: {str(e)}")
            if attempt < self.max_retries:
                return await self._make_request(prompt, attempt + 1)
            return None

    def _parse_ollama_response(self, response: str) -> Dict[str, Any]:
        """Parse and structure the Ollama response."""
        lines = [line.strip() for line in response.split('\n') if line.strip()] # Clean lines
        structured_data = {} # Store key-value
        
        for line in lines:
            # Skip lines without key-value 
            if ':' not in line:
                continue
                
            try:
                key, value = [part.strip() for part in line.split(':', 1)]
                
                # Skip empty
                if not value or value.lower() in ['none', 'n/a', 'unknown', 'not found']:
                    continue
                    
                key = key.strip().title()
                
                if 'date' in key.lower():
                    # Ensure date
                    if value and len(value) >= 8:
                        try:
                            from datetime import datetime
                            date_obj = datetime.strptime(value.split()[0], '%Y-%m-%d')
                            value = date_obj.strftime('%Y-%m-%d')
                        except ValueError:
                            continue
                
                if 'amount' in key.lower() or 'price' in key.lower() or 'cost' in key.lower():
                    if 'jod' not in value.lower():
                        value = f"{value.split()[0]} JOD"
                
                structured_data[key] = value
                
            except Exception as e:
                self.logger.error(f"Error parsing line '{line}': {str(e)}")
                continue
        
        return structured_data

    async def summarize(self, text: str) -> Dict[str, Any]:
        """Summarize text using Ollama.""" #check empty
        if not text or not text.strip():
            return {
                "status": "error",
                "message": "No text provided for summarization"
            }
        
        try:
            max_length = 4000  # Ollama input limit
            if len(text) > max_length:
                text = text[:max_length] + "..."
            
            prompt = self._create_prompt(text)
            result = await self._make_request(prompt)
            
            if result is None:
                return {
                    "status": "error",
                    "message": f"Failed to get response from Ollama after {self.max_retries} attempts. Please ensure the service is running."
                }
            
            if not result:
                return {
                    "status": "error",
                    "message": "No structured information could be extracted from the text"
                }
            
            return {
                "status": "ok",
                "summary": result,
                "metadata": {
                    "model": self.model,
                    "timestamp": datetime.now().isoformat(),
                    "text_length": len(text),
                    "extracted_fields": len(result)
                }
            }
            
        except aiohttp.ClientError as e:
            self.logger.error(f"Ollama connection error: {str(e)}")
            return {
                "status": "error",
                "message": "Could not connect to Ollama service. Please ensure it is running."
            }
        except Exception as e:
            self.logger.error(f"Error summarizing text: {str(e)}")
            return {
                "status": "error",
                "message": f"Summarization failed: {str(e)}"
            }