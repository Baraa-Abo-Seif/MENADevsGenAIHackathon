from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import tempfile
import logging
import mimetypes
import json
import pandas as pd
from datetime import datetime
from typing import List, Optional, Dict, Any
from io import BytesIO
import aiohttp
import uvicorn
from dateutil import parser
import re
from document_processor import DocumentProcessor
from text_extractor import TextExtractor
# processor logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='processor.log'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Document Processor API")

# configure static files and templates
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Configure uploads
UPLOAD_FOLDER = Path('uploads')
UPLOAD_FOLDER.mkdir(exist_ok=True)
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB limit

# Configure allowed file types
ALLOWED_EXTENSIONS = {'.pdf', '.doc', '.docx', '.xlsx', '.xls', '.json'}
ALLOWED_MIMETYPES = {
    'application/pdf',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/vnd.ms-excel',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/json'
}

# Configure Ollama
OLLAMA_CONFIG = {
    'api_url': 'http://localhost:11434/api/chat',
    'model': 'llama3.2:1b',
    'timeout': 130,  # seconds
    'max_length': 4000  # characters
}

OLLAMA_MODEL = OLLAMA_CONFIG['model']
OLLAMA_API_URL = OLLAMA_CONFIG['api_url']

DATE_PATTERNS = [
    r'\d{2}/\d{2}/\d{4}',
    r'\d{2}-\d{2}-\d{4}',
    r'\d{4}/\d{2}/\d{2}',
    r'\d{4}-\d{2}-\d{2}'
]

async def validate_file(file: UploadFile) -> tuple[bool, Optional[str]]:
    """Validate uploaded file for size, type, and content."""
    try:
        # Check if file is empty
        if not file.filename:
            return False, "No file selected"
            
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return False, f"File type not allowed. Supported types: {', '.join(ALLOWED_EXTENSIONS)}"
            
        mime_type, _ = mimetypes.guess_type(file.filename)
        if not mime_type or mime_type not in ALLOWED_MIMETYPES:
            return False, f"Invalid file type: {mime_type}"
                
        return True, None
        
    except Exception as e:
        logger.error(f"File validation error: {str(e)}")
        return False, "Error validating file"

def normalize_date(date_str: str) -> str:
    """Convert date string to ISO 8601 format."""
    try:
        for pattern in DATE_PATTERNS:
            match = re.search(pattern, date_str)
            if match:
                date_str = match.group(0)
                parsed_date = parser.parse(date_str)
                return parsed_date.strftime('%Y-%m-%d')
        return date_str
    except Exception as e:
        logger.error(f"Date normalization failed: {str(e)}")
        return date_str

async def process_file(file: UploadFile) -> Dict[str, Any]:
    """Process uploaded file and return extracted content."""
    try:
        content = await file.read()
        ext = Path(file.filename).suffix.lower()
        
        # Handle Excel files
        if ext in ['.xlsx', '.xls']:
            try:
                df = pd.read_excel(BytesIO(content))
                # Convert DataFrame to structured format
                records = df.to_dict('records')
                text = df.to_string(index=False)
                return {
                    'text': text,
                    'structured_data': {
                        'type': 'spreadsheet',
                        'metadata': {
                            'rows': len(df),
                            'columns': len(df.columns),
                            'column_names': df.columns.tolist()
                        },
                        'data': records
                    }
                }
            except Exception as e:
                logger.error(f"Excel processing error: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to process Excel file: {str(e)}"
                )
        
        # Handle JSON files
        if ext == '.json':
            try:
                data = json.loads(content.decode('utf-8'))
                return {
                    'text': json.dumps(data, indent=2),
                    'structured_data': {
                        'type': 'json',
                        'data': data
                    }
                }
            except Exception as e:
                logger.error(f"JSON processing error: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to process JSON file: {str(e)}"
                )
        
        # Handle PDF, DOC, and DOCX files using DocumentProcessor
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_file.write(content)
            temp_file.flush()
            
            try:
                processor = DocumentProcessor('config_template.yaml')
                result = processor.process_document(temp_file.name)
                return result
            finally:
                Path(temp_file.name).unlink()  # Clean up temp file
    except Exception as e:
        logger.error(f"File processing error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process file: {str(e)}"
        )

async def summarize_with_ollama(text: str) -> Dict[str, Any]:
    """Send text to Ollama API for summarization."""
    async with aiohttp.ClientSession() as session:
        try:
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": f"""Extract and organize key information from the following text.
                            Focus on dates, amounts, names, and important details.
                            Format the response as clear, structured key-value pairs.
                            
                            Text:
                            {text}
                            
                            Key Information:"""
            }
            
            async with session.post(OLLAMA_API_URL, json=payload) as response:
                if response.status != 200:
                    raise HTTPException(status_code=500, detail="Ollama API request failed")
                    
                result = await response.text()
                return {"summary": result}
                
        except aiohttp.ClientError as e:
            logger.error(f"Ollama API error: {str(e)}")
            raise HTTPException(status_code=500, detail="Error connecting to Ollama API")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main page."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": "Document Processor"
        }
    )

@app.post("/upload")
async def upload_files(request: Request, file: UploadFile = File(...)):
    """Handle file upload and return processed content."""
    try:
        # Check if file uploaded
        if not file:
            return JSONResponse(
                status_code=400,
                content={"error": "No file uploaded"}
            )
        
        # Check Content-Type
        content_type = file.content_type
        if content_type not in ALLOWED_MIMETYPES:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid file type: {content_type}. Allowed types: {', '.join(ALLOWED_MIMETYPES)}"}
            )
            
        # Validate file
        is_valid, error_msg = await validate_file(file)
        if not is_valid:
            return JSONResponse(
                status_code=400,
                content={"error": error_msg}
            )
        
        # Create a temporary file
        ext = Path(file.filename).suffix.lower()
        try:
            # Read file content
            content = await file.read()
            if not content:
                return JSONResponse(
                    status_code=400,
                    content={"error": "File is empty"}
                )
                
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
                temp_file.write(content)
                temp_file.flush()
                
                try:
                    # Process file
                    processor = DocumentProcessor('config_template.yaml')
                    result = processor.process_document(temp_file.name)
                    
                    # Extract textt
                    if isinstance(result, dict):
                        # extract text  dictionaries
                        def extract_text(data):
                            if isinstance(data, str):
                                return data + "\n"
                            elif isinstance(data, dict):
                                return "".join(extract_text(v) for v in data.values())
                            elif isinstance(data, list):
                                return "".join(extract_text(item) for item in data)
                            return str(data) + "\n"
                        
                        text = extract_text(result)
                    else:
                        text = str(result)
                    
                    # Normalize dates structured
                    if isinstance(result, dict):
                        for key, value in result.items():
                            if 'date' in key.lower() and isinstance(value, str):
                                result[key] = normalize_date(value)
                    
                    # Store result in app state
                    request.app.state.last_result = {
                        "filename": file.filename,
                        "text": text.strip(),
                        "structured_data": result,
                        "summary": {}
                    }
                    
                    return JSONResponse(
                        status_code=200,
                        content=request.app.state.last_result
                    )
                    
                except Exception as e:
                    logger.error(f"Error processing file {file.filename}: {str(e)}")
                    return JSONResponse(
                        status_code=500,
                        content={"error": f"Error processing file: {str(e)}"}
                    )
                finally:
                    # Clean up temp file
                    try:
                        Path(temp_file.name).unlink()
                    except Exception as e:
                        logger.error(f"Error deleting temp file: {str(e)}")
                        
        except Exception as e:
            logger.error(f"Error handling file {file.filename}: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Error handling file: {str(e)}"}
            )
                
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Upload failed: {str(e)}"}
        )

@app.post("/extract")
async def extract_text(file_ids: List[str]):
    """Extract text from processed files."""
    try:
        temp_files = []
        text_extractor = TextExtractor('config.yaml')
        
        #paths all processed files
        file_paths = [f for f in file_ids if Path(f).exists()]
        
        if not file_paths:
            raise HTTPException(
                status_code=400,
                detail="No valid files found for text extraction"
            )
            
        # Extract text from all files
        extraction_results = await text_extractor.extract_text_from_files(file_paths)
        
        if extraction_results['metadata']['successful_extractions'] == 0:
            raise HTTPException(
                status_code=500,
                detail="Failed to extract text from any of the files"
            )
            
        return JSONResponse(content={"results": extraction_results})
    except Exception as e:
        logger.error(f"Text extraction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
async def summarize_text(request: Request):
    """Summarize extracted text using Ollama."""
    try:
        data = await request.json()
        text = data.get('text')
        
        if not text:
            return JSONResponse(
                status_code=400,
                content={"error": "No text provided"}
            )
        
        # Log the request for debugging
        logger.info(f"Sending request to Ollama with {len(text)} characters of text")
        
        # cut too long txt
        if len(text) > OLLAMA_CONFIG['max_length']:
            text = text[:OLLAMA_CONFIG['max_length']] + '...'
            
        def parse_key_value(content: str) -> dict:
            """Parse Ollama response into key-value pairs."""
            result = {}
            lines = content.strip().split('\n')
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower().replace(' ', '_')
                    value = value.strip()
                    if key and value:
                        result[key] = value
            return result
            
        async with aiohttp.ClientSession() as session:
            try:
                # Prepare payload with clearer instructions
                payload = {
                    "model": OLLAMA_CONFIG['model'],
                    "messages": [{
                        "role": "system",
                        "content": "You are a data extraction assistant. Respond only with key-value pairs."
                    }, {
                        "role": "user",
                        "content": f"""Extract key information from this text.

Rules:
1. Respond ONLY with key-value pairs
2. Format: 'Key: Value' (one per line)
3. Use YYYY-MM-DD for dates
4. Use 'XXX.XX JOD' for currency
5. No explanations or additional text
6. At least 3-5 key pieces of information

Text to analyze:
{text}

Extracted Information:"""
                    }],
                    "stream": True
                }
                async with session.post(OLLAMA_CONFIG['api_url'], json=payload) as response:
                    if response.status != 200:
                        return JSONResponse(
                            status_code=500,
                            content={"error": "Ollama service unavailable"}
                        )
                    
                    complete_response = ""
                    parsed_data = {}
                    line_buffer = ""
                    
                    async for line in response.content:
                        if line:
                            try:
                                json_data = json.loads(line.decode('utf-8'))
                                if "message" in json_data and "content" in json_data["message"]:
                                    content = json_data["message"]["content"]
                                    complete_response += content
                                    line_buffer += content
                                    
                                    while '\n' in line_buffer:
                                        current_line, remaining = line_buffer.split('\n', 1)
                                        if ':' in current_line:
                                            key, value = current_line.split(':', 1)
                                            key = key.strip().lower().replace(' ', '_')
                                            value = value.strip()
                                            
                                            if any(date_word in key for date_word in ['date', 'issued', 'due']):
                                                try:
                                                    parsed_date = parser.parse(value)
                                                    value = parsed_date.strftime('%Y-%m-%d')
                                                except:
                                                    pass  
                                                    
                                            if any(amount_word in key for amount_word in ['amount', 'total', 'price', 'cost']):
                                                try:
                                                    numeric_value = float(''.join(c for c in value if c.isdigit() or c == '.'))
                                                    value = f"{numeric_value:.2f} JOD"
                                                except:
                                                    pass  
                                                    
                                            if key and value:
                                                parsed_data[key] = value
                                        line_buffer = remaining
                            except json.JSONDecodeError:
                                continue
                            except Exception as e:
                                logger.error(f"Error processing line: {str(e)}")
                                continue
                    
                    # Process any remaining buffer
                    if line_buffer and ':' in line_buffer:
                        key, value = line_buffer.split(':', 1)
                        key = key.strip().lower().replace(' ', '_')
                        value = value.strip()
                        if key and value:
                            parsed_data[key] = value
                    
                    if not parsed_data:
                        logger.error(f"Failed to extract any key-value pairs from response: {complete_response}")
                        return JSONResponse(
                            status_code=400,
                            content={
                                "status": "error",
                                "message": "Invalid summary format received"
                            }
                        )
                        
                    logger.info(f"Successfully parsed {len(parsed_data)} key-value pairs from Ollama response")
                    
                    request.app.state.last_summary = parsed_data
                    
                    if hasattr(request.app.state, 'last_result'):
                        request.app.state.last_result['summary'] = parsed_data
                    
                    table_data = [
                        {"variable": key, "value": value}
                        for key, value in parsed_data.items()
                    ]
                    
                    return JSONResponse(
                        status_code=200,
                        content={
                            "status": "ok",
                            "table_data": table_data,
                            "summary": parsed_data
                        }
                    )
            except Exception as e:
                logger.error(f"Ollama processing error: {str(e)}")
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Error processing with Ollama: {str(e)}"}
                )
                
    except aiohttp.ClientError:
        return JSONResponse(
            status_code=500,
            content={"error": "Could not connect to Ollama service"}
        )
    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Summarization failed: {str(e)}"}
        )
                    #END Ollama_processor.py
@app.get("/download/{format}")
async def download_results(format: str, request: Request):
    """Download processed results in specified format."""
    try:
        if not hasattr(request.app.state, 'last_result'):
            return JSONResponse(
                status_code=404,
                content={"error": "No processed data available"}
            )
        
        data = request.app.state.last_result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        ollama_summary = {}
        if hasattr(data, 'summary') and isinstance(data.get('summary'), dict):
            ollama_summary = data['summary']
        elif hasattr(request.app.state, 'last_summary') and isinstance(request.app.state.last_summary, dict):
            ollama_summary = request.app.state.last_summary
        
        if format.lower() == 'json':
            export_data = {
                "metadata": {
                    "generated_at": timestamp,
                    "source_file": data.get("filename", "unknown")
                },
                "structured_data": data.get("structured_data", {}),
                "summary": ollama_summary
            }
            
            content = json.dumps(export_data, indent=2, ensure_ascii=False)
            media_type = 'application/json'
            filename = f'processed_{timestamp}.json'
            
        elif format.lower() == 'csv':
            rows = []
            
            rows.extend([
                {"variable": "generated_at", "value": timestamp},
                {"variable": "source_file", "value": data.get("filename", "unknown")},
                {"variable": "", "value": ""}  # Empty row as separator
            ])
            
            rows.append({"variable": "=== STRUCTURED DATA ===", "value": ""}) # Section header
            if isinstance(data.get('structured_data'), dict):
                for key, value in data["structured_data"].items():
                    rows.append({
                        "variable": key,
                        "value": str(value).replace('\n', ' ')
                    })
            
            # Add Ollama summary section
            rows.extend([
                {"variable": "", "value": ""},  # Empty row as separator
                {"variable": "=== OLLAMA SUMMARY ===", "value": ""}  # Section header
            ])
            if ollama_summary:
                for key, value in ollama_summary.items():
                    rows.append({
                        "variable": key,
                        "value": str(value).replace('\n', ' ')
                    })
            else:
                rows.append({"variable": "note", "value": "No Ollama summary available"})
            
            # Convert to DataFrame and get CSV
            df = pd.DataFrame(rows)
            content = df.to_csv(index=False)
            media_type = 'text/csv'
            filename = f'processed_{timestamp}.csv'
            
        else:
            return JSONResponse(
                status_code=400,
                content={"error": f"Unsupported format: {format}. Use 'json' or 'csv'."}
            )
        
        # Create a BytesIO buffer with the content
        buffer = BytesIO(content.encode('utf-8'))
        
        return StreamingResponse(
            buffer,
            media_type=media_type,
            headers={
                'Content-Disposition': f'attachment; filename="{filename}"',
                'Cache-Control': 'no-cache'
            }
        )
        
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to generate download: {str(e)}"}
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)