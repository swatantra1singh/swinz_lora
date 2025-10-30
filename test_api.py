import requests
import json
import asyncio
import aiohttp

# Test script for the Swinz LoRA API

BASE_URL = "http://localhost:8000"  # Change to your deployed URL

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:", response.json())

def test_models():
    """Test models endpoint"""
    response = requests.get(f"{BASE_URL}/v1/models")
    print("Available Models:", response.json())

def test_simple_chat():
    """Test simple chat endpoint"""
    data = {
        "message": "What is Swinz insurance?",
        "temperature": 0.7,
        "max_tokens": 200
    }
    response = requests.post(f"{BASE_URL}/chat", json=data)
    print("Simple Chat Response:", response.json())

def test_streaming_chat():
    """Test streaming chat endpoint"""
    data = {
        "model": "swinz-3b-lora",
        "messages": [
            {"role": "user", "content": "Explain Swinz insurance benefits"}
        ],
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 200
    }
    
    response = requests.post(f"{BASE_URL}/v1/chat/completions", json=data, stream=True)
    
    print("Streaming Response:")
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data_str = line[6:]  # Remove 'data: ' prefix
                if data_str.strip() == '[DONE]':
                    break
                try:
                    data = json.loads(data_str)
                    if 'choices' in data and len(data['choices']) > 0:
                        delta = data['choices'][0].get('delta', {})
                        if 'content' in delta:
                            print(delta['content'], end='', flush=True)
                except json.JSONDecodeError:
                    continue
    print("\n")

def test_openai_format():
    """Test OpenAI-compatible format"""
    data = {
        "model": "swinz-3b-lora",
        "messages": [
            {"role": "user", "content": "What are the key features of Swinz insurance?"}
        ],
        "temperature": 0.8,
        "max_tokens": 150,
        "stream": False
    }
    
    response = requests.post(f"{BASE_URL}/v1/chat/completions", json=data)
    result = response.json()
    
    print("OpenAI Format Response:")
    print(f"Model: {result['model']}")
    print(f"Content: {result['choices'][0]['message']['content']}")
    if 'usage' in result:
        print(f"Usage: {result['usage']}")

if __name__ == "__main__":
    print("Testing Swinz LoRA API...")
    print("="*50)
    
    try:
        test_health()
        print("\n" + "="*50)
        
        test_models()
        print("\n" + "="*50)
        
        test_simple_chat()
        print("\n" + "="*50)
        
        test_openai_format()
        print("\n" + "="*50)
        
        test_streaming_chat()
        print("\n" + "="*50)
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running.")
    except Exception as e:
        print(f"Error: {e}")