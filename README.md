# Swinz LoRA API Deployment

A FastAPI-based deployment solution for your fine-tuned Swinz LoRA model with streaming support, compatible with OpenAI API format.

## 🚀 Features

- **OpenAI-Compatible API**: Drop-in replacement for OpenAI API calls
- **Streaming Support**: Real-time token streaming for chat applications
- **LoRA Integration**: Efficient loading of your fine-tuned LoRA weights
- **Auto-scaling**: Ready for Render.com deployment
- **Health Monitoring**: Built-in health checks and logging
- **Flutter Ready**: Perfect for mobile app integration

## 📁 Project Structure

```
swinz_lora/
├── app.py              # Main FastAPI application
├── requirements.txt    # Python dependencies
├── Dockerfile         # Docker configuration
├── render.yaml        # Render.com deployment config
├── setup.sh           # Setup script
├── test_api.py        # API testing script
├── swinz-3b-lora/     # Your trained LoRA model files
│   ├── adapter_model.bin
│   ├── adapter_config.json
│   └── tokenizer files...
└── README.md          # This file
```

## 🛠️ Local Setup

### Prerequisites
- Python 3.11+
- Your trained LoRA model files
- At least 8GB RAM (16GB recommended)

### Installation

1. **Clone and navigate to your project:**
   ```bash
   cd /Users/swatantra/projects/swinz_lora
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your LoRA model files:**
   Place your trained model files in the `swinz-3b-lora/` directory:
   - `adapter_model.bin`
   - `adapter_config.json`
   - Any custom tokenizer files

4. **Run the server:**
   ```bash
   python app.py
   ```

5. **Test the API:**
   ```bash
   python test_api.py
   ```

## ☁️ Render.com Deployment

### Step 1: Prepare Your Repository

1. **Create a GitHub repository** and push your code:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/swinz_lora.git
   git push -u origin main
   ```

2. **Upload your model files** (since they're likely large):
   - Use Git LFS for large files, or
   - Upload them directly via Render dashboard after deployment

### Step 2: Deploy on Render.com

1. **Login to Render.com** and create a new Web Service

2. **Connect your GitHub repository**

3. **Configure the service:**
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python app.py`
   - **Environment:** `Python 3`
   - **Plan:** Choose based on your needs (Standard+ recommended for ML models)

4. **Add Environment Variables:**
   ```
   PORT=8000
   PYTHON_VERSION=3.11.0
   ```

5. **Deploy and wait** for the build to complete

### Step 3: Upload Model Files

If using Render's file upload:
1. Go to your service dashboard
2. Navigate to the "Shell" tab
3. Upload your LoRA model files to `/app/swinz-3b-lora/`

## 📡 API Usage

### Base URL
- **Local:** `http://localhost:8000`
- **Render:** `https://your-app-name.onrender.com`

### Endpoints

#### Health Check
```bash
GET /health
```

#### List Models
```bash
GET /v1/models
```

#### Chat Completion (OpenAI Compatible)
```bash
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "swinz-3b-lora",
  "messages": [
    {"role": "user", "content": "What is Swinz insurance?"}
  ],
  "temperature": 0.7,
  "max_tokens": 200,
  "stream": false
}
```

#### Streaming Chat
```bash
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "swinz-3b-lora",
  "messages": [
    {"role": "user", "content": "Explain Swinz benefits"}
  ],
  "stream": true,
  "temperature": 0.7,
  "max_tokens": 200
}
```

#### Simple Chat (Custom Endpoint)
```bash
POST /chat
Content-Type: application/json

{
  "message": "What are Swinz insurance plans?",
  "temperature": 0.7,
  "max_tokens": 200,
  "stream": false
}
```

## 📱 Flutter Integration

### HTTP Client Example

```dart
import 'dart:convert';
import 'dart:async';
import 'package:http/http.dart' as http;

class SwinzApiClient {
  final String baseUrl;
  
  SwinzApiClient(this.baseUrl);
  
  // Non-streaming chat
  Future<String> chat(String message) async {
    final response = await http.post(
      Uri.parse('$baseUrl/chat'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'message': message,
        'temperature': 0.7,
        'max_tokens': 200,
      }),
    );
    
    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return data['choices'][0]['message']['content'];
    }
    throw Exception('Failed to get response');
  }
  
  // Streaming chat
  Stream<String> chatStream(String message) async* {
    final request = http.Request(
      'POST',
      Uri.parse('$baseUrl/v1/chat/completions'),
    );
    
    request.headers['Content-Type'] = 'application/json';
    request.body = jsonEncode({
      'model': 'swinz-3b-lora',
      'messages': [
        {'role': 'user', 'content': message}
      ],
      'stream': true,
      'temperature': 0.7,
      'max_tokens': 200,
    });
    
    final streamedResponse = await request.send();
    
    await for (final chunk in streamedResponse.stream.transform(utf8.decoder)) {
      final lines = chunk.split('\\n');
      for (final line in lines) {
        if (line.startsWith('data: ')) {
          final data = line.substring(6);
          if (data.trim() == '[DONE]') return;
          
          try {
            final json = jsonDecode(data);
            final content = json['choices'][0]['delta']['content'];
            if (content != null) yield content;
          } catch (e) {
            // Skip invalid JSON
          }
        }
      }
    }
  }
}
```

### Usage in Flutter Widget

```dart
class ChatScreen extends StatefulWidget {
  @override
  _ChatScreenState createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final SwinzApiClient apiClient = SwinzApiClient('https://your-app.onrender.com');
  String response = '';
  
  void sendMessage(String message) {
    setState(() => response = '');
    
    apiClient.chatStream(message).listen(
      (token) => setState(() => response += token),
      onError: (error) => print('Error: $error'),
      onDone: () => print('Stream completed'),
    );
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        children: [
          Expanded(
            child: SingleChildScrollView(
              child: Text(response),
            ),
          ),
          // Add your message input widget here
        ],
      ),
    );
  }
}
```

## 🔧 Configuration

### Model Configuration
Edit these variables in `app.py`:
```python
BASE_MODEL_ID = "openlm-research/open_llama_3b_v2"
LORA_MODEL_PATH = "./swinz-3b-lora"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
```

### Performance Tuning
- **CPU Only:** Set `device = torch.device("cpu")` for CPU-only deployment
- **Memory:** Adjust batch sizes and context length based on available RAM
- **Caching:** Enable model caching for faster subsequent loads

## 🐛 Troubleshooting

### Common Issues

1. **Model not loading:**
   - Ensure LoRA files are in the correct directory
   - Check file permissions
   - Verify model compatibility

2. **Out of memory:**
   - Reduce `MAX_NEW_TOKENS`
   - Use CPU instead of GPU for small deployments
   - Upgrade your Render plan

3. **Slow response:**
   - This is expected for CPU inference
   - Consider upgrading to GPU-enabled plans
   - Optimize model parameters

4. **Connection timeout:**
   - Increase timeout settings in your client
   - Check Render service logs
   - Ensure health check is passing

### Logs and Monitoring

Check logs on Render:
1. Go to your service dashboard
2. Click on "Logs" tab
3. Monitor startup and request logs

## 💰 Cost Considerations

### Render.com Plans
- **Starter Plan:** Good for testing (may timeout on large models)
- **Standard Plan:** Recommended for production
- **Pro Plan:** For high-traffic applications

### Optimization Tips
- Use model quantization for smaller memory footprint
- Implement request caching
- Set up auto-scaling based on traffic

## 🔄 Updates and Maintenance

### Updating the Model
1. Train new LoRA weights
2. Replace files in `swinz-3b-lora/` directory
3. Redeploy the service

### Monitoring
- Set up health check alerts
- Monitor response times
- Track API usage

## 📞 Support

For issues with:
- **Deployment:** Check Render.com documentation
- **Model Loading:** Verify PEFT and transformers versions
- **API Integration:** Test with the provided `test_api.py` script

## 🎯 Next Steps

1. **Deploy to Render.com** following the steps above
2. **Test the API** with the provided test script
3. **Integrate with Flutter** using the example code
4. **Monitor and optimize** based on usage patterns
5. **Scale as needed** based on traffic

Your Swinz LoRA model is now ready for production deployment! 🚀