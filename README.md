# LiveKit RAG Voice Assistant Demo w/ Deepseek

This demo showcases different approaches to handling the delay during RAG (Retrieval-Augmented Generation) lookups in a voice-enabled AI assistant w/ Deepseek AI. When users ask questions about LiveKit, the system needs time to search the knowledge base and generate responses. The demo provides three different methods to maintain user engagement during this process.

## Features

- Three different delay-handling options during RAG lookups:
  1. Text responses from a curated list (e.g., "Let me look that up...")
  2. Dynamic LLM-generated "thinking" responses
  3. Pre-recorded audio file playback
- Voice input/output through LiveKit's real-time communication
- RAG-powered responses using OpenAI embeddings
- Efficient vector search using Annoy index

## Prerequisites

- Python 3.10+
- OpenAI API key (For embeddings)
- Deepseek API key (LLM)
- LiveKit API key and secret
- Deepgram API Key (STT)

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your credentials:
   ```
   OPENAI_API_KEY=your_openai_api_key
   DEEPSEEK_API_KEY=your_deepseek_api_key
   LIVEKIT_API_KEY=your_livekit_api_key
   LIVEKIT_API_SECRET=your_livekit_api_secret
   DEEPGRAM_API_KEY=your_deepgram_api_key
   ```

## Usage

1. Build the knowledge base:
   ```bash
   python build_data.py
   ```
   This will process the content in `raw_data.txt` and create the necessary vector database files.

2. Start the agent:
   ```bash
   python agent.py
   ```

3. Connect to the agent:
   - Go to [agents-playground.livekit.io](https://agents-playground.livekit.io)
   - Connect to your LiveKit server
   - Enable your microphone
   - Start talking with the assistant!

## How It Works

1. The system first processes your text data into embeddings using OpenAI's text-embedding-3-small model
2. These embeddings are stored in an Annoy index for efficient similarity search
3. When you ask a question about LiveKit:
   - One of the three delay-handling methods is activated to keep you engaged
   - Meanwhile, in the background:
     - Your question is converted to an embedding
     - The most relevant information is retrieved from the knowledge base
     - OpenAI's GPT model generates a response
   - Once ready, the final response is converted to speech using OpenAI's TTS

## Delay Handling Options

The demo implements four different approaches to handle the RAG lookup delay:

1. **System Prompt**
   - Instructs the agent through the system prompt to always announce when it's looking up information
   - Most seamless integration with the agent's behavior
   - Consistent messaging that feels natural

2. **Static Text Messages**
   - Uses a predefined list of responses like "Let me look that up..." or "One moment while I check..."
   - Simple to implement and customize
   - Consistent user experience

3. **Dynamic LLM Responses**
   - Generates unique "thinking" messages using the LLM
   - More varied and contextual responses
   - Slightly higher latency and cost

4. **Audio File Playback**
   - Plays a pre-recorded audio message
   - Most rigid option
   - Can play sounds other than speech

You can experiment with these options by uncommenting the desired section in the `enrich_with_rag` function in `agent.py`. The system prompt option is enabled by default in the current implementation.
