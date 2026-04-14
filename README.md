🎙️ \*\*DevWhisper — Voice-Native Developer Experience Agent\*\*



DevWhisper is a voice-first AI agent built for developers. Instead of stopping to search through files or documentation, you just ask out loud — and it answers based on your actual codebase.



\---



🚨 \*\*The Problem\*\*



Developers lose focus constantly. Switching between your editor, a browser, Stack Overflow, and documentation breaks the flow of thinking. Most AI tools still require you to type, copy-paste code, and wait.



DevWhisper lets you stay in flow. Ask a question with your voice, get an answer in seconds, and keep coding.



\---



✨ \*\*What It Does\*\*



🎤 You ask a question about your code

🔍 It searches your actual codebase semantically

🔊 It responds in plain spoken English, like a senior dev sitting next to you



Example questions that work:

\- "What does the preprocess function do?"

\- "Where is the model saved after training?"

\- "How do I debug a KeyError in the pipeline?"



\---



🏗️ \*\*Architecture\*\*



Developer speaks

↓

Vapi — Speech to Text

↓

FastAPI Webhook Server

↓

Qdrant Vector Search

↓

Groq LLaMA 3.3 70B

↓

FastAPI sends answer back

↓

Vapi — Text to Speech

↓

Developer hears the response



\---



🛠️ \*\*Tech Stack\*\*



🎙️ Vapi — handles voice input and output

🗄️ Qdrant — stores and searches code as vectors

🤖 Groq with LLaMA 3.3 70B — generates the response

⚡ FastAPI — receives webhooks from Vapi and orchestrates everything



\---



🚀 \*\*How To Run It\*\*



1\. Clone this repo

2\. Install dependencies: pip install -r requirements.txt

3\. Create a .env file in the root folder



QDRANT\_URL=your\_qdrant\_cluster\_url

QDRANT\_API\_KEY=your\_qdrant\_api\_key

GROQ\_API\_KEY=your\_groq\_api\_key



4\. Add your Python files to the sample\_codebase folder

5\. Index your codebase: python indexer.py

6\. Start the server: uvicorn main:app --reload --port 8000

7\. Expose publicly: ngrok http 8000

8\. Update Vapi tool Server URL with your ngrok URL plus /webhook



\---



📁 \*\*Project Structure\*\*



main.py — FastAPI webhook server, handles all Vapi events

indexer.py — chunks your code files and uploads them to Qdrant

retriever.py — takes a query and finds the most relevant code chunks

llm.py — sends the query and context to Groq and returns the answer

sample\_codebase/ — put your own Python project files here



\---



⚠️ \*\*Notes\*\*



The ngrok URL changes every time you restart it. Remember to update the Server URL in your Vapi tool settings each time.

The .env file is not included in this repo for security. You need to create your own with the keys above.



\---



👩‍💻 \*\*Built By Aharshi Sinha.



