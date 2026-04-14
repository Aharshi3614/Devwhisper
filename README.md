🎙️ DevWhisper



DevWhisper is a voice-first AI agent built for developers. Instead of stopping to search through files or documentation, you just ask out loud — and it answers based on your actual codebase.



🚨 The problem



Developers lose focus constantly. Switching between your editor, a browser, Stack Overflow, and documentation breaks the flow of thinking. Most AI tools still require you to type, copy-paste code, and wait.



DevWhisper lets you stay in flow. Ask a question with your voice, get an answer in seconds, and keep coding.





✨ What it does



🎤 You ask a question about your code

🔍 It searches your actual codebase semantically

🔊 It responds in plain spoken English, like a senior dev sitting next to you



Example questions that work:

\- "What does the preprocess function do?"

\- "Where is the model saved after training?"

\- "How do I debug a KeyError in the pipeline?"



🏗️ Architecture



```mermaid

flowchart TD

&#x20;   A\[🎤 Developer speaks a question] --> B\[Vapi\\nSpeech to Text]

&#x20;   B --> C\[FastAPI Webhook Server\\nmain.py]

&#x20;   C --> D\[Retriever\\nretriever.py]

&#x20;   D --> E\[Qdrant Vector DB\\nSemantic Code Search]

&#x20;   E --> D

&#x20;   D --> C

&#x20;   C --> F\[LLM Layer\\nllm.py]

&#x20;   F --> G\[Groq API\\nLLaMA 3.3 70B]

&#x20;   G --> F

&#x20;   F --> C

&#x20;   C --> B

&#x20;   B --> H\[🔊 Vapi speaks the answer back]

```



\## Tech stack



🛠️ Vapi — handles voice input and output

🗄️ Qdrant — stores and searches code as vectors

🤖 Groq with LLaMA 3.3 70B — generates the response

⚡ FastAPI — receives webhooks from Vapi and orchestrates everything





🚀 How to run it



1\. Clone this repo

2\. Install dependencies



pip install -r requirements.txt



3\. Create a .env file in the root folder



QDRANT\_URL=your\_qdrant\_cluster\_url

QDRANT\_API\_KEY=your\_qdrant\_api\_key

GROQ\_API\_KEY=your\_groq\_api\_key



4\. Add your Python files to the sample\_codebase folder



5\. Index your codebase into Qdrant



python indexer.py



6\. Start the backend server



uvicorn main:app --reload --port 8000



7\. Expose it publicly using ngrok



ngrok http 8000



8\. In your Vapi dashboard, set the tool Server URL to your ngrok URL plus /webhook





📁 Project structure



main.py — FastAPI webhook server, handles all Vapi events

indexer.py — chunks your code files and uploads them to Qdrant

retriever.py — takes a query and finds the most relevant code chunks

llm.py — sends the query and context to Groq and returns the answer

sample\_codebase/ — put your own Python project files here



\---



⚠️ Notes



The ngrok URL changes every time you restart it. Remember to update the Server URL in your Vapi tool settings each time.



The .env file is not included in this repo for security. You need to create your own with the keys above.

