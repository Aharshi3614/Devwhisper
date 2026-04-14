&#x20;🏗️ DevWhisper — Architecture



​```mermaid

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



Flow explanation



1.Developer speaks a question out loud

2.Vapi converts speech to text and hits the FastAPI webhook

3.FastAPI calls the retriever which encodes the query as a vector

4.Qdrant finds the most semantically similar code chunks

5.The code context is sent to Groq LLaMA for reasoning

6.Groq returns a spoken-friendly answer

7.FastAPI sends it back to Vapi

8.Vapi speaks the answer to the developer

