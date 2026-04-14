&#x20;🏗️ DevWhisper — Architecture



Developer speaks a question out loud

&#x20;             ↓

&#x20;      Vapi — Speech to Text

&#x20;             ↓

&#x20;   FastAPI Webhook Server (main.py)

&#x20;        ↓              ↓

&#x20;  Retriever        sends context

&#x20;  (retriever.py)       ↓

&#x20;        ↓         Groq LLaMA 3.3 70B

&#x20;  Qdrant Vector        ↓

&#x20;  Database        spoken answer

&#x20;        ↓              ↓

&#x20;  top code chunks → FastAPI

&#x20;                        ↓

&#x20;                  Vapi — Text to Speech

&#x20;                        ↓

&#x20;             Developer hears the answer



Flow :

1\. Developer speaks a question out loud

2\. Vapi transcribes speech and triggers the FastAPI webhook

3\. FastAPI calls the retriever which encodes the query as a vector

4\. Qdrant finds the most semantically similar code chunks

5\. The code context is passed to Groq LLaMA for reasoning

6\. Groq returns a concise spoken-friendly answer

7\. FastAPI sends the answer back to Vapi

8\. Vapi speaks the answer back to the developer

