# Agentic AI QnA

### What is an agent in LangChain?
- An agent is a framework in LangChain that decides **what actions to take** based on user input.
- It can use **tools** like search or calculators.
- Agents rely on an LLM to choose the next tool or action based on intermediate results.

---
### What tools did your AI support agent use?
- A **retriever tool** (using FAISS vector store) to pull answers from product docs.
- A fallback tool to **escalate** queries to humans (mocked as a log function).
- Optionally, a **search tool** to simulate looking up live data.
---

### What is Retrieval-Augmented Generation (RAG)?
- RAG combines **retrieved documents** with an **LLM response**.
- Helps ground LLM answers in trusted content (like internal FAQs).
- Reduces hallucination risk and improves accuracy in niche domains.

---
### How did you store and use memory in the agent?
- I used **ConversationBufferMemory** to retain the chat history.
- This helped the bot maintain context across multiple user turns.
- LangChain allows easily injecting memory into agents or chains.