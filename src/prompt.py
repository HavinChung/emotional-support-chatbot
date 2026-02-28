SYSTEM_PROMPT = """You are a compassionate emotional support chatbot. Your role is to:
- Listen actively and validate the user's feelings
- Respond with empathy and genuine understanding
- Ask follow-up questions to help the user feel heard
- Keep responses concise and focused (2-4 sentences max)
- Only suggest coping strategies when the user asks for help or advice
- Prioritize listening over advising

You must NOT:
- Diagnose any mental health conditions
- Prescribe or recommend medications
- Replace professional therapy or counseling
- Give long lists of advice unless explicitly asked
"""

def build_prompt(user_message, emotion, retrieved_docs, conversation_history=None):
    context = ""
    for _, doc in retrieved_docs.iterrows():
        if doc['strategies']:
            context += f"- Emotion: {doc['emotion']}, Strategies: {doc['strategies']}\n"
        if doc['source'] != 'esconv':
            context += f"- Reference: {doc['text'][:300]}\n"

    history = ""
    if conversation_history:
        for turn in conversation_history[-4:]:
            history += f"{turn['role'].upper()}: {turn['content']}\n"

    return f"""Detected emotion: {emotion}

Relevant knowledge:
{context}

Conversation history:
{history}

User: {user_message}

Respond with empathy and provide emotional support. Use the relevant knowledge to guide your response."""