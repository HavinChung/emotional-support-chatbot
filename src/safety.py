CRISIS_KEYWORDS = [
    "suicide", "kill myself", "want to die", "end my life", "self-harm",
    "cut myself", "hurt myself", "no reason to live", "better off dead",
    "can't go on", "don't want to be here anymore",
    "ending my life", "end it all", "end my suffering", "end it tonight",
    "disappear forever", "never come back", "goodbye letter",
    "stockpiling pills", "take all my pills", "hanging myself",
    "drive my car off", "jump off", "jumping off",
    "nobody would miss me", "world would be better without me",
    "better off without me", "burden to everyone",
    "fantasizing about dying", "imagining my own death",
    "hurting myself", "cutting myself", "self-harming",
    "don't want to wake up", "too tired to keep fighting",
    "giving away my belongings", "said goodbye",
    "no point in living", "can't see a future",
    "completely hopeless and suicidal", "feel like dying",
    "don't feel safe around myself", "already dead inside",
    "dying is the only way", "dying is the kindest",
    "keep fighting to stay alive", "courage to end it"
]

UNSAFE_RESPONSE_PATTERNS = [
    "you have", "you are diagnosed", "you should take", "medication",
    "prescribe", "you suffer from", "your diagnosis"
]

CRISIS_RESPONSE = """I'm really concerned about what you've shared. Your life has value and you deserve support.

Please reach out to a crisis helpline immediately:
- International: https://www.befrienders.org
"""

def safety_triage(user_message):
    message_lower = user_message.lower()
    for keyword in CRISIS_KEYWORDS:
        if keyword in message_lower:
            return True, CRISIS_RESPONSE
    return False, None

def post_generation_filter(response):
    response_lower = response.lower()
    for pattern in UNSAFE_RESPONSE_PATTERNS:
        if pattern in response_lower:
            return False, "Response contains unsafe content"
    return True, None

def safe_pipeline_run(pipeline, user_message, emotion):
    is_crisis, crisis_response = safety_triage(user_message)

    response, retrieved_docs = pipeline.run(user_message, emotion)
    is_safe, reason = post_generation_filter(response)

    if not is_safe:
        response = "I'm here to support you. Could you tell me more about how you're feeling?"

    if is_crisis:
        final_response = crisis_response + "\n\n" + response
        return final_response, retrieved_docs, "crisis"

    return response, retrieved_docs, "ok"