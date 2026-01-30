import re

def is_question(sentence: str):
    if sentence.strip().endswith("?"):
        return True
    
    question_words = r"(?i)^\s*(qui|que|qu'|quoi|quel(le)?s?|quand|où|pourquoi|comment|combien|est[- ]ce que)\b"
    if re.match(question_words, sentence.strip(), re.IGNORECASE):
        return True
    
    return False


