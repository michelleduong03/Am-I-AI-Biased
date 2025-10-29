# Type Token Ratio Script
import re

def type_token_ratio(text):
    # Lowercase and remove punctuation
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0

    total_tokens = len(words)
    unique_types = len(set(words))
    ttr = unique_types / total_tokens
    return ttr

# Example usage
sample_text = """
In times long past and places veiled in mist, there once was a quiet village nestled between rolling hills blanketed with verdant meadows under a sky of endless blue. The villagers led simple lives full of song and dance during harvest festivals but always yearned for adventure beyond their known horizons. One evening as twilight embraced the land, fires crackling softly against encroaching darkness and children playing hide-and-seek near homes whispering secrets through chimneys, something miraculous unfolded—a light streaked across the heavens like liquid silver cutting into night itself. Whispers rose among men and women alike; they spoke ancient tongues not heard since days when magic still roamed free unbridled by skepticism’s evergreen scorn. A young maiden named Lila listened intently from her open windowpane below, senses tingling with every shimmer, daring thoughts spinning within herself like leaves caught upward on wind's whimsical choreography. Tonight might just be forever etched amongst legends; tomorrow could awaken history reborn. And so it began...
"""
ratio = type_token_ratio(sample_text)

print(f"Type-Token Ratio: {ratio:.3f}")