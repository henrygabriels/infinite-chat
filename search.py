import re
from typing import List, Dict, Any, Tuple
from difflib import SequenceMatcher

class FuzzySearch:
    def __init__(self):
        pass

    def fuzzy_match_score(self, pattern: str, text: str) -> float:
        """Calculate fuzzy match score (0-1) similar to fzf algorithm."""
        pattern = pattern.lower()
        text = text.lower()

        # Direct character sequence matching (fzf-style)
        text_index = 0
        pattern_index = 0
        matches = []

        while pattern_index < len(pattern) and text_index < len(text):
            if pattern[pattern_index] == text[text_index]:
                matches.append(text_index)
                pattern_index += 1
            text_index += 1

        # If we couldn't match all pattern characters
        if pattern_index < len(pattern):
            return 0.0

        # Calculate score based on contiguity and position
        if not matches:
            return 0.0

        # Bonus for contiguous matches
        contiguity_bonus = 0
        for i in range(1, len(matches)):
            if matches[i] == matches[i-1] + 1:
                contiguity_bonus += 1

        # Bonus for matches at word boundaries
        word_boundary_bonus = 0
        for match_pos in matches:
            if match_pos == 0 or text[match_pos-1] in ' \t\n_-':
                word_boundary_bonus += 1

        # Prefer earlier matches
        early_match_bonus = max(0, 1 - matches[0] / len(text))

        # Calculate final score
        base_score = len(matches) / len(pattern)
        final_score = (
            base_score * 0.5 +
            (contiguity_bonus / len(pattern)) * 0.3 +
            (word_boundary_bonus / len(pattern)) * 0.1 +
            early_match_bonus * 0.1
        )

        return min(final_score, 1.0)

    def extract_snippet(self, content: str, match_start: int, match_end: int, context_sentences: int = 1) -> str:
        """Extract snippet around matched text with sentence boundaries."""
        # Split into sentences (simple split on . ! ?)
        sentences = re.split(r'([.!?]+)', content)

        # Combine punctuation with sentences
        full_sentences = []
        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                full_sentences.append(sentences[i] + sentences[i+1])
            else:
                full_sentences.append(sentences[i])

        # Find which sentence contains the match
        char_count = 0
        target_sentence_idx = 0

        for i, sentence in enumerate(full_sentences):
            sentence_end = char_count + len(sentence)
            if char_count <= match_start <= sentence_end:
                target_sentence_idx = i
                break
            char_count = sentence_end

        # Extract sentences around the target
        start_idx = max(0, target_sentence_idx - context_sentences)
        end_idx = min(len(full_sentences), target_sentence_idx + context_sentences + 1)

        snippet = ''.join(full_sentences[start_idx:end_idx]).strip()

        # Truncate if too long
        if len(snippet) > 300:
            snippet = snippet[:300] + "..."

        return snippet

    def find_match_positions(self, pattern: str, text: str) -> List[Tuple[int, int]]:
        """Find all fuzzy match positions in text."""
        pattern = pattern.lower()
        text_lower = text.lower()
        matches = []

        # Simple sliding window approach for potential matches
        window_size = max(len(pattern), 3)

        for i in range(len(text_lower) - window_size + 1):
            window = text_lower[i:i + window_size + 10]  # Add some buffer

            if self.fuzzy_match_score(pattern, window) > 0.5:
                # Find the actual match boundaries in this window
                pattern_chars = list(pattern)
                window_chars = list(window)
                match_positions = []
                p_idx = 0

                for w_idx, char in enumerate(window_chars):
                    if p_idx < len(pattern_chars) and char == pattern_chars[p_idx]:
                        match_positions.append(i + w_idx)
                        p_idx += 1

                if p_idx == len(pattern_chars):  # Full match found
                    matches.append((match_positions[0], match_positions[-1] + 1))

        return matches

    def search_messages(self, messages: List[Dict[str, Any]], query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search messages using fuzzy matching, return snippets."""
        results = []

        for message in messages:
            content = message['content']
            matches = self.find_match_positions(query, content)

            if matches:
                # Use the best match (highest score)
                best_match = matches[0]
                snippet = self.extract_snippet(content, best_match[0], best_match[1])
                score = self.fuzzy_match_score(query, content)

                results.append({
                    'message_id': message['id'],
                    'snippet': snippet,
                    'timestamp': message['timestamp'],
                    'role': message['role'],
                    'score': score,
                    'match_positions': best_match
                })

        # Sort by score and limit results
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]

    def expand_context(self, messages: List[Dict[str, Any]], message_id: str,
                      direction: str = "both", pairs: int = 3) -> List[Dict[str, Any]]:
        """Expand context around a specific message."""
        # Find the target message index
        target_idx = -1
        for i, msg in enumerate(messages):
            if msg['id'] == message_id:
                target_idx = i
                break

        if target_idx == -1:
            return []

        result = [messages[target_idx]]  # Start with the target message

        if direction in ["before", "both"]:
            # Add pairs before the target
            before_start = max(0, target_idx - pairs * 2)
            before_messages = messages[before_start:target_idx]
            result = before_messages + result

        if direction in ["after", "both"]:
            # Add pairs after the target
            after_end = min(len(messages), target_idx + 1 + pairs * 2)
            after_messages = messages[target_idx + 1:after_end]
            result = result + after_messages

        return result