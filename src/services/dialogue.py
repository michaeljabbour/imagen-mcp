"""
Simple dialogue system for conversational image generation.

Analyzes prompts and generates clarifying questions to help users
refine their image requests before generation.
"""

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DialogueResult:
    """Result from dialogue analysis."""

    should_generate: bool  # True if ready to generate, False if needs more info
    questions: list[str] = field(default_factory=list)
    enhanced_prompt: str | None = None
    detected_intent: str | None = None  # portrait, product, scene, etc.
    suggestions: list[str] = field(default_factory=list)


# Prompt patterns that indicate completeness (pre-compiled)
COMPLETE_PATTERNS = [
    re.compile(r"\d+x\d+", re.IGNORECASE),
    re.compile(r"(detailed|specific|exact|precise)", re.IGNORECASE),
    re.compile(r"(style of|in the style|like a)", re.IGNORECASE),
    re.compile(r"(lighting|lit by|illuminated)", re.IGNORECASE),
    re.compile(r"(background|backdrop|setting)", re.IGNORECASE),
]

# Vague terms that suggest more detail needed
VAGUE_TERMS = [
    "something",
    "anything",
    "whatever",
    "nice",
    "cool",
    "good",
    "interesting",
    "simple",
    "basic",
]

# Image type detection patterns
IMAGE_TYPE_PATTERNS = {
    "portrait": [
        "portrait",
        "headshot",
        "face",
        "person",
        "selfie",
        "profile",
        "character",
    ],
    "product": [
        "product",
        "item",
        "merchandise",
        "bottle",
        "package",
        "shoe",
        "watch",
    ],
    "scene": ["scene", "landscape", "environment", "setting", "place", "location"],
    "food": ["food", "dish", "meal", "recipe", "cuisine", "restaurant", "menu"],
    "abstract": ["abstract", "pattern", "texture", "geometric", "artistic"],
    "logo": ["logo", "brand", "icon", "emblem", "badge", "symbol"],
    "illustration": [
        "illustration",
        "drawing",
        "sketch",
        "cartoon",
        "comic",
        "anime",
    ],
}

# Questions by image type
TYPE_QUESTIONS = {
    "portrait": [
        "What mood or expression should the person have?",
        "What's the setting or background?",
        "Any specific lighting preference (studio, natural, dramatic)?",
    ],
    "product": [
        "What background would best showcase the product?",
        "Should there be any props or context items?",
        "What angle would you prefer (front, 3/4, top-down)?",
    ],
    "scene": [
        "What time of day should it be?",
        "What's the weather or atmosphere?",
        "Should there be any people or activity in the scene?",
    ],
    "food": [
        "What style of food photography (overhead, 45-degree, close-up)?",
        "What props or table setting should be included?",
        "What lighting mood (bright and airy, moody, warm)?",
    ],
    "abstract": [
        "What colors should dominate?",
        "What mood should it convey?",
        "Any specific geometric or organic shapes?",
    ],
    "logo": [
        "What industry or context is this for?",
        "Should it include text or be purely symbolic?",
        "What colors or style (minimal, detailed, vintage, modern)?",
    ],
    "illustration": [
        "What art style (realistic, stylized, flat, detailed)?",
        "What's the target audience or use case?",
        "What mood or tone should it convey?",
    ],
}

# Generic questions for unclear prompts
GENERIC_QUESTIONS = [
    "What's the primary subject or focus of the image?",
    "What style are you looking for (photorealistic, artistic, cartoon)?",
    "What mood or atmosphere should the image convey?",
    "Any specific colors or color palette preferences?",
]


class DialogueSystem:
    """
    Simple dialogue system for prompt refinement.

    Analyzes prompts and generates relevant clarifying questions
    based on detected image type and prompt completeness.
    """

    def __init__(self, mode: str = "guided"):
        """
        Initialize dialogue system.

        Args:
            mode: Dialogue mode
                - "skip": No dialogue, direct generation
                - "quick": 1-2 questions max
                - "guided": 3-5 questions (default)
                - "explorer": Deep exploration with 6+ questions
        """
        self.mode = mode
        self.max_questions = {
            "skip": 0,
            "quick": 2,
            "guided": 4,
            "explorer": 6,
        }.get(mode, 4)

    def analyze(self, prompt: str) -> DialogueResult:
        """
        Analyze a prompt and determine if more info is needed.

        Args:
            prompt: User's image generation prompt

        Returns:
            DialogueResult with questions or generation readiness
        """
        if self.mode == "skip":
            return DialogueResult(should_generate=True, enhanced_prompt=prompt)

        prompt_lower = prompt.lower()

        # Detect image type
        detected_type = self._detect_image_type(prompt_lower)

        # Check prompt completeness
        completeness_score = self._score_completeness(prompt_lower)

        # If prompt is very complete, generate directly
        if completeness_score >= 0.7:
            return DialogueResult(
                should_generate=True,
                enhanced_prompt=prompt,
                detected_intent=detected_type,
            )

        # Generate questions based on type and completeness
        questions = self._generate_questions(prompt_lower, detected_type, completeness_score)

        # Generate suggestions
        suggestions = self._generate_suggestions(prompt_lower, detected_type)

        # For quick mode or high completeness, generate with minimal questions
        if self.mode == "quick" or completeness_score >= 0.5:
            questions = questions[: self.max_questions]
            if not questions:
                return DialogueResult(
                    should_generate=True,
                    enhanced_prompt=prompt,
                    detected_intent=detected_type,
                    suggestions=suggestions,
                )

        return DialogueResult(
            should_generate=False,
            questions=questions[: self.max_questions],
            detected_intent=detected_type,
            suggestions=suggestions,
        )

    def _detect_image_type(self, prompt: str) -> str | None:
        """Detect the type of image being requested."""
        for img_type, keywords in IMAGE_TYPE_PATTERNS.items():
            if any(kw in prompt for kw in keywords):
                return img_type
        return None

    def _score_completeness(self, prompt: str) -> float:
        """
        Score how complete/specific a prompt is.

        Returns:
            Float from 0.0 (vague) to 1.0 (very specific)
        """
        score = 0.0
        word_count = len(prompt.split())

        # Length bonus (longer prompts tend to be more detailed)
        if word_count >= 20:
            score += 0.3
        elif word_count >= 10:
            score += 0.2
        elif word_count >= 5:
            score += 0.1

        # Check for complete patterns (already compiled with IGNORECASE)
        for pattern in COMPLETE_PATTERNS:
            if pattern.search(prompt):
                score += 0.15

        # Penalty for vague terms
        for term in VAGUE_TERMS:
            if term in prompt:
                score -= 0.1

        # Has color specification
        colors = [
            "red",
            "blue",
            "green",
            "yellow",
            "black",
            "white",
            "gold",
            "silver",
            "purple",
            "orange",
        ]
        if any(c in prompt for c in colors):
            score += 0.1

        return max(0.0, min(1.0, score))

    def _generate_questions(
        self, prompt: str, detected_type: str | None, completeness: float
    ) -> list[str]:
        """Generate relevant clarifying questions."""
        questions = []

        # Get type-specific questions
        if detected_type and detected_type in TYPE_QUESTIONS:
            type_qs = TYPE_QUESTIONS[detected_type].copy()
            # Filter out questions that the prompt might already answer
            for q in type_qs:
                # Simple check: skip if key terms from question are in prompt
                q_terms = set(q.lower().split()) - {"what", "should", "the", "a", "any", "be"}
                if not any(term in prompt for term in q_terms if len(term) > 3):
                    questions.append(q)

        # Add generic questions if needed
        if len(questions) < self.max_questions:
            for q in GENERIC_QUESTIONS:
                if q not in questions:
                    questions.append(q)
                if len(questions) >= self.max_questions:
                    break

        return questions

    def _generate_suggestions(self, prompt: str, detected_type: str | None) -> list[str]:
        """Generate suggestions to improve the prompt."""
        suggestions = []

        if not detected_type:
            suggestions.append(
                "Consider specifying the type of image (portrait, scene, product, etc.)"
            )

        word_count = len(prompt.split())
        if word_count < 10:
            suggestions.append("Adding more detail usually improves results")

        if not any(c in prompt for c in ["light", "lit", "bright", "dark", "shadow"]):
            suggestions.append("Describing lighting can significantly improve image quality")

        return suggestions[:3]  # Max 3 suggestions

    def enhance_prompt(self, original: str, answers: dict[str, str]) -> str:
        """
        Enhance a prompt with user's answers to clarifying questions.

        Args:
            original: Original prompt
            answers: Dict mapping questions to user answers

        Returns:
            Enhanced prompt incorporating answers
        """
        if not answers:
            return original

        # Simple enhancement: append relevant answers
        enhancements = []
        for _question, answer in answers.items():
            if answer and answer.lower() not in ["skip", "none", "no", "n/a"]:
                enhancements.append(answer)

        if enhancements:
            return f"{original}. {'. '.join(enhancements)}"
        return original


def create_dialogue_response(result: DialogueResult, provider: str | None = None) -> str:
    """
    Format a dialogue result as a user-friendly response.

    Args:
        result: DialogueResult from analysis
        provider: Selected provider name (optional)

    Returns:
        Formatted markdown response
    """
    if result.should_generate:
        lines = ["## Ready to Generate"]
        if result.detected_intent:
            lines.append(f"\n**Detected Type:** {result.detected_intent.title()}")
        if provider:
            lines.append(f"**Provider:** {provider.title()}")
        if result.suggestions:
            lines.append("\n**Tips for better results:**")
            for s in result.suggestions:
                lines.append(f"- {s}")
        return "\n".join(lines)

    lines = ["## Let's Refine Your Image"]

    if result.detected_intent:
        lines.append(f"\n*I see you want to create a {result.detected_intent}.*\n")

    lines.append("Please answer these questions to help me create the best image:\n")

    for i, q in enumerate(result.questions, 1):
        lines.append(f"{i}. {q}")

    if result.suggestions:
        lines.append("\n**Suggestions:**")
        for s in result.suggestions:
            lines.append(f"- {s}")

    lines.append("\n*Reply with your answers, or say 'generate' to proceed with current prompt.*")

    return "\n".join(lines)
