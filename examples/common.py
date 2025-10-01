import re
from typing import List

from markdown_it import MarkdownIt
from attachments.core import Attachment, presenter
from attachments import present


def clean_markdown_text(text_content: str) -> str:
    """
    Strip markdown syntax to plain text using markdown-it-py.
    """
    if not text_content:
        return ""

    md = MarkdownIt()

    # Parse markdown
    tokens = md.parse(text_content)

    lines = []

    for token in tokens:
        if token.type == "inline" and token.children:
            # Extract text from inline content, skipping formatting tokens
            text_parts = []
            for child in token.children:
                if child.type == "text":
                    text_parts.append(child.content)
                elif child.type == "code_inline":
                    text_parts.append(child.content)
            if text_parts:
                lines.append("".join(text_parts))

        elif token.type in ("code_block", "fence"):
            # Preserve code block content
            lines.append(token.content.strip())

        elif token.type in ("heading_close", "paragraph_close"):
            lines.append("")  # Add blank line after blocks

    # Join and clean up
    text = "\n".join(lines)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


@presenter
def markdown_to_text(att: Attachment) -> Attachment:
    """
    Strip markdown syntax to plain text using markdown-it-py.
    """
    # Get the markdown content from text_content (populated by load.text_to_string)
    text_content = att.text_content if hasattr(att, "text_content") else att.text

    # Use the extracted core logic
    att.text = clean_markdown_text(text_content)
    return att


present.register_new_function(markdown_to_text)


def split_into_sentences(text: str) -> List[str]:
    """Splits a text into sentences using a simple regex."""
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [s.strip() for s in sentences if s.strip()]


class GenerationError(Exception):
    """Raised when dataset generation fails."""

    pass


def load_embedder(embedder_config):
    """Dynamically load the embedder plugin."""
    import importlib

    embedder_type = embedder_config.type
    try:
        module_path = f"aclarai_claimify.embeddings.{embedder_type}"
        module = importlib.import_module(module_path)

        # Convention: class name is CamelCase version of module name (e.g., sentence_transformer -> SentenceTransformerEmbedder)
        class_name = (
            "".join(word.capitalize() for word in embedder_type.split("_")) + "Embedder"
        )
        embedder_class = getattr(module, class_name)

        return embedder_class(model_name=embedder_config.model)
    except (ImportError, AttributeError) as e:
        raise GenerationError(f"Failed to load embedder '{embedder_type}': {e}")
