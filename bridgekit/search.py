import os
from pathlib import Path
import anthropic
from .config import DEFAULT_MODEL
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

CHUNK_SIZE = 150  # words per chunk
CHUNK_OVERLAP = 20


def _load_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import pypdf
        reader = pypdf.PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif suffix == ".docx":
        import docx
        doc = docx.Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    elif suffix == ".pptx":
        from pptx import Presentation
        prs = Presentation(str(path))
        lines = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    lines.append(shape.text)
        return "\n".join(lines)
    elif suffix == ".ipynb":
        import nbformat
        nb = nbformat.read(str(path), as_version=4)
        lines = []
        for cell in nb.cells:
            if cell.cell_type in ("markdown", "code") and cell.source.strip():
                lines.append(cell.source)
        return "\n\n".join(lines)
    else:
        return path.read_text(encoding="utf-8")


def _chunk(text: str) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + CHUNK_SIZE]))
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if c.strip()]


def ask(question: str, source: str = None, text: str = None) -> str:
    """
    Ask a question across a collection of analysis documents or raw text.

    Args:
        question: The question to answer.
        source:   Path to a folder containing .txt, .md, .pdf, .docx, .pptx, or .ipynb files.
        text:     A raw text string to search instead of a folder.

    Returns:
        An answer grounded in the provided documents.
    """
    if not source and not text:
        raise ValueError("Provide either 'source' (folder path) or 'text'.")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not found. Set it with: export ANTHROPIC_API_KEY=your_key_here"
        )

    # Collect chunks
    chunks = []

    if text:
        chunks.extend(_chunk(text))

    if source:
        folder = Path(source).expanduser().resolve()
        supported = {".txt", ".md", ".pdf", ".docx", ".pptx", ".ipynb"}
        for file in sorted(folder.iterdir()):
            if file.suffix.lower() in supported:
                content = _load_file(file)
                chunks.extend(_chunk(content))

    if not chunks:
        raise ValueError("No content found. Check your source folder or text input.")

    # Embed and store in ChromaDB
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client = chromadb.Client()
    collection = client.get_or_create_collection(
        name="bridgekit_ask",
        embedding_function=embedding_fn
    )
    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    # Retrieve most relevant chunks
    results = collection.query(query_texts=[question], n_results=min(8, len(chunks)))
    context = "\n\n".join(results["documents"][0])

    # Generate answer with Claude
    anthropic_client = anthropic.Anthropic(api_key=api_key)
    message = anthropic_client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=1024,
        system=(
            "You are a senior data scientist answering questions based on analysis reports. "
            "Answer only from the provided context. Be specific and cite findings where relevant. "
            "If the context does not contain enough information to answer, say so clearly."
        ),
        messages=[{
            "role": "user",
            "content": f"Context from analysis reports:\n\n{context}\n\nQuestion: {question}"
        }]
    )

    return message.content[0].text
