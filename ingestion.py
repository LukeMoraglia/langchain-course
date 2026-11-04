import asyncio
import os
import random
import re
import ssl
from typing import Any, Dict, List



import certifi
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import Colors, log_error, log_header, log_info, log_success, log_warning

load_dotenv()

# Configure SSL 
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=True, 
    chunk_size=50, 
    retry_min_seconds=10
)

vectorstore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


def _retry_delay_seconds(exc: Exception, retry_number: int) -> float:
    """Honor a server-provided wait time, otherwise use capped exponential backoff."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) or {}

    # OpenAI may send either of these headers. The exception text may also include
    # a precise suggestion such as "Please try again in 1.7s".
    for header in ("retry-after-ms", "Retry-After"):
        value = headers.get(header) or headers.get(header.lower())
        if value:
            try:
                delay = float(value) / (1000 if header.lower().endswith("-ms") else 1)
                return max(0.1, delay) + random.uniform(0.1, 0.5)
            except (TypeError, ValueError):
                pass

        match = re.search(r"try again in ([0-9]+(?:\.[0-9]+)?)\s*s", str(exc), re.IGNORECASE)

    if match:
        return float(match.group(1)) + random.uniform(0.1, 0.5)

    # Full jitter prevents retries from multiple workers synchronizing.
    return random.uniform(0, min(60.0, 1.0 * (2 ** retry_number)))


def _is_retryable_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is not None:
        return status in (408, 429, 500, 502, 503, 504)

    message = str(exc).lower()
    return "429" in message or "rate limit" in message or "temporarily unavailable" in message


async def index_documents_async(
    documents: List[Document],
    batch_size: int = 50,
    max_concurrency: int = 1,
    max_retries: int = 8,
):
    """Index batches with bounded concurrency and retries for transient API errors."""

    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )

    # Create batches
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

            # Avoid sending all batches at once: that creates a large TPM burst. Keep the
    # semaphore even if concurrency is raised so retrying workers remain bounded.

    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def add_batch(batch: List[Document], batch_num: int):
        async with semaphore:
            for retry_number in range(max_retries + 1):
                try:
                    await vectorstore.aadd_documents(batch)
                    log_success(
                        f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
                    )
                    return True
                except Exception as exc:
                    if not _is_retryable_error(exc) or retry_number >= max_retries:
                        log_error(
                            f"VectorStore Indexing: Failed to add batch {batch_num} "
                            f"after {retry_number + 1} attempt(s) - {exc}"
                        )
                        return False

                    delay = _retry_delay_seconds(exc, retry_number)
                    log_warning(
                        f"VectorStore Indexing: Batch {batch_num} hit a temporary error; "
                        f"retrying in {delay:.1f}s (attempt {retry_number + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)

    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks)
    successful = sum(results)

    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        failed = len(batches) - successful
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully; "
            f"{failed} batch(es) failed"
        )
        raise RuntimeError(f"Vector indexing incomplete: {failed} batch(es) failed")


async def main():
    "Main async to orchestrate whole thing"
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "🗺️  TavilyCrawl: Starting to crawl the documentation site",
        Colors.PURPLE,
    )

    res = tavily_crawl.invoke(
        {
            "url": "https://python.langchain.com/",
            "max_depth": 3,
            "extract_depth": "advanced",
        }
    )

    all_docs = [Document(page_content=result['raw_content'], metadata={"source": result['url']}) for result in res['results'] if result['raw_content'] is not None]

    log_success(f"Successfully crawled {len(all_docs)} pages")

    # Split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"✂️  Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )


    # Process documents async
            # One 50-document batch maps to one embeddings request with the configured
    # OpenAIEmbeddings chunk_size, helping smooth token usage across the minute.

    await index_documents_async(splitted_docs, batch_size=50, max_concurrency=1)



    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(main())
