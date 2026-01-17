import asyncio
import json
import logging
import sys
import time
import re
import argparse
from pathlib import Path
import os

import httpx
from loguru import logger
from habanero import Crossref
from rapidfuzz import fuzz
import arxiv
from tqdm.asyncio import tqdm
from glom import glom, Coalesce
from dotenv import load_dotenv
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- Utilities ---
def normalize_text(text):
    """Normalize text for comparison: lowercase, remove non-alphanumeric."""
    if not text:
        return ""
    return re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()

def extract_doi(text):
    """Extract DOI from text using regex."""
    match = re.search(r'10.\d{4,9}/[-._;()/:a-zA-Z0-9]+', text)
    return match.group(0) if match else None

def extract_arxiv_id(text):
    """Extract ArXiv ID from text."""
    match = re.search(r'arxiv[:\s]*(\d{4}\.\d{4,5})', text, re.IGNORECASE)
    return match.group(1) if match else None

def calculate_similarity(text1, text2):
    """Calculate fuzzy similarity score."""
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    return fuzz.token_set_ratio(norm1, norm2)

class TimerBlock:
    def __init__(self, label="Block"):
        self.label = label
        self.start = 0
        self.end = 0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        logger.info(f"{self.label} took {self.end - self.start:.4f} seconds")

# --- Services ---

class ArxivService:
    @staticmethod
    def fetch_and_score(ref_string):
        """
        Check for ArXiv ID and fetch metadata if found.
        Returns mapped result dict or None.
        """
        arxiv_id = extract_arxiv_id(ref_string)
        if not arxiv_id:
            return None

        try:
            client = arxiv.Client()
            search = arxiv.Search(id_list=[arxiv_id])
            results = list(client.results(search))
            
            if not results:
                return None
                
            res = results[0]
            
            # Map Authors
            authors_list = []
            for auth in res.authors:
                name_parts = auth.name.split()
                if len(name_parts) > 1:
                    authors_list.append({"given": " ".join(name_parts[:-1]), "family": name_parts[-1]})
                else:
                    authors_list.append({"family": auth.name})

            # Calculate Score
            arxiv_title = res.title
            arxiv_auth_str = " ".join([a.name for a in res.authors])
            compare_content = f"{arxiv_title} {arxiv_auth_str}"
            score = calculate_similarity(ref_string, compare_content)

            if score < 70:
                logger.warning(f"ArXiv ID {arxiv_id} found but text similarity low: {score}")

            return [{
                "title": [res.title],
                "author": authors_list,
                "reference-count": 0,
                "funder": [],
                "similarity_score": score,
                "DOI": res.doi if res.doi else "",
                "source": "arxiv_api"
            }]
            
        except Exception as e:
            logger.warning(f"ArXiv lookup failed for {arxiv_id}: {e}")
            return None

class WebVerifier:
    @staticmethod
    async def verify_doi_page(client, doi, title, authors):
        """
        Scrape DOI landing page to verify content.
        """
        if not doi:
            return False
            
        try:
            url = f"https://doi.org/{doi}"
            logger.info(f"Attempting web verification for DOI: {doi}")
            
            # Mimic browser
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
            }
            
            response = await client.get(url, headers=headers, follow_redirects=True)
            response.raise_for_status()
            
            page_text = normalize_text(response.text)
            norm_title = normalize_text(title)
            
            # Check Title (relaxed)
            title_tokens = norm_title.split()
            found_tokens = sum(1 for t in title_tokens if t in page_text)
            title_match = (found_tokens / len(title_tokens)) > 0.75 if title_tokens else False
            
            # Check Authors
            author_match = False
            for auth in authors:
                family = normalize_text(auth.get('family', ''))
                if family and family in page_text:
                    author_match = True
                    break
            
            if title_match and author_match:
                logger.info(f"Web verification SUCCESS for {doi}")
                return True
            else:
                logger.info(f"Web verification FAILED for {doi} (Title Match: {title_match}, Author Match: {author_match})")
                return False
                
        except Exception as e:
            logger.warning(f"Web verification error for {doi}: {e}")
            return False

class CrossrefService:
    def __init__(self, mailto):
        self.base_url = "https://api.crossref.org/works"
        self.mailto = mailto

    async def search(self, client, query, rows=5):
        params = {"query": query, "rows": rows}
        if self.mailto:
            params["mailto"] = self.mailto
            
        try:
            res = await client.get(self.base_url, params=params)
            res.raise_for_status()
            data = res.json()
            return data.get('message', {}).get('items', [])
        except Exception as e:
            logger.error(f"Crossref query failed: {repr(e)}")
            return []

    def rank_results(self, items, ref_string):
        """Calculate scores for items and sort them."""
        for item in items:
            # Extract content for matching
            title_list = item.get('title', [])
            title = " ".join(title_list) if isinstance(title_list, list) else str(title_list)
            
            authors_list = item.get('author', [])
            author_names = []
            for auth in authors_list:
                given = auth.get('given', '')
                family = auth.get('family', '')
                full = f"{given} {family}".strip()
                initial = f"{given[0]} {family}".strip() if given else ""
                author_names.append(f"{full} {initial}")
            
            author_str = " ".join(author_names)
            compare_content = f"{title} {author_str}"
            
            score = calculate_similarity(ref_string, compare_content)
            item['similarity_score'] = score
            
        # Sort descending
        items.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        return items

    def extract_data(self, items):
        """Extract standardized fields using Glom."""
        spec = {
            "title": Coalesce("title", default=[]),
            "author": Coalesce("author", default=[]),
            "reference-count": Coalesce("reference-count", default=0),
            "funder": Coalesce("funder", default=[]),
            "similarity_score": Coalesce("similarity_score", default=0),
            "DOI": Coalesce("DOI", default="")
        }
        return [glom(item, spec) for item in items]

class VerificationEngine:
    @staticmethod
    async def process(ref_string, arxiv_service, cr_service, web_verifier, client):
        # 1. Check ArXiv
        arxiv_result = arxiv_service.fetch_and_score(ref_string)
        if arxiv_result:
            return arxiv_result, True # ArXiv results are verified if score is handled (checked in service)
            # Actually service returns score. If score is low, is it verified?
            # My logic: Arxiv result includes score.
            # Decision to "Verify" (put in Verified folder) depends on score.
            # But here we just return the result.
            # Let's return (result, is_source_verified)
            # If it comes from ArXiv service, we treat it as valid lookup. 
            # The folder sorting logic is separate.
            return arxiv_result, True

        # 2. Crossref Search
        items = await cr_service.search(client, ref_string)
        if not items:
            return {"error": "No results"}, False

        # Rank
        ranked_items = cr_service.rank_results(items, ref_string)
        if not ranked_items:
             return {"error": "No results after ranking"}, False
             
        # Extract Standardized Data
        final_result = cr_service.extract_data(ranked_items)
        top_item = final_result[0]
        top_score = top_item.get('similarity_score', 0)

        # 3. Decision Logic
        is_verified = False

        # Tier 1: High Fuzzy Score
        if top_score >= 75:
            is_verified = True
        else:
            # Tier 2: Identifier Rescue
            found_doi = extract_doi(ref_string)
            found_arxiv = extract_arxiv_id(ref_string)
            result_doi = top_item.get('DOI', '').lower()

            if found_doi and found_doi.lower() in result_doi:
                logger.info(f"Verified via DOI match: {found_doi}")
                is_verified = True
            elif found_arxiv and found_arxiv in result_doi:
                logger.info(f"Verified via ArXiv ID match in DOI: {found_arxiv}")
                is_verified = True
            
            # Tier 3: Web Verification (Last Resort)
            elif result_doi and not is_verified:
                title = top_item.get('title', [''])[0] if top_item.get('title') else ""
                authors = top_item.get('author', [])
                
                if await web_verifier.verify_doi_page(client, result_doi, title, authors):
                    top_item['verification_method'] = "web_doi_check"
                    is_verified = True

        return final_result, is_verified

class Pipeline:
    def __init__(self, email, output_main):
        self.email = email
        self.output_main = Path(output_main)
        self.arxiv = ArxivService()
        self.crossref = CrossrefService(email)
        self.crossref = CrossrefService(email)
        self.web = WebVerifier()
        self.sem = asyncio.Semaphore(5)

    async def process_batch(self, ref_collection, prefix="ref", output_subfolder=None):
        results_path = self.output_main / "Results"
        if output_subfolder:
            results_path = results_path / output_subfolder
            
        verified_path = results_path / "Verified"
        manual_path = results_path / "Manual_Check"
        
        verified_path.mkdir(parents=True, exist_ok=True)
        manual_path.mkdir(parents=True, exist_ok=True)

        async with httpx.AsyncClient(headers={"User-Agent": f"PdfPrism ({self.email})"}, timeout=30.0) as client:
            tasks = []
            for i, ref in enumerate(ref_collection):
                tasks.append(self.process_single(client, ref, i, prefix, verified_path, manual_path))
            
            for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing", unit="ref"):
                 await f

    async def process_single(self, client, ref, index, prefix, verified_path, manual_path):
        async with self.sem: # Throttling
            try:
                result, is_verified = await VerificationEngine.process(ref, self.arxiv, self.crossref, self.web, client)
                
                # Add original query
                if isinstance(result, list):
                    for item in result:
                        item['original_query'] = ref
                elif isinstance(result, dict):
                    result['original_query'] = ref

                # Save
                folder = verified_path if is_verified else manual_path
                filename = folder / f"{prefix}_{index}.json"
                
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=4)
                    
            except Exception as e:
                logger.error(f"Error processing {index}: {e}")

class ReferenceHandler(FileSystemEventHandler):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def on_created(self, event):
        if event.is_directory:
            return
        if not event.src_path.lower().endswith('.txt'):
            return
            
        logger.info(f"New reference file detected: {event.src_path}")
        
        # Slight delay to ensure file write completes
        time.sleep(1)
        
        try:
             path = Path(event.src_path)
             # Logic to extract prefix
             if "_references" in path.name:
                prefix = path.name.split("_references")[0]
             else:
                prefix = path.stem
             
             # User requested output folder with same name as new text file
             output_subfolder = path.stem
             
             ref_collection = []
             with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        ref_collection.append(line.strip())
             
             logger.info(f"Auto-processing {len(ref_collection)} references from {path.name}...")
             # Run sync wrapper for async function
             asyncio.run(self.pipeline.process_batch(ref_collection, prefix=prefix, output_subfolder=output_subfolder))
             logger.info(f"Finished processing {path.name}")
             
        except Exception as e:
            logger.error(f"Error processing new file {event.src_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crossref API Reference Verifier")
    parser.add_argument("file", nargs="?", help="Input reference file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose console output")
    parser.add_argument("--monitor", action="store_true", help="Monitor data/samples folder for new files")
    args = parser.parse_args()

    # Configure Logger
    logger.remove()
    logger.add("crossref_runtime.log", rotation="1 MB")
    
    if args.verbose or args.monitor:
        logger.add(
            sys.stderr, 
            format="<green>{time:HH:mm:ss}</green> | <cyan>Line:{line}</cyan> | <level>{message}</level>"
        )

    logger.info("Starting Processing")
    if not args.verbose and not args.monitor:
        logger.info("Quiet mode enabled (default). Use --verbose to see logs in console.")
    
    # Load environment variables
    load_dotenv()
    
    EMAIL = os.getenv('CROSSREF_EMAIL')
    if not EMAIL:
        logger.warning("CROSSREF_EMAIL not set in .env file. API calls may fail or be throttled.")
    else:
        logger.info(f"Loaded CROSSREF_EMAIL: {EMAIL[:3]}***{EMAIL[-3:]}")
    
    # Resolve to project root for correct Results directory placement
    base_dir = Path(__file__).resolve().parent.parent
    pipeline = Pipeline(EMAIL, base_dir)

    if args.monitor:
        monitor_dir = base_dir / "data" / "samples"
        if not monitor_dir.exists():
            monitor_dir.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Monitoring directory: {monitor_dir}")
        logger.info("Press Ctrl+C to stop.")
        
        event_handler = ReferenceHandler(pipeline)
        observer = Observer()
        observer.schedule(event_handler, str(monitor_dir), recursive=False)
        observer.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()
        sys.exit(0)

    # Input Logic
    if args.file:
        ref_file = Path(args.file).resolve()
    else:
        # Default
        ref_file = base_dir / "data" / "samples" / "2312.01797v3_references.txt"

    if ref_file.exists():
        logger.info(f"Reading references from {ref_file}")
        
        # Prefix Extraction
        filename = ref_file.name
        if "_references" in filename:
            prefix = filename.split("_references")[0]
        else:
            prefix = ref_file.stem
            
        logger.info(f"Using prefix: {prefix}")
            
        ref_collection = []    
        with open(ref_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ref_collection.append(line.strip())
        asyncio.run(pipeline.process_batch(ref_collection, prefix))
    else:
        logger.error(f"Reference file not found: {ref_file}")
        sys.exit(1)