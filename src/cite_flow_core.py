import asyncio
import time
import sys
from pathlib import Path
from contextlib import contextmanager
import re

import httpx
from loguru import logger
from habanero import Crossref
from rapidfuzz import fuzz
import arxiv
from tqdm.asyncio import tqdm
import argparse
import json
from glom import glom, Coalesce

# Load environment variables
from dotenv import load_dotenv
import os

class TimerBlock:
    def __init__(self, label="Block"):
        self.label = label

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        # Using logger to ensure it goes to file and respects quiet mode
        logger.info(f"{self.label} took {self.end - self.start:.4f} seconds")



class CrossrefClient:
    def __init__(self, mailto, output_dir=None, concurrency=1):
        self.mailto = mailto
        self.headers = {
            "User-Agent": f"Ref_Analysis/1.0 (mailto:{mailto})"
        }
        self.cr_base_link = "https://api.crossref.org/works"
        self.sem = asyncio.Semaphore(concurrency)
        
        if output_dir is None:
             self.base_path = Path(__file__).resolve().parent.parent      
             self.output_dir = self.base_path / "Results"
        else:
             self.output_dir = Path(output_dir)
             
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        self.cr_analysis_list = self.output_dir / "cr_analysis_list.txt"
        # self.cr_master_list = self.output_dir / "cr_master_list.txt" # Unused in original script

    @contextmanager
    def managed_file(self, filename, mode='w', encoding="utf-8"):
        logger.info(f"Opening file: {filename}")
        f = open(filename, mode, encoding="utf-8")
        try:
            yield f
        finally:
            f.close()
            logger.info(f"File {filename} closed successfully.")

    def write_to_file(self, file_path, data):
        with self.managed_file(file_path, mode="a", encoding="utf-8") as f:
            f.write(data)

    async def verify_via_doi_page(self, client, doi, title, authors):
        if not doi:
            return False
            
        try:
            # DOI resolution url
            url = f"https://doi.org/{doi}"
            logger.info(f"Attempting web verification for DOI: {doi}")
            
            # Use specific headers to look like a browser to avoid bot detection/content negotiation
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
            }
            
            res = await client.get(url, headers=headers, follow_redirects=True, timeout=10.0)
            if res.status_code != 200:
                return False
                
            content = res.text.lower()
            
            # 1. Check Title (normalized)
            # Remove punctuation for better matching
            def clean(t): return re.sub(r'[^a-z0-9\s]', '', t.lower())
            
            clean_title = clean(title)
            # Split into significant words (skip 'a', 'the', etc if needed, but for now simple check)
            # Checking exact phrase presence might be too strict due to HTML formatting, 
            # so let's check token set intersection or fuzzy? 
            # User asked to "check title", let's try token set ratio > 80? or presence of sufficient keywords.
            # Let's use simple presence of the main title string (ignoring spaces/punctuation)
            
            # Actually, `fuzz.partial_ratio` or `token_set_ratio` against the whole page is expensive/noisy.
            # Let's search for the clean title in the clean content.
            clean_content = clean(content)
            
            # Check title tokens presence - heuristic: 80% distinct words found?
            title_tokens = set(clean_title.split())
            if not title_tokens:
                return False
                
            found_tokens = sum(1 for t in title_tokens if t in clean_content)
            title_match_ratio = found_tokens / len(title_tokens)
            
            if title_match_ratio < 0.75: # Strict-ish
                logger.debug(f"Web/DOI Title mismatch: {title_match_ratio:.2f}")
                return False
                
            # 2. Check Author (at least the first one or surname)
            if authors:
                # content is already lower
                # Just check if at least one surname is present
                found_author = False
                for auth in authors:
                    family = auth.get('family', '').lower()
                    if family and family in clean_content:
                        found_author = True
                        break
                
                if not found_author:
                    logger.debug("Web/DOI Author mismatch")
                    return False
            
            logger.info(f"Web verification passed for {doi}")
            return True
            
        except Exception as e:
            logger.warning(f"DOI Web verification failed for {doi}: {e}")
            return False

    async def safe_ref_search(self, client, ref_item, index=None, prefix="ref"):
        async with self.sem:
            with TimerBlock("Async query"):
                final_result = None
                
                # Check for ArXiv ID first
                arxiv_match = re.search(r'arxiv[:\s]*(\d{4}\.\d{4,5}(v\d+)?)', ref_item, re.IGNORECASE)
                if arxiv_match:
                    arxiv_id = arxiv_match.group(1)
                    try:
                        # Use arxiv library to fetch details
                        search = arxiv.Search(id_list=[arxiv_id])
                        # Use Client explicitly to avoid deprecation warning
                        arxiv_client = arxiv.Client()
                        results = list(arxiv_client.results(search))
                        if results:
                            res = results[0]
                            # Map arXiv result to our schema
                            authors_list = []
                            for auth in res.authors:
                                name_parts = auth.name.split()
                                if len(name_parts) > 1:
                                    authors_list.append({"given": " ".join(name_parts[:-1]), "family": name_parts[-1]})
                                else:
                                    authors_list.append({"family": auth.name})
                                    
                            arxiv_result = None
                            
                            # Calculate similarity to verify it's not a wrong ID lookup
                            def normalize_text(text):
                                return re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()
                                
                            arxiv_title = res.title
                            # Create a simple author string for matching
                            arxiv_auth_str = " ".join([a.name for a in res.authors])
                            compare_content = f"{arxiv_title} {arxiv_auth_str}"
                            
                            # Normalize
                            norm_ref = normalize_text(ref_item)
                            norm_content = normalize_text(compare_content)
                            
                            # Calculate score
                            score = fuzz.token_set_ratio(norm_ref, norm_content)
                            
                            if score < 70:
                                logger.warning(f"ArXiv ID {arxiv_id} found but text similarity low: {score}")

                            final_result = [{
                                "title": [res.title],
                                "author": authors_list,
                                "reference-count": 0, # Not available in basic arXiv API
                                "funder": [],
                                "similarity_score": score, 
                                "DOI": res.doi if res.doi else "",
                                "source": "arxiv_api"
                            }]
                            logger.info(f"Resolving via ArXiv API for ID: {arxiv_id} (Score: {score})")
                    except Exception as e:
                        logger.warning(f"ArXiv API lookup failed for {arxiv_id}: {e}")

                if not final_result:
                    params = {"query": ref_item, "rows": 5}
                    try:
                        res = await client.get(self.cr_base_link, params=params)
                        res.raise_for_status()
                        data = res.json()
                        
                        # Fuzzy verification for top 5 results
                        items = data.get('message', {}).get('items', [])
                        # Normalize by removing non-alphanumeric (except spaces) and lowercasing
                        def normalize(text):
                            return re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()

                        for item in items:
                            # Define spec for extraction needed for matching
                            temp_spec = {
                                "title": Coalesce("title", default=[]),
                                "author": Coalesce("author", default=[])
                            }
                            extracted_for_match = glom(item, temp_spec)

                            # Construct comparison string (Title + Authors)
                            title_list = extracted_for_match.get('title', [])
                            title = " ".join(title_list) if isinstance(title_list, list) else str(title_list)
                            
                            authors_list = extracted_for_match.get('author', [])
                            author_names = []
                            for auth in authors_list:
                                given = auth.get('given', '')
                                family = auth.get('family', '')
                                # Add Initial + Family as well
                                initial = given[0] if given else ""
                                
                                # Variants to help matching: Full Name & Initial + Family
                                full = f"{given} {family}".strip()
                                abbrev = f"{initial} {family}".strip()
                                
                                author_names.append(f"{full} {abbrev}")
                            
                            author_str = " ".join(author_names)
                            
                            custom_content = f"{title} {author_str}"
                            
                            # Normalize both sides (ref_item and custom_content)
                            norm_ref = normalize(ref_item)
                            norm_content = normalize(custom_content)
                            
                            # Use token_set_ratio for best partial match
                            score = fuzz.token_set_ratio(norm_ref, norm_content)
                            item['similarity_score'] = score
                        
                        # Re-sort items based on fuzzy similarity score
                        items.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)

                        if items:
                            best_item = items[0]
                            best_score = best_item['similarity_score']
                            best_title = best_item.get('title', [''])[0]
                            
                            if best_score < 70:
                                 logger.warning(f"Low similarity ({best_score}) for top match: {ref_item[:30]}... vs {best_title[:30]}...")
                            else:
                                 logger.info(f"Best similarity score: {best_score}")

                            spec = {
                                "title": Coalesce("title", default=[]),
                                "author": Coalesce("author", default=[]),
                                "reference-count": Coalesce("reference-count", default=0),
                                "funder": Coalesce("funder", default=[]),
                                "similarity_score": Coalesce("similarity_score", default=0),
                                "DOI": Coalesce("DOI", default="")
                            }
                            
                            # Apply glom to all items
                            extracted_items = [glom(item, spec) for item in items]
                            logger.debug(json.dumps(extracted_items, indent=4))

                            logger.info(f"Query success for: {ref_item[:30]}...") 
                            final_result = extracted_items
                        else:
                            logger.info(f"Query success for: {ref_item[:30]}...") 
                            final_result = data
                    except Exception as e:
                        logger.error(f"Error querying {ref_item[:30]}...: {e}")
                        final_result = {"error": str(e), "query": ref_item}

                if index is not None:
                    # Determine save location based on score OR identifier match
                    is_verified = False
                    if isinstance(final_result, list) and len(final_result) > 0:
                        top_item = final_result[0]
                        top_score = top_item.get('similarity_score', 0)
                        
                        # Check 1: Fuzzy Score
                        if top_score >= 75:
                            is_verified = True
                        else:
                            # Check 2: Identifier Verification (Rescue)
                            # Extract DOI from reference string
                            doi_match = re.search(r'10.\d{4,9}/[-._;()/:a-zA-Z0-9]+', ref_item)
                            found_doi = doi_match.group(0) if doi_match else None
                            
                            # Extract ArXiv ID from reference string (e.g., arXiv:2312.01797)
                            arxiv_match = re.search(r'arxiv[:\s]*(\d{4}\.\d{4,5})', ref_item, re.IGNORECASE)
                            found_arxiv = arxiv_match.group(1) if arxiv_match else None

                            result_doi = top_item.get('DOI', '').lower()
                            
                            if found_doi and found_doi.lower() in result_doi:
                                logger.info(f"Verified via DOI match: {found_doi}")
                                is_verified = True
                            elif found_arxiv and found_arxiv in result_doi:
                                logger.info(f"Verified via ArXiv ID match: {found_arxiv}")
                                is_verified = True
                            
                            # Check 3: Web Verification via DOI (LAST RESORT)
                            # Only runs if Fuzzy Score < 75 AND no direct ID match in reference string
                            elif result_doi and not is_verified:
                                top_title = top_item.get('title', [''])[0] if top_item.get('title') else ""
                                top_authors = top_item.get('author', [])
                                
                                if await self.verify_via_doi_page(client, result_doi, top_title, top_authors):
                                     is_verified = True
                                     top_item['verification_method'] = "web_doi_check"
                    
                    sub_dir = "Verified" if is_verified else "Manual_Check"
                    save_dir = self.output_dir / sub_dir
                    save_dir.mkdir(exist_ok=True)
                    
                    output_file = save_dir / f"{prefix}_{index}.json"
                    
                    # Add original query if not present
                    if isinstance(final_result, list):
                        for item in final_result:
                            item['original_query'] = ref_item
                    elif isinstance(final_result, dict):
                         final_result['original_query'] = ref_item

                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(final_result, f, indent=4)
                
                return final_result

    async def find_correct_record(self, ref_collection, prefix="ref"):
        async with httpx.AsyncClient(headers=self.headers, timeout=30.0) as client:
            ref_tasks = [self.safe_ref_search(client, ref, i, prefix) for i, ref in enumerate(ref_collection)]
            for task in tqdm(asyncio.as_completed(ref_tasks), total=len(ref_tasks), desc="Processing", unit="ref"):
                result = await task
                self.write_to_file(self.cr_analysis_list, f"{json.dumps(result, indent=4)}\n")
    
    def process_references(self, ref_collection, prefix="ref"):
         with TimerBlock("All queries"):
            asyncio.run(self.find_correct_record(ref_collection, prefix))

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Crossref API Reference Verifier")
    parser.add_argument("file", nargs="?", help="Input reference file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose console output")
    args = parser.parse_args()

    # Configure Logger
    logger.remove()
    # File sink
    logger.add("crossref_runtime.log", rotation="1 MB")
    
    # Console sink (only if verbose)
    if args.verbose:
        logger.add(
            sys.stderr, 
            format="<green>{time:HH:mm:ss}</green> | <cyan>Line:{line}</cyan> | <level>{message}</level>"
        )

    logger.info("This log will show the line number.")
    if not args.verbose:
        logger.info("Quiet mode enabled (default). Use --verbose to see logs in console.")
    


    load_dotenv()
    
    EMAIL = os.getenv('CROSSREF_EMAIL')
    if not EMAIL:
        logger.warning("CROSSREF_EMAIL not set in .env file. Using default or failing.")
        # Optional: Handle missing email more gracefully or error out
    
    client = CrossrefClient(mailto=EMAIL)
    
    # Determine input file
    if args.file:
        ref_file = Path(args.file).resolve()
    else:
        # Default priority
        default_file = client.base_path / "data" / "samples" / "2312.01797v3_references.txt"
        fallback_file = client.output_dir / "cr_analysis_list.txt"
        
        if default_file.exists():
            ref_file = default_file
        elif fallback_file.exists():
            ref_file = fallback_file
        else:
            # Fallback to hardcoded default even if missing (trigger error log)
            ref_file = default_file

    if ref_file.exists():
        logger.info(f"Reading references from {ref_file}")
        
        # Extract prefix
        # "2312.01797v3_references.txt" -> "2312.01797v3"
        filename = ref_file.name
        if "_references" in filename:
            prefix = filename.split("_references")[0]
        else:
            prefix = ref_file.stem
            
        logger.info(f"Using output prefix: {prefix}")
            
        ref_collection = []    
        with open(ref_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ref_collection.append(line.strip())
        
        # Limit for testing if needed, or run all
        # ref_collection = ref_collection[:5] 
        
        client.process_references(ref_collection, prefix=prefix)
    else:
        logger.error(f"Reference file not found: {ref_file}")