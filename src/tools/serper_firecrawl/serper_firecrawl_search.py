# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
Serper + Firecrawl Combined Search Tool

This tool combines:
1. Serper API for Google search to get URLs
2. Firecrawl for scraping each URL's content

Configuration:
    SERPER_API_KEY: API key for Serper search
    FIRECRAWL_API_KEY: API key for Firecrawl scraping

    SERPER_FIRECRAWL_TOP_K: Number of top results to scrape (default: 3)
    FIRECRAWL_TIMEOUT: Scrape timeout in ms (default: 30000)
    FIRECRAWL_ONLY_MAIN_CONTENT: Extract only main content (default: true)
"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import requests
from langchain_core.tools import BaseTool
from pydantic import Field

from src.crawler.firecrawl_client import FirecrawlClient

logger = logging.getLogger(__name__)


class SerperFirecrawlSearch(BaseTool):
    """
    Combined Serper + Firecrawl search tool.

    Uses Serper API to search Google and get top URLs,
    then uses Firecrawl to scrape content from each URL.
    """

    name: str = "web_search"
    description: str = (
        "Search the web using Google (via Serper) and retrieve full content "
        "from top results using Firecrawl. Returns search results with scraped content."
    )

    # Configuration fields
    serper_api_key: str = Field(default="")
    top_k: int = Field(default=3, description="Number of top results to scrape")
    include_images: bool = Field(default=True)
    gl: str = Field(default="us", description="Country code for search")
    hl: str = Field(default="en", description="Language code for search")

    # Firecrawl settings
    firecrawl_timeout: int = Field(default=30000)
    firecrawl_only_main_content: bool = Field(default=True)
    firecrawl_remove_base64_images: bool = Field(default=True)
    firecrawl_block_ads: bool = Field(default=True)
    firecrawl_wait_for: int = Field(default=0)

    # Parallel scraping settings
    max_workers: int = Field(default=3, description="Max parallel scrape workers")

    def __init__(
        self,
        serper_api_key: Optional[str] = None,
        top_k: int = 3,
        include_images: bool = True,
        gl: str = "us",
        hl: str = "en",
        firecrawl_timeout: int = 30000,
        firecrawl_only_main_content: bool = True,
        firecrawl_remove_base64_images: bool = True,
        firecrawl_block_ads: bool = True,
        firecrawl_wait_for: int = 0,
        max_workers: int = 3,
        **kwargs: Any,
    ):
        super().__init__(
            serper_api_key=serper_api_key or os.getenv("SERPER_API_KEY", ""),
            top_k=top_k,
            include_images=include_images,
            gl=gl,
            hl=hl,
            firecrawl_timeout=firecrawl_timeout,
            firecrawl_only_main_content=firecrawl_only_main_content,
            firecrawl_remove_base64_images=firecrawl_remove_base64_images,
            firecrawl_block_ads=firecrawl_block_ads,
            firecrawl_wait_for=firecrawl_wait_for,
            max_workers=max_workers,
            **kwargs,
        )

        if not self.serper_api_key:
            logger.warning("Serper API key is not set. Set SERPER_API_KEY environment variable.")

    def _search_serper(self, query: str) -> Dict[str, Any]:
        """
        Search using Serper API.

        Args:
            query: Search query string

        Returns:
            Serper API response as dictionary
        """
        headers = {
            "X-API-KEY": self.serper_api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "q": query,
            "gl": self.gl,
            "hl": self.hl,
            "num": min(self.top_k + 2, 10),  # Request a few extra in case some fail
        }

        try:
            response = requests.post(
                "https://google.serper.dev/search",
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Serper API request failed: {e}")
            return {"organic": [], "error": str(e)}

    def _scrape_url(self, url: str, title: str) -> Dict[str, Any]:
        """
        Scrape a single URL using Firecrawl.

        Args:
            url: URL to scrape
            title: Title from search result

        Returns:
            Dictionary with url, title, and scraped content
        """
        try:
            client = FirecrawlClient(
                timeout=self.firecrawl_timeout,
                only_main_content=self.firecrawl_only_main_content,
                remove_base64_images=self.firecrawl_remove_base64_images,
                block_ads=self.firecrawl_block_ads,
                wait_for=self.firecrawl_wait_for,
            )

            # Get markdown content (better for LLMs)
            content = client.scrape_to_markdown(url)

            if content.startswith("Error:"):
                logger.warning(f"Failed to scrape {url}: {content}")
                return {
                    "url": url,
                    "title": title,
                    "content": None,
                    "error": content,
                }

            # Truncate content if too long
            max_content_length = 8000
            if len(content) > max_content_length:
                content = content[:max_content_length] + "\n\n[Content truncated...]"

            return {
                "url": url,
                "title": title,
                "content": content,
                "error": None,
            }

        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return {
                "url": url,
                "title": title,
                "content": None,
                "error": str(e),
            }

    def _run(self, query: str) -> str:
        """
        Run the combined Serper + Firecrawl search.

        Args:
            query: Search query string

        Returns:
            JSON string with search results and scraped content
        """
        if not self.serper_api_key:
            return json.dumps({
                "error": "Serper API key is not configured",
                "results": [],
            })

        # Step 1: Search with Serper
        logger.info(f"Searching with Serper: {query}")
        search_results = self._search_serper(query)

        if "error" in search_results and not search_results.get("organic"):
            return json.dumps({
                "error": search_results.get("error", "Search failed"),
                "results": [],
            })

        organic_results = search_results.get("organic", [])
        if not organic_results:
            return json.dumps({
                "query": query,
                "results": [],
                "message": "No search results found",
            })

        # Step 2: Extract top URLs and scrape with Firecrawl
        urls_to_scrape = []
        for result in organic_results[:self.top_k]:
            url = result.get("link", "")
            title = result.get("title", "")
            snippet = result.get("snippet", "")

            if url and not self._is_excluded_url(url):
                urls_to_scrape.append({
                    "url": url,
                    "title": title,
                    "snippet": snippet,
                })

        logger.info(f"Scraping {len(urls_to_scrape)} URLs with Firecrawl")

        # Step 3: Scrape URLs in parallel
        scraped_results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {
                executor.submit(self._scrape_url, item["url"], item["title"]): item
                for item in urls_to_scrape
            }

            for future in as_completed(future_to_url):
                original = future_to_url[future]
                try:
                    result = future.result()
                    result["snippet"] = original.get("snippet", "")
                    scraped_results.append(result)
                except Exception as e:
                    logger.error(f"Scraping failed for {original['url']}: {e}")
                    scraped_results.append({
                        "url": original["url"],
                        "title": original["title"],
                        "snippet": original.get("snippet", ""),
                        "content": None,
                        "error": str(e),
                    })

        # Sort results by original order
        url_order = {item["url"]: i for i, item in enumerate(urls_to_scrape)}
        scraped_results.sort(key=lambda x: url_order.get(x["url"], 999))

        # Build response
        response = {
            "query": query,
            "results": scraped_results,
            "total_results": len(scraped_results),
            "successful_scrapes": sum(1 for r in scraped_results if r.get("content")),
        }

        # Include images if requested
        if self.include_images:
            images = search_results.get("images", [])
            if images:
                response["images"] = [
                    {
                        "title": img.get("title", ""),
                        "imageUrl": img.get("imageUrl", ""),
                        "link": img.get("link", ""),
                    }
                    for img in images[:5]  # Limit to 5 images
                ]

        return json.dumps(response, ensure_ascii=False)

    def _is_excluded_url(self, url: str) -> bool:
        """
        Check if URL should be excluded from scraping.

        Args:
            url: URL to check

        Returns:
            True if URL should be excluded
        """
        excluded_patterns = [
            ".pdf",
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
            ".ppt",
            ".pptx",
            "youtube.com",
            "youtu.be",
            "twitter.com",
            "x.com",
            "facebook.com",
            "instagram.com",
            "tiktok.com",
        ]

        url_lower = url.lower()
        return any(pattern in url_lower for pattern in excluded_patterns)

    async def _arun(self, query: str) -> str:
        """Async version - currently delegates to sync."""
        return self._run(query)
