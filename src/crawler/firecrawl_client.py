# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class FirecrawlClient:
    """
    Firecrawl client for web scraping.

    Firecrawl converts web pages to clean markdown/HTML content optimized for LLMs.
    This client only performs scraping (single page), not crawling (following links).

    Environment variables:
        FIRECRAWL_API_KEY: API key for Firecrawl service

    Scrape parameters:
        - formats: Output formats (markdown, html, links, screenshot, etc.)
        - onlyMainContent: Extract only main content, excluding headers/footers/navigation
        - includeTags: Only include specific HTML tags
        - excludeTags: Exclude specific HTML tags
        - waitFor: Wait for specific element or time before scraping (ms)
        - timeout: Request timeout in milliseconds
        - mobile: Use mobile user agent
        - skipTlsVerification: Skip TLS certificate verification
        - removeBase64Images: Remove base64 encoded images from output
        - blockAds: Block ads on the page
    """

    DEFAULT_API_URL = "https://api.firecrawl.dev/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        timeout: int = 30000,
        only_main_content: bool = True,
        remove_base64_images: bool = True,
        block_ads: bool = True,
        wait_for: int = 0,
    ):
        """
        Initialize Firecrawl client.

        Args:
            api_key: Firecrawl API key (defaults to FIRECRAWL_API_KEY env var)
            api_url: Firecrawl API URL (defaults to https://api.firecrawl.dev/v1)
            timeout: Request timeout in milliseconds (default: 30000)
            only_main_content: Extract only main content (default: True)
            remove_base64_images: Remove base64 images from output (default: True)
            block_ads: Block ads on the page (default: True)
            wait_for: Wait time in ms before scraping (default: 0)
        """
        self.api_key = api_key or os.getenv("FIRECRAWL_API_KEY", "")
        self.api_url = api_url or os.getenv("FIRECRAWL_API_URL", self.DEFAULT_API_URL)
        self.timeout = timeout
        self.only_main_content = only_main_content
        self.remove_base64_images = remove_base64_images
        self.block_ads = block_ads
        self.wait_for = wait_for

        if not self.api_key:
            logger.warning(
                "Firecrawl API key is not set. Set FIRECRAWL_API_KEY environment variable."
            )

    def crawl(self, url: str, return_format: str = "html") -> str:
        """
        Scrape a single URL and return content.

        Note: This method is named 'crawl' for interface compatibility with other
        crawler clients (JinaClient, InfoQuestClient), but it only performs
        single-page scraping, not multi-page crawling.

        Args:
            url: The URL to scrape
            return_format: Output format - "html" or "markdown" (default: "html")

        Returns:
            Scraped content in the requested format
        """
        if not self.api_key:
            error_message = "Firecrawl API key is not configured"
            logger.error(error_message)
            return f"Error: {error_message}"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # Map return_format to Firecrawl formats
        formats = ["markdown"] if return_format == "markdown" else ["html", "markdown"]

        payload = {
            "url": url,
            "formats": formats,
            "onlyMainContent": self.only_main_content,
            "removeBase64Images": self.remove_base64_images,
            "blockAds": self.block_ads,
            "timeout": self.timeout,
        }

        # Add waitFor if specified
        if self.wait_for > 0:
            payload["waitFor"] = self.wait_for

        try:
            response = requests.post(
                f"{self.api_url}/scrape",
                headers=headers,
                json=payload,
                timeout=self.timeout / 1000 + 10,  # Convert to seconds + buffer
            )

            if response.status_code != 200:
                error_message = f"Firecrawl API returned status {response.status_code}: {response.text}"
                logger.error(error_message)
                return f"Error: {error_message}"

            result = response.json()

            if not result.get("success", False):
                error_message = f"Firecrawl scrape failed: {result.get('error', 'Unknown error')}"
                logger.error(error_message)
                return f"Error: {error_message}"

            data = result.get("data", {})

            # Return requested format
            if return_format == "markdown":
                content = data.get("markdown", "")
            else:
                # Prefer HTML, fall back to markdown
                content = data.get("html", "") or data.get("markdown", "")

            if not content or not content.strip():
                error_message = "Firecrawl returned empty content"
                logger.warning(error_message)
                return f"Error: {error_message}"

            return content

        except requests.Timeout:
            error_message = f"Firecrawl request timed out for URL: {url}"
            logger.error(error_message)
            return f"Error: {error_message}"
        except requests.RequestException as e:
            error_message = f"Firecrawl request failed: {str(e)}"
            logger.error(error_message)
            return f"Error: {error_message}"
        except Exception as e:
            error_message = f"Unexpected error in Firecrawl client: {str(e)}"
            logger.error(error_message)
            return f"Error: {error_message}"

    def scrape_to_markdown(self, url: str) -> str:
        """
        Convenience method to scrape URL directly to markdown.

        Args:
            url: The URL to scrape

        Returns:
            Scraped content in markdown format
        """
        return self.crawl(url, return_format="markdown")
