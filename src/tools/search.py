# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging
import os
from typing import List, Optional

from langchain_community.tools import (
    BraveSearch,
    DuckDuckGoSearchResults,
    GoogleSerperRun,
    SearxSearchRun,
    WikipediaQueryRun,
)
from langchain_community.tools.arxiv import ArxivQueryRun
from langchain_community.utilities import (
    ArxivAPIWrapper,
    BraveSearchWrapper,
    GoogleSerperAPIWrapper,
    SearxSearchWrapper,
    WikipediaAPIWrapper,
)

from src.config import SELECTED_SEARCH_ENGINE, SearchEngine, load_yaml_config
from src.tools.decorators import create_logged_tool
from src.tools.infoquest_search.infoquest_search_results import InfoQuestSearchResults
from src.tools.tavily_search.tavily_search_results_with_images import (
    TavilySearchWithImages,
)
from src.tools.serper_firecrawl.serper_firecrawl_search import SerperFirecrawlSearch

logger = logging.getLogger(__name__)

# Create logged versions of the search tools
LoggedTavilySearch = create_logged_tool(TavilySearchWithImages)
LoggedInfoQuestSearch = create_logged_tool(InfoQuestSearchResults)
LoggedDuckDuckGoSearch = create_logged_tool(DuckDuckGoSearchResults)
LoggedBraveSearch = create_logged_tool(BraveSearch)
LoggedSerperSearch = create_logged_tool(GoogleSerperRun)
LoggedArxivSearch = create_logged_tool(ArxivQueryRun)
LoggedSearxSearch = create_logged_tool(SearxSearchRun)
LoggedWikipediaSearch = create_logged_tool(WikipediaQueryRun)
LoggedSerperFirecrawlSearch = create_logged_tool(SerperFirecrawlSearch)


def get_search_config():
    config = load_yaml_config("conf.yaml")
    search_config = config.get("SEARCH_ENGINE", {})
    return search_config


# Get the selected search tool
def get_web_search_tool(max_search_results: int):
    search_config = get_search_config()

    if SELECTED_SEARCH_ENGINE == SearchEngine.TAVILY.value:
        # Get all Tavily search parameters from configuration with defaults
        include_domains: Optional[List[str]] = search_config.get("include_domains", [])
        exclude_domains: Optional[List[str]] = search_config.get("exclude_domains", [])
        include_answer: bool = search_config.get("include_answer", False)
        search_depth: str = search_config.get("search_depth", "advanced")
        include_raw_content: bool = search_config.get("include_raw_content", True)
        include_images: bool = search_config.get("include_images", True)
        include_image_descriptions: bool = include_images and search_config.get(
            "include_image_descriptions", True
        )

        logger.info(
            f"Tavily search configuration loaded: include_domains={include_domains}, "
            f"exclude_domains={exclude_domains}, include_answer={include_answer}, "
            f"search_depth={search_depth}, include_raw_content={include_raw_content}, "
            f"include_images={include_images}, include_image_descriptions={include_image_descriptions}"
        )

        return LoggedTavilySearch(
            name="web_search",
            max_results=max_search_results,
            include_answer=include_answer,
            search_depth=search_depth,
            include_raw_content=include_raw_content,
            include_images=include_images,
            include_image_descriptions=include_image_descriptions,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        )
    elif SELECTED_SEARCH_ENGINE == SearchEngine.INFOQUEST.value:
        time_range = search_config.get("time_range", -1)
        site = search_config.get("site", "")
        logger.info(
            f"InfoQuest search configuration loaded: time_range={time_range}, site={site}"
        )
        return LoggedInfoQuestSearch(
            name="web_search",
            time_range=time_range,
            site=site,
        )
    elif SELECTED_SEARCH_ENGINE == SearchEngine.DUCKDUCKGO.value:
        return LoggedDuckDuckGoSearch(
            name="web_search",
            num_results=max_search_results,
        )
    elif SELECTED_SEARCH_ENGINE == SearchEngine.BRAVE_SEARCH.value:
        return LoggedBraveSearch(
            name="web_search",
            search_wrapper=BraveSearchWrapper(
                api_key=os.getenv("BRAVE_SEARCH_API_KEY", ""),
                search_kwargs={"count": max_search_results},
            ),
        )
    elif SELECTED_SEARCH_ENGINE == SearchEngine.SERPER.value:
        return LoggedSerperSearch(
            name="web_search",
            api_wrapper=GoogleSerperAPIWrapper(
                k=max_search_results,
                serper_api_key=os.getenv("SERPER_API_KEY", ""),
            ),
        )
    elif SELECTED_SEARCH_ENGINE == SearchEngine.ARXIV.value:
        return LoggedArxivSearch(
            name="web_search",
            api_wrapper=ArxivAPIWrapper(
                top_k_results=max_search_results,
                load_max_docs=max_search_results,
                load_all_available_meta=True,
            ),
        )
    elif SELECTED_SEARCH_ENGINE == SearchEngine.SEARX.value:
        return LoggedSearxSearch(
            name="web_search",
            wrapper=SearxSearchWrapper(
                k=max_search_results,
            ),
        )
    elif SELECTED_SEARCH_ENGINE == SearchEngine.WIKIPEDIA.value:
        wiki_lang = search_config.get("wikipedia_lang", "en")
        wiki_doc_content_chars_max = search_config.get(
            "wikipedia_doc_content_chars_max", 4000
        )
        return LoggedWikipediaSearch(
            name="web_search",
            api_wrapper=WikipediaAPIWrapper(
                lang=wiki_lang,
                top_k_results=max_search_results,
                load_all_available_meta=True,
                doc_content_chars_max=wiki_doc_content_chars_max,
            ),
        )
    elif SELECTED_SEARCH_ENGINE == SearchEngine.SERPER_FIRECRAWL.value:
        # Serper + Firecrawl combined search
        # Gets search results from Serper, then scrapes each URL with Firecrawl
        top_k = search_config.get("top_k", max_search_results)
        include_images = search_config.get("include_images", True)
        gl = search_config.get("gl", "us")
        hl = search_config.get("hl", "en")
        firecrawl_timeout = search_config.get("firecrawl_timeout", 30000)
        firecrawl_only_main_content = search_config.get("firecrawl_only_main_content", True)
        firecrawl_remove_base64_images = search_config.get("firecrawl_remove_base64_images", True)
        firecrawl_block_ads = search_config.get("firecrawl_block_ads", True)
        firecrawl_wait_for = search_config.get("firecrawl_wait_for", 0)
        max_workers = search_config.get("max_workers", 3)

        logger.info(
            f"Serper+Firecrawl search configuration loaded: top_k={top_k}, "
            f"include_images={include_images}, gl={gl}, hl={hl}, "
            f"firecrawl_timeout={firecrawl_timeout}, firecrawl_only_main_content={firecrawl_only_main_content}"
        )

        return LoggedSerperFirecrawlSearch(
            name="web_search",
            serper_api_key=os.getenv("SERPER_API_KEY", ""),
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
        )
    else:
        raise ValueError(f"Unsupported search engine: {SELECTED_SEARCH_ENGINE}")
