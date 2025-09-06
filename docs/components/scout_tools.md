# Scout Tools Documentation

## Arxiv Search Tool

The `arxiv_search` tool allows you to search for academic papers on Arxiv.org. It's a free tool that doesn't require API keys.

### Usage
```python
from aclarai_claimify.scout.tools import arxiv_search

result = arxiv_search("machine learning")
```

## Wikipedia Search Tool

The `wikipedia_search` tool allows you to search for information on Wikipedia. It's a free tool that doesn't require API keys.

### Usage
```python
from aclarai_claimify.scout.tools import wikipedia_search

result = wikipedia_search("artificial intelligence")
```

## Documentation Crawler Tool

The `documentation_crawler` tool performs deep crawling of documentation sites using Playwright for JavaScript rendering.

### Best Practices

1. **Examine the starting page first**: Before running the crawler, use the `url_to_markdown` tool to download and examine the starting page. This will help you understand the site structure and configure appropriate paths.

2. **Configure paths carefully**: Based on your examination of the starting page, set appropriate `allowed_paths` and `ignore_paths` parameters to focus the crawl on relevant content and avoid irrelevant sections.

3. **Start shallow**: Begin with a shallow `max_depth` (1-2) and increase gradually to avoid crawling too much content.

### Usage
```python
from aclarai_claimify.scout.tools import documentation_crawler

result = documentation_crawler(
    base_url="https://example.com",
    starting_point="/docs/",
    max_depth=2,
    allowed_paths=["/docs/guides", "/docs/api"],
    ignore_paths=["/docs/admin", "/docs/private"]
)
```

## Response Truncation

All search tools automatically truncate their responses based on the max_tokens setting for the calling agent role. This prevents overwhelming the model with too much information. When a response is truncated, a note is added indicating the truncation.

Both tools are rate-limited to prevent abuse and ensure fair usage.