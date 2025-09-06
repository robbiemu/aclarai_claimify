# Arxiv and Wikipedia Tools

This document describes the Arxiv and Wikipedia tools available in the Data Scout Agent.

## Search vs Content Tools

We have separated the functionality into two distinct types of tools:

1. **Search Tools** - These tools perform searches and return brief summaries of results
2. **Content Tools** - These tools retrieve detailed content from documents

## Arxiv Tools

### `arxiv_search`
Performs a search on Arxiv and returns brief summaries of papers.

**Input**: A search query string
**Output**: A string containing summaries of the top search results

### `arxiv_get_content`
Retrieves detailed content from Arxiv papers based on a search query.

**Input**: A search query string
**Output**: A list of detailed documents with full content
**Note**: Requires PyMuPDF package for PDF processing

## Wikipedia Tools

### `wikipedia_search`
Performs a search on Wikipedia and returns brief summaries of articles.

**Input**: A search query string
**Output**: A string containing summaries of the top search results

### `wikipedia_get_content`
Retrieves detailed content from Wikipedia articles based on a search query.

**Input**: A search query string
**Output**: A list of detailed documents with full content

## Usage in ReAct Loop

The new tools are specifically designed for use within the ReAct loop of the scout agent. The search tools are useful for initial exploration and information gathering, while the content tools are useful when detailed analysis of specific documents is needed.

For the scout agent, both search and content tools are available in the "research" role.