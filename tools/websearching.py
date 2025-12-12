import os
from typing import Optional
from perplexity import Perplexity 


def perpSearch(
    query: str, max_results: int = 3
) -> str:
    """Search the web using Perplexity AI for information not available in local documents.

    Args:
        query: The search query to find web information
        max_results: Maximum number of results to return (default: 5, max: 10)
        search_mode: Search mode - 'web', 'academic', or 'sec' (default: 'web')

    Returns:
        Formatted string of web search results with titles, URLs, and content snippets
    """
    try:
        # Check API key
        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            return "Error: PERPLEXITY_API_KEY environment variable not set. Please configure your Perplexity API key."

        # Initialize client
        client = Perplexity(api_key=api_key)

        # Perform search
        response = client.search.create(
            query=query, max_results=min(max_results, 10)
        )

        # Handle empty results
        if not response.results:
            return f"No web search results found for query: '{query}'"

        # Format results
        formatted_results = []
        for i, result in enumerate(response.results, 1):
            title = getattr(result, "title", "No title available")
            url = getattr(result, "url", "No URL available")
            content = getattr(result, "content", "No content available")

            formatted_results.append(
                f"Web Result {i}:\n"
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Content: {content[:500]}{'...' if len(content) > 500 else ''}\n"
                f"{'-' * 60}"
            )

        return "\n\n".join(formatted_results)

    except Exception as e:
        return f"Error performing web search: {str(e)}. Please check your API key and network connection."

if __name__ == "__main__":
    result = perpSearch("latest AI developments of 2025", 2)
    print(result)
