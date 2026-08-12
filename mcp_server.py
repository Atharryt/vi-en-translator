"""

"""

from mcp.server.fastmcp import FastMCP
import voyageai
import os
import glossary

mcp = FastMCP("VI-EN Glossary Server")
voyage_client = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY"))
glossary_vectors = glossary.embed_glossary(voyage_client)

@mcp.tool()
def lookup_glossary(query: str) -> str:
    """Search the VI-EN glossary for terms related to the query text."""
    hits = glossary.retrieve_glossary_hits(voyage_client, glossary_vectors, query)
    if not hits:
        return "no relevant glossary terms found"
    return "; ".join(f"{h['vi']} = {h['en']}" for h in hits)

if __name__ == "__main__":
    mcp.run()