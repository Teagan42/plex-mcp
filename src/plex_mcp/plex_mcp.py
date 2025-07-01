"""
Module: plex_mcp

This module provides tools for interacting with a Plex server via FastMCP.
It includes functions to search for movies, retrieve movie details, manage playlists,
and obtain recent movies and movie genres. Logging and asynchronous execution are used
to handle non-blocking I/O and to provide informative error messages.
"""
import argparse
import asyncio
from enum import StrEnum
import logging
import os
from dataclasses import dataclass

# --- Import Statements ---
from typing import Annotated, Any, Dict, List, Optional
from starlette.requests import Request
from starlette.responses import Response
from fastmcp import FastMCP
from mcp.types import ToolAnnotations
from plexapi.exceptions import NotFound, Unauthorized
from plexapi.library import MovieSection, ShowSection
from plexapi.client import PlexClient as PlexAPIClient
from plexapi.server import PlexServer
from pydantic import Field

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,  # Use DEBUG for more verbosity during development
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --- FastMCP Server Initialization ---
mcp = FastMCP("plex")

# --- Utility Formatting Functions ---


def format_movie(movie) -> str:
    """
    Format a movie object into a human-readable string.

    Parameters:
        movie: A Plex movie object.

    Returns:
        A formatted string containing movie details.
    """
    title = getattr(movie, "title", "Unknown Title")
    year = getattr(movie, "year", "Unknown Year")
    summary = getattr(movie, "summary", "No summary available")
    duration = (
        getattr(movie, "duration", 0) // 60000 if hasattr(movie, "duration") else 0
    )
    rating = getattr(movie, "rating", "Unrated")
    studio = getattr(movie, "studio", "Unknown Studio")
    directors = [director.tag for director in getattr(movie, "directors", [])[:3]]
    actors = [role.tag for role in getattr(movie, "roles", [])[:5]]

    return (
        f"Title: {title} ({year})\n"
        f"Rating: {rating}\n"
        f"Duration: {duration} minutes\n"
        f"Studio: {studio}\n"
        f"Directors: {', '.join(directors) if directors else 'Unknown'}\n"
        f"Starring: {', '.join(actors) if actors else 'Unknown'}\n"
        f"Summary: {summary}\n"
    )


def format_episode(episode) -> str:
    """
    Format an episode object into a human-readable string.

    Parameters:
        episode: A Plex episode object.

    Returns:
        A formatted string containing episode details.
    """
    show_title = getattr(episode, "grandparentTitle", "Unknown Show")
    season_number = getattr(episode, "parentIndex", "Unknown Season")
    episode_number = getattr(episode, "index", "Unknown Episode")
    title = getattr(episode, "title", "Unknown Title")
    summary = getattr(episode, "summary", "No summary available")
    duration = (
        getattr(episode, "duration", 0) // 60000 if hasattr(episode, "duration") else 0
    )
    rating = getattr(episode, "rating", "Unrated")
    studio = getattr(episode, "studio", "Unknown Studio")
    directors = [director.tag for director in getattr(episode, "directors", [])[:3]]
    actors = [role.tag for role in getattr(episode, "roles", [])[:5]]
    year = getattr(episode, "year", "Unknown Year")

    return (
        f"Show: {show_title}\n"
        f"Season: {season_number}, Episode: {episode_number}\n"
        f"Year: {year}\n"
        f"Title: {title} ({year})\n"
        f"Rating: {rating}\n"
        f"Duration: {duration} minutes\n"
        f"Studio: {studio}\n"
        f"Directors: {', '.join(directors) if directors else 'Unknown'}\n"
        f"Starring: {', '.join(actors) if actors else 'Unknown'}\n"
        f"Summary: {summary}\n"
    )

def format_client(client: PlexAPIClient) -> str:
    """
    Format a Plex client object into a human-readable string.
    Parameters:
        client: A Plex client object.
    Returns:
        A formatted string containing client details.
    """
    return (
        f"Client: {client.title}\n"
        f"Platform: {client.platform}\n"
        f"Product: {client.product}\n"
        f"Version: {client.version}\n"
        f"Device: {client.device}\n"
        f"State: {client.state}\n"
        f"address: {client.address}\n"
        f"User: {client.userID if client.userID else 'Unknown'}\n"
    )


def format_playlist(playlist) -> str:
    """
    Format a playlist into a human-readable string.

    Parameters:
        playlist: A Plex playlist object.

    Returns:
        A formatted string containing playlist details.
    """
    duration_mins = (
        sum(item.duration for item in playlist.items()) // 60000
        if playlist.items()
        else 0
    )
    updated = (
        playlist.updatedAt.strftime("%Y-%m-%d %H:%M:%S")
        if hasattr(playlist, "updatedAt")
        else "Unknown"
    )
    return (
        f"Playlist: {playlist.title}\n"
        f"Items: {len(playlist.items())}\n"
        f"Duration: {duration_mins} minutes\n"
        f"Last Updated: {updated}\n"
    )


def movie_section(server: PlexServer) -> Optional[MovieSection]:
    """Get the first movie section from the Plex server."""
    return next(
        (
            section
            for section in server.library.sections()
            if isinstance(section, MovieSection)
        ),
        None,
    )


def show_section(server: PlexServer) -> Optional[ShowSection]:
    """Get the first show section from the Plex server."""
    return next(
        (
            section
            for section in server.library.sections()
            if isinstance(section, ShowSection)
        ),
        None,
    )


# --- Plex Client Class ---


class PlexClient:
    """
    Encapsulate the Plex connection logic.
    This class handles initialization and caching of the PlexServer instance.
    """

    def __init__(self, server_url: str | None = None, token: str | None = None):
        self.server_url = server_url or os.environ.get("PLEX_SERVER_URL", "").rstrip(
            "/"
        )
        self.token = token or os.environ.get("PLEX_TOKEN")

        if not self.server_url or not self.token:
            raise ValueError(
                "Missing required configuration: Ensure PLEX_SERVER_URL and PLEX_TOKEN are set."
            )

        self._server = None

    def get_server(self) -> PlexServer:
        """
        Return a cached PlexServer instance or initialize one if not already available.

        Returns:
            A connected PlexServer instance.

        Raises:
            Exception: If connection initialization fails.
        """
        if self._server is None:
            try:
                logger.info("Initializing PlexServer with URL: %s", self.server_url)
                self._server = PlexServer(self.server_url, self.token)
                logger.info("Successfully initialized PlexServer.")

                # Validate the connection
                self._server.library.sections()  # Attempt to fetch library sections
                logger.info("Plex server connection validated.")
            except Unauthorized as exc:
                logger.error("Unauthorized: Invalid Plex token provided.")
                raise Exception("Unauthorized: Invalid Plex token provided.") from exc
            except Exception as exc:
                logger.exception("Error initializing Plex server: %s", exc)
                raise Exception(f"Error initializing Plex server: {exc}") from exc
        return self._server


# --- Data Classes ---


@dataclass
class MovieSearchParams:
    title: Optional[str] = None
    year: Optional[int] = None
    director: Optional[str] = None
    studio: Optional[str] = None
    genre: Optional[str] = None
    actor: Optional[str] = None
    rating: Optional[str] = None
    country: Optional[str] = None
    language: Optional[str] = None
    watched: Optional[bool] = None  # True=only watched, False=only unwatched

    def to_filters(self) -> Dict[str, Any]:
        FIELD_MAP = {
            "title": "title",
            "year": "year",
            "director": "director",
            "studio": "studio",
            "genre": "genre",
            "actor": "actor",
            "rating": "contentRating",
            "country": "country",
            "language": "audioLanguage",
            "watched": "unwatched",
        }

        filters: Dict[str, Any] = {"libtype": "movie"}

        for field_name, plex_arg in FIELD_MAP.items():
            value = getattr(self, field_name)
            if value is None:
                continue

            if field_name == "watched":
                # invert for Plex 'unwatched' flag
                filters["unwatched"] = not value
                continue

            filters[plex_arg] = value

        return filters


@dataclass
class ShowSearchParams:
    show_title: Optional[str] = None
    show_year: Optional[int] = None
    director: Optional[str] = None
    studio: Optional[str] = None
    genre: Optional[str] = None
    actor: Optional[str] = None
    rating: Optional[str] = None
    country: Optional[str] = None
    language: Optional[str] = None
    watched: Optional[bool] = None  # True=only watched, False=only unwatched
    season: Optional[int] = None
    episode: Optional[int] = None
    episode_title: Optional[str] = None
    episode_year: Optional[int] = None

    def to_filters(self) -> Dict[str, Any]:
        FIELD_MAP = {
            "show_title": "show.title",
            "show_year": "show.year",
            "director": "show.director",
            "studio": "episode.studio",
            "genre": "episode.genre",
            "actor": "show.actor",
            "rating": "show.contentRating",
            "country": "show.country",
            "language": "episode.audioLanguage",
            "watched": "episode.unwatched",
            "season": "season.index",
            "episode": "episode.index",
            "episode_title": "episode.title",
            "episode_year": "episode.year",
        }

        params: Dict[str, Any] = {"libtype": "episode"}
        filters: Dict[str, Any] = {}
        for field_name, plex_arg in FIELD_MAP.items():
            value = getattr(self, field_name)
            if value is None:
                continue

            if field_name == "watched":
                # invert for Plex 'unwatched' flag
                filters["unwatched"] = not value
                continue

            filters[plex_arg] = value
        params["filters"] = filters
        return params


# --- Global Singleton and Access Functions ---

_plex_client_instance: PlexClient | None = None


def get_plex_client() -> PlexClient:
    """
    Return the singleton PlexClient instance, initializing it if necessary.

    Returns:
        A PlexClient instance.
    """
    global _plex_client_instance
    if _plex_client_instance is None:
        _plex_client_instance = PlexClient()
    return _plex_client_instance


async def get_plex_server() -> PlexServer:
    """
    Asynchronously get a PlexServer instance via the singleton PlexClient.

    Returns:
        A PlexServer instance.

    Raises:
        Exception: When the Plex server connection fails.
    """
    try:
        plex_client = get_plex_client()  # Singleton accessor
        plex = await asyncio.to_thread(plex_client.get_server)
        return plex
    except Exception as e:
        logger.exception("Failed to get Plex server instance")
        raise e


@mcp.custom_route("/health", ["GET"], "health", False)
async def health(request: Request) -> Response:
    try:
        await get_plex_server()
        return Response("OK", status_code=200, media_type="text/plain")
    except Exception:
        return Response("ERROR", status_code=500, media_type="text/plain")


# --- Tool Methods ---


@mcp.tool(
    name="search_movies",
    description="Search for movies in your Plex library using various filters.",
    annotations=ToolAnnotations(
        title="Search For Movies",
    ),
)
async def search_movies(
    title: Annotated[
        Optional[str],
        Field(
            description="Title or substring to match",
            examples=["Inception", "Terminator", "Star Wars"],
            default=None,
        ),
    ] = None,
    year: Annotated[
        Optional[int],
        Field(
            description="Release year to filter by",
            default=None,
            examples=[1994, 2020, 1985],
        ),
    ] = None,
    director: Annotated[
        Optional[str],
        Field(
            description="Director name to filter by",
            default=None,
            examples=["Christopher Nolan", "James Cameron", "George Lucas"],
        ),
    ] = None,
    studio: Annotated[
        Optional[str],
        Field(
            description="Studio name to filter by",
            default=None,
            examples=["Warner Bros.", "20th Century Fox", "Universal Pictures"],
        ),
    ] = None,
    genre: Annotated[
        Optional[str],
        Field(
            description="Genre tag to filter by",
            default=None,
            examples=["Action", "Sci-Fi", "Drama"],
        ),
    ] = None,
    actor: Annotated[
        Optional[str],
        Field(
            description="Actor name to filter by",
            default=None,
            examples=["Leonardo DiCaprio", "Arnold Schwarzenegger", "Harrison Ford"],
        ),
    ] = None,
    rating: Annotated[
        Optional[str],
        Field(
            description="Rating to filter by (e.g., 'PG-13', 'R')",
            default=None,
            examples=["PG-13", "R", "G"],
        ),
    ] = None,
    country: Annotated[
        Optional[str],
        Field(
            description="Country of origin to filter by",
            default=None,
            examples=["USA", "UK", "France"],
        ),
    ] = None,
    language: Annotated[
        Optional[str],
        Field(
            description="Audio or subtitle language to filter by",
            default=None,
            examples=["English", "Spanish", "French"],
        ),
    ] = None,
    watched: Annotated[
        Optional[bool],
        Field(
            description="Filter by watched status; True for watched, False for unwatched",
            default=None,
            examples=[True, False],
        ),
    ] = None,
    limit: Optional[int] = 5,
) -> str:
    """
    Search for movies in your Plex library using optional filters.

    Parameters:
        title: Optional title or substring to match.
        year: Optional release year to filter by.
        director: Optional director name to filter by.
        studio: Optional studio name to filter by.
        genre: Optional genre tag to filter by.
        actor: Optional actor name to filter by.
        rating: Optional rating (e.g., "PG-13") to filter by.
        country: Optional country of origin to filter by.
        language: Optional audio or subtitle language to filter by.
        watched: Optional boolean; True returns only watched movies, False only unwatched.
        min_duration: Optional minimum duration in minutes.
        max_duration: Optional maximum duration in minutes.

    Returns:
        A formatted string of up to 5 matching movies (with a count of any additional results),
        or an error message if the search fails or no movies are found.
    """

    # Validate the limit parameter
    limit = max(1, limit) if limit else 5  # Default to 5 if limit is 0 or negative

    params = MovieSearchParams(
        title,
        year,
        director,
        studio,
        genre,
        actor,
        rating,
        country,
        language,
        watched,
    )
    filters = params.to_filters()
    logger.info("Searching Plex with filters: %r", filters)

    try:
        plex = await get_plex_server()
        library_section = movie_section(plex)
        if not library_section:
            return "ERROR: No movie section found in your Plex library."
        movies = await asyncio.to_thread(library_section.search, **filters)
    except Exception as e:
        logger.exception("search_movies failed connecting to Plex")
        return f"ERROR: Could not search Plex. {e}"

    if not movies:
        return f"No movies found matching filters {filters!r}."

    logger.info("Found %d movies matching filters: %r", len(movies), filters)

    results: List[str] = []
    for i, m in enumerate(movies[:limit], start=1):
        results.append(f"Result #{i}:\nKey: {m.ratingKey}\n{format_movie(m)}")  # type: ignore

    if len(movies) > limit:
        results.append(f"\n... and {len(movies)-limit} more results.")

    return "\n---\n".join(results)


@mcp.tool(
    name="get_movie_details",
    description="Get detailed information about a specific movie by its key.",
    annotations=ToolAnnotations(
        title="Get Movie Details",
    ),
)
async def get_movie_details(
    movie_key: Annotated[
        str,
        Field(
            description="The key identifying the movie to retrieve details for.",
            examples=["12345", "67890"],
        ),
    ],
) -> str:
    """
    Get detailed information about a specific movie.

    Parameters:
        movie_key: The key identifying the movie.

    Returns:
        A formatted string with movie details or an error message.
    """
    try:
        plex = await get_plex_server()
    except Exception as e:
        return f"ERROR: Could not connect to Plex server. {str(e)}"

    try:
        key = int(movie_key)
        library_section = movie_section(plex)
        if not library_section:
            return "ERROR: No movie section found in your Plex library."
        matching_movies = await asyncio.to_thread(library_section.fetchItems, key)  # type: ignore
        movie = next((m for m in matching_movies if m.ratingKey == key), None)  # type: ignore

        if not movie:
            return f"No movie found with key {movie_key}."
        return format_movie(movie)
    except NotFound:
        return f"ERROR: Movie with key {movie_key} not found."
    except Exception as e:
        logger.exception("Failed to fetch movie details for key '%s'", movie_key)
        return f"ERROR: Failed to fetch movie details. {str(e)}"


@mcp.tool(
    name="get_new_movies",
    description="Get a list of recently added movies in your Plex library.",
    annotations=ToolAnnotations(
        title="Get New Movies",
    ),
)
async def get_new_movies() -> str:
    """
    Get list of recently added movies in your Plex library.

    Returns:
        A formatted string with the new movie details or an error message.
    """
    try:
        plex = await get_plex_server()
    except Exception as e:
        return f"ERROR: Could not connect to Plex server. {str(e)}"

    try:
        library_section = movie_section(plex)
        if not library_section:
            return "ERROR: No movie section found in your Plex library."
        movies = await asyncio.to_thread(library_section.recentlyAdded, 10)  # type: ignore

        if not movies:
            return "No new movies found in your Plex library."
        results: List[str] = []
        for i, m in enumerate(movies[:10], start=1):
            results.append(f"Result #{i}:\nKey: {m.ratingKey}\n{format_movie(m)}")  # type: ignore

        return "\n---\n".join(results)
    except Exception as e:
        logger.exception("Failed to fetch new movie list.")
        return f"ERROR: Failed to fetch new movie list. {str(e)}"


@mcp.tool(
    name="search_shows",
    description="Search for shows or episodes in your Plex library using various filters.",
    annotations=ToolAnnotations(
        title="Search For Shows or Episodes",
    ),
)
async def search_shows(
    show_title: Annotated[
        Optional[str],
        Field(
            description="Title or substring of the show to match",
            examples=["Family Guy", "South Park", "3 body problem"],
            default=None,
        ),
    ] = None,
    show_year: Annotated[
        Optional[int],
        Field(
            description="Show release year to filter by",
            default=None,
            examples=[1994, 2020, 1985],
        ),
    ] = None,
    director: Annotated[
        Optional[str],
        Field(
            description="Director name to filter by",
            default=None,
            examples=["Christopher Nolan", "James Cameron", "George Lucas"],
        ),
    ] = None,
    studio: Annotated[
        Optional[str],
        Field(
            description="Studio name to filter by",
            default=None,
            examples=["Warner Bros.", "20th Century Fox", "Universal Pictures"],
        ),
    ] = None,
    genre: Annotated[
        Optional[str],
        Field(
            description="Genre tag to filter by",
            default=None,
            examples=["Action", "Sci-Fi", "Drama"],
        ),
    ] = None,
    actor: Annotated[
        Optional[str],
        Field(
            description="Actor name to filter by",
            default=None,
            examples=["Leonardo DiCaprio", "Arnold Schwarzenegger", "Harrison Ford"],
        ),
    ] = None,
    rating: Annotated[
        Optional[str],
        Field(
            description="Rating to filter by (e.g., 'PG-13', 'R')",
            default=None,
            examples=["PG-13", "R", "G"],
        ),
    ] = None,
    country: Annotated[
        Optional[str],
        Field(
            description="Country of origin to filter by",
            default=None,
            examples=["USA", "UK", "France"],
        ),
    ] = None,
    language: Annotated[
        Optional[str],
        Field(
            description="Audio or subtitle language to filter by",
            default=None,
            examples=["English", "Spanish", "French"],
        ),
    ] = None,
    watched: Annotated[
        Optional[bool],
        Field(
            description="Filter by watched status; True for watched, False for unwatched",
            default=None,
            examples=[True, False],
        ),
    ] = None,
    season: Annotated[
        Optional[int],
        Field(
            description="Season number to filter by",
            default=None,
            examples=[1, 2, 3],
        ),
    ] = None,
    episode: Annotated[
        Optional[int],
        Field(
            description="Episode number to filter by",
            default=None,
            examples=[1, 2, 3],
        ),
    ] = None,
    episode_title: Annotated[
        Optional[str],
        Field(
            description="Episode title or substring to match",
            default=None,
            examples=["Pilot", "The End"],
        ),
    ] = None,
    episode_year: Annotated[
        Optional[int],
        Field(
            description="Episode release year to filter by",
            default=None,
            examples=[2020, 2021, 2022],
        ),
    ] = None,
    limit: Optional[int] = 5,
) -> str:
    """
    Search for movies in your Plex library using optional filters.

    Parameters:
        show_title: Optional show title or substring to match.
        show_year: Optional show release year to filter by.
        director: Optional director name to filter by.
        studio: Optional studio name to filter by.
        genre: Optional genre tag to filter by.
        actor: Optional actor name to filter by.
        rating: Optional rating (e.g., "PG-13") to filter by.
        country: Optional country of origin to filter by.
        language: Optional audio or subtitle language to filter by.
        watched: Optional boolean; True returns only watched movies, False only unwatched.
        season: Optional season number to filter by.
        episode: Optional episode number to filter by.
        episode_title: Optional episode title or substring to match.
        episode_year: Optional episode release year to filter by.


    Returns:
        A formatted string of up to 5 matching shows (with a count of any additional results),
        or an error message if the search fails or no movies are found.
    """

    # Validate the limit parameter
    limit = max(1, limit) if limit else 5  # Default to 5 if limit is 0 or negative

    params = ShowSearchParams(
        show_title,
        show_year,
        director,
        studio,
        genre,
        actor,
        rating,
        country,
        language,
        watched,
        season,
        episode,
        episode_title,
        episode_year,
    )
    filters = params.to_filters()
    logger.info("Searching Plex with filters: %r", filters)

    try:
        plex = await get_plex_server()
        library_section = show_section(plex)
        if not library_section:
            return "ERROR: No show section found in your Plex library."
        episodes = await asyncio.to_thread(library_section.search, **filters)
    except Exception as e:
        logger.exception("search_shows failed connecting to Plex")
        return f"ERROR: Could not search Plex. {e}"

    if not episodes:
        logger.info("No shows found matching filters: %r", filters)
        return f"No shows found matching filters {filters!r}."

    logger.info("Found %d shows matching filters: %r", len(episodes), filters)

    results: List[str] = []
    for i, m in enumerate(episodes[:limit], start=1):
        results.append(f"Result #{i}:\nKey: {m.ratingKey}\n{format_episode(m)}")  # type: ignore

    if len(episodes) > limit:
        results.append(f"\n... and {len(episodes)-limit} more results.")

    return "\n---\n".join(results)


@mcp.tool(
    name="get_episode_details",
    description="Get detailed information about a specific episode identified by its key.",
    annotations=ToolAnnotations(
        title="Get Movie Details",
    ),
)
async def get_episode_details(
    episode_key: Annotated[
        str,
        Field(
            description="The key identifying the episode to retrieve details for.",
            examples=["12345", "67890"],
        ),
    ],
) -> str:
    """
    Get detailed information about a specific episode.

    Parameters:
        episode_key (str): The key identifying the episode.

    Returns:
        A formatted string with episode details or an error message.
    """
    try:
        plex = await get_plex_server()
    except Exception as e:
        return f"ERROR: Could not connect to Plex server. {str(e)}"

    try:
        key = int(episode_key)
        library_section = show_section(plex)
        if not library_section:
            return "ERROR: No show section found in your Plex library."
        matching_episodes = await asyncio.to_thread(library_section.fetchItems, key)  # type: ignore
        episode = next((m for m in matching_episodes if m.ratingKey == key), None)  # type: ignore

        if not episode:
            return f"No episode found with key {episode_key}."
        return format_episode(episode)  # type: ignore
    except NotFound:
        return f"ERROR: Episode with key {episode_key} not found."
    except Exception as e:
        logger.exception("Failed to fetch episode details for key '%s'", episode_key)
        return f"ERROR: Failed to fetch episode details. {str(e)}"


@mcp.tool(
    name="get_new_shows",
    description="Get a list of recently added episodes in your Plex library.",
    annotations=ToolAnnotations(
        title="Get New Shows",
    ),
)
async def get_new_shows() -> str:
    """
    Get list of recently added episodes in your Plex library.

    Returns:
        A formatted string with the new episodes details or an error message.
    """
    try:
        plex = await get_plex_server()
    except Exception as e:
        return f"ERROR: Could not connect to Plex server. {str(e)}"

    try:
        library_section = show_section(plex)
        if not library_section:
            return "ERROR: No show section found in your Plex library."
        episodes = await asyncio.to_thread(library_section.recentlyAdded, 10)  # type: ignore

        if not episodes:
            return "No new episodes found in your Plex library."
        results: List[str] = []
        for i, m in enumerate(episodes[:10], start=1):
            results.append(f"Result #{i}:\nKey: {m.ratingKey}\n{format_episode(m)}")  # type: ignore

        return "\n---\n".join(results)
    except Exception as e:
        logger.exception("Failed to fetch new episode list.")
        return f"ERROR: Failed to fetch new episode list. {str(e)}"


@mcp.tool(
    name="get_active_clients",
    description="Get a list of active clients on your Plex server.",
    annotations=ToolAnnotations(
        title="Get Active Clients",
    ),
)
async def get_active_clients(
    controllable: Annotated[
        Optional[bool],
        Field(
            description="If True, only return clients that can be controlled.",
            default=True,
            examples=[True, False],
        ),
    ] = True,
) -> str:
    """
    Get list active clients on your Plex server.

    Parameters:
        controllable: If True, only return clients that can be controlled.

    Returns:
        A formatted string with the clients details or an error message.
    """
    try:
        plex = await get_plex_server()
    except Exception as e:
        return f"ERROR: Could not connect to Plex server. {str(e)}"

    try:
        clients = await asyncio.to_thread(plex.sessions)

        if not clients:
            return "No active clients connected to your Plex server."
        results: List[str] = []
        for i, m in enumerate(clients, start=1):
            if controllable and "playback" not in m.player.protocolCapabilities: # type: ignore
                continue
            results.append(
                f"Result #{i}:\nMachine Identifier: {m.machineIdentifier}\n{format_client(m.player)}"
            )  # type: ignore

        return "\n---\n".join(results)
    except Exception as e:
        logger.exception("Failed to fetch client list.")
        return f"ERROR: Failed to fetch client list. {str(e)}"


@mcp.tool(
    name="play_media_on_client",
    description="Play specified media on a given Plex client.",
    annotations=ToolAnnotations(
        title="Play Media on Client",
    ),
)
async def play_media_on_client(
    machine_identifier: Annotated[
        str,
        Field(
            description="The machine identifier of the Plex client find this by calling the get_active_clients tool.",
            examples=["abcd1234efgh5678ijkl9012mnop3456qrst7890uvwx"],
        ),
    ],
    media_key: Annotated[
        str,
        Field(
            description="The key of the media item to play.",
            examples=["12345", "67890"],
        ),
    ],
) -> str:
    """
    Play specified media on a given Plex client.
    Parameters:
        machine_identifier: The machine identifier of the Plex client.
        media_key: The key of the media item to play.
    Returns:
        A success message or an error message.
    """
    try:
        plex = await get_plex_server()
    except Exception as e:
        return f"ERROR: Could not connect to Plex server. {str(e)}"

    try:
        clients: list[PlexAPIClient] = await asyncio.to_thread(plex.clients)

        if not clients:
            return "No active clients connected to your Plex server."
        client = next((m for m in clients if m.machineIdentifier == machine_identifier), None)
        if not client:
            return f"No client found with machine identifier {machine_identifier}."
        if "playback" not in client.protocolCapabilities:
            return f"Client {client.title} does not support playback control."

        media = plex.fetchItems(int(media_key))
        if not media:
            return f"No media found with key {media_key}."
        media = media[0]
        await asyncio.to_thread(client.playMedia, media)
        return f"Playing {media.title} on {client.title}." # type: ignore
    except Exception as e:
        logger.exception("Failed to play media on client.")
        return f"ERROR: Failed to play media on client. {str(e)}"


class MediaCommand(StrEnum):
    PLAY = "play"
    PAUSE = "pause"
    STOP = "stop"
    FAST_FORWARD = "fastForward"
    REWIND = "rewind"
    SKIP_NEXT = "skipNext"
    SKIP_PREVIOUS = "skipPrevious"
    SEEK = "seek"


@mcp.tool(
        name="control_client_playback",
        description="Control playback on a specified Plex client (play, pause, stop).",
        annotations=ToolAnnotations(
            title="Control Client Playback",
        ),)
async def control_client_playback(
    machine_identifier: Annotated[
        str,
        Field(
            description="The machine identifier of the Plex client find this by calling the get_active_clients tool.",
            examples=["1234567890abcdef", "abcdef1234567890"],
        ),
    ],
    command: Annotated[
        str,
        Field(
            description="The playback command to send (play, pause, stop).",
            examples=["play", "pause", "stop", "fastForward", "rewind", "skipNext", "skipPrevious", "seek"],
        ),
    ],
    seek_position: Annotated[
        Optional[int],
        Field(
            description="The position in seconds to seek to (required if command is 'seek').",
            examples=[60, 120, 300],
            default=None,
        ),
    ] = None,
) -> str:
    """
    Control playback on a specified Plex client (play, pause, stop).
    Parameters
        machine_identifier: The machine identifier of the Plex client.
        command: The playback command to send (play, pause, stop).
        seek_position: The position in seconds to seek to (required if command is 'seek').
        duration: The duration in seconds to fast forward or rewind (required if command is 'fastForward' or 'rewind').
    Returns:
        A success message or an error message.
    """
    try:
        plex = await get_plex_server()
    except Exception as e:
        return f"ERROR: Could not connect to Plex server. {str(e)}"
    try:
        client: PlexAPIClient = plex.client(machine_identifier)
    except NotFound:
        return f"ERROR: Client with machine identifier '{machine_identifier}' not found."
    except Exception as e:
        return f"ERROR: Could not retrieve client. {str(e)}"

    if "playback" not in client.protocolCapabilities:
        return f"ERROR: Client '{client.title}' does not support playback control."

    try:
        if command == MediaCommand.SEEK and seek_position is not None:
            client.seekTo(seek_position * 1000)
        elif command == MediaCommand.FAST_FORWARD:
            client.stepForward()
        elif command == MediaCommand.REWIND:
            client.stepBack()
        elif command == MediaCommand.SKIP_NEXT:
            client.skipNext()
        elif command == MediaCommand.SKIP_PREVIOUS:
            client.skipPrevious()
        elif command == MediaCommand.STOP:
            client.stop()
        elif command == MediaCommand.PAUSE:
            client.pause()
        elif command == MediaCommand.PLAY:
            client.play()
        else:
            getattr(client, command.value)()
        return f"Command '{command.value}' executed on client '{client.title}'."
    except Exception as e:
        logger.exception("Failed to execute command '%s' on client '%s'.", command.value, machine_identifier)
        return f"ERROR: Failed to execute command '{command.value}' on client '{client.title}'. {str(e)}"


@mcp.tool(
    name="turn_off_client_subtitles",
    description="Turn off Plex client subtitles.",
    annotations=ToolAnnotations(
        title="Turn Off Client Subtitles",
    ),
)
async def turn_off_client_subtitles(
    machine_identifier: Annotated[
        str,
        Field(
            description="The machine identifier of the Plex client find this by calling the get_active_clients tool.",
            examples=["1234567890abcdef", "abcdef1234567890"],
        ),
    ],
) -> str:
    """
    Turns off subtitles for a specified Plex client.
    """
    return await set_client_subtitles(machine_identifier, False)


@mcp.tool(
    name="turn_on_client_subtitles",
    description="Turn on Plex client subtitles.",
    annotations=ToolAnnotations(
        title="Turn On Client Subtitles",
    ),
)
async def turn_on_client_subtitles(
    machine_identifier: Annotated[
        str,
        Field(
            description="The machine identifier of the Plex client find this by calling the get_active_clients tool.",
            examples=["1234567890abcdef", "abcdef1234567890"],
        ),
    ],
) -> str:
    """
    Turns on subtitles for a specified Plex client.
    """
    return await set_client_subtitles(machine_identifier, True)


async def set_client_subtitles(
    machine_identifier: Annotated[
        str,
        Field(
            description="The machine identifier of the Plex client.",
            examples=["1234567890abcdef", "abcdef1234567890"],
        ),
    ],
    subtitles_on: Annotated[
        bool,
        Field(
            description="Whether to turn subtitles on or off.",
            examples=[True, False],
        ),
    ],
) -> str:
    """
    Sets subtitles on or off for a specified Plex client.
    Parameters:
        machine_identifier: The machine identifier of the Plex client.
        subtitles_on: Whether to turn subtitles on or off.
    Returns
        a success message or an error message if the operation fails.
    """
    try:
        plex = await get_plex_server()
    except Exception as e:
        return f"ERROR: Could not connect to Plex server. {str(e)}"
    try:
        client: PlexAPIClient = plex.client(machine_identifier)
        if "playback" not in client.protocolCapabilities:
            return f"ERROR: Client '{client.title}' does not support playback control."
    except NotFound:
        return f"ERROR: Client with machine identifier '{machine_identifier}' not found."
    except Exception as e:
        return f"ERROR: Could not retrieve client. {str(e)}"

    try:
        sessions = await asyncio.to_thread(plex.sessions)
        session = next((s for s in sessions if s.player.machineIdentifier == machine_identifier), None) # type: ignore
        if not session:
            return f"ERROR: No active session found for client '{client.title}'."
        if not session.key:
            return f"ERROR: No session key found for client '{client.title}'."
        if not subtitles_on:
            client.setSubtitleStream(-1)
            return f"Subtitles disabled on client '{client.title}'."
        items = plex.fetchItems(session.key)
        if not items:
            return f"ERROR: No media items found for session on client '{client.title}'."
        for _, item in enumerate(items):
            if not item.media or not item.media[0].parts:
                return f"ERROR: No media found for item  session on client '{client.title}'."
            for _, part in enumerate(item.media[0].parts):
                if not part.subtitleStreams:
                    continue
                for _, subtitle in enumerate(part.subtitleStreams):
                    if subtitle.language.lower() == "english" and not subtitle.forced:
                        client.setSubtitleStream(subtitle.index)
                        return f"Subtitles enabled on client '{client.title}'."
        return f"ERROR: No English subtitles found for current media on client '{client.title}'."
    except NotFound:
        return f"ERROR: Client with machine identifier '{machine_identifier}' not found."
    except Exception as e:
        logger.exception("Failed to set subtitles on client '%s'.", machine_identifier)
        return f"ERROR: Could not set subtitles on client. {str(e)}"


@mcp.tool(
    name="list_playlists",
    description="List all playlists in the Plex server.",
    annotations=ToolAnnotations(
        title="List Playlists",
    ),
)
async def list_playlists() -> str:
    """
    List all playlists in the Plex server.

    Returns:
        A formatted string of playlists or an error message.
    """
    try:
        plex = await get_plex_server()
    except Exception as e:
        return f"ERROR: Could not connect to Plex server. {str(e)}"

    try:
        playlists = await asyncio.to_thread(plex.playlists)
        if not playlists:
            return "No playlists found in your Plex server."
        formatted_playlists = []
        for i, playlist in enumerate(playlists, 1):
            formatted_playlists.append(
                f"Playlist #{i}:\nKey: {playlist.ratingKey}\n{format_playlist(playlist)}"  # type: ignore
            )
        return "\n---\n".join(formatted_playlists)
    except Exception as e:
        logger.exception("Failed to fetch playlists")
        return f"ERROR: Failed to fetch playlists. {str(e)}"


@mcp.tool(
    name="get_playlist_items",
    description="Get items in a specific playlist by its key.",
    annotations=ToolAnnotations(
        title="Get Playlist Items",
    ),
)
async def get_playlist_items(
    playlist_key: Annotated[
        str,
        Field(
            description="The key of the playlist to retrieve items from.",
            examples=["12345", "67890"],
        ),
    ],
) -> str:
    """
    Get the items in a specific playlist.

    Parameters:
        playlist_key: The key of the playlist to retrieve items from.

    Returns:
        A formatted string of playlist items or an error message.
    """
    try:
        plex = await get_plex_server()
    except Exception as e:
        return f"ERROR: Could not connect to Plex server. {str(e)}"

    try:
        key = int(playlist_key)
        all_playlists = await asyncio.to_thread(plex.playlists)
        playlist = next((p for p in all_playlists if p.ratingKey == key), None)  # type: ignore
        if not playlist:
            return f"No playlist found with key {playlist_key}."

        items = playlist.items()
        if not items:
            return "No items found in this playlist."

        formatted_items = []
        for i, item in enumerate(items, 1):
            title = item.title
            year = getattr(item, "year", "")
            type_str = item.type.capitalize()
            formatted_items.append(f"{i}. {title} ({year}) - {type_str}")
        return "\n".join(formatted_items)
    except NotFound:
        return f"ERROR: Playlist with key {playlist_key} not found."
    except Exception as e:
        logger.exception("Failed to fetch items for playlist key '%s'", playlist_key)
        return f"ERROR: Failed to fetch playlist items. {str(e)}"


@mcp.tool(
    name="create_playlist",
    description="Create a new playlist with specified movies.",
    annotations=ToolAnnotations(
        title="Create Playlist",
    ),
)
async def create_playlist(
    name: Annotated[
        str,
        Field(
            description="The name for the new playlist.",
            examples=["My Favorite Movies", "Action Classics", "Sci-Fi Hits"],
            min_length=1,
            max_length=100,
        ),
    ],
    movie_keys: Annotated[
        list[str],
        Field(
            description="List of movie keys to include in the playlist.",
            examples=[["12345", "67890", "112233"]],
        ),
    ],
) -> str:
    """
    Create a new playlist with specified movies.

    Parameters:
        name: The desired name for the new playlist.
        movie_keys: A list of movie keys to include.

    Returns:
        A success message with playlist details or an error message.
    """
    try:
        plex = await get_plex_server()
    except Exception as e:
        return f"ERROR: Could not connect to Plex server. {str(e)}"

    try:
        movie_key_list = [int(key.strip()) for key in movie_keys if key.strip()]
        if not movie_key_list:
            return "ERROR: No valid movie keys provided."

        logger.info("Creating playlist '%s' with movie keys: %s", name, movie_keys)
        all_movies = await asyncio.to_thread(
            lambda: plex.library.search(libtype="movie")
        )
        logger.info("Found %d total movies in library", len(all_movies))
        movie_map = {movie.ratingKey: movie for movie in all_movies}  # type: ignore
        movies = []
        not_found_keys = []

        for key in movie_key_list:
            if key in movie_map:
                movies.append(movie_map[key])
                logger.info("Found movie: %s (Key: %d)", movie_map[key].title, key)  # type: ignore
            else:
                not_found_keys.append(key)
                logger.warning("Could not find movie with key: %d", key)

        if not_found_keys:
            return f"ERROR: Some movie keys were not found: {', '.join(str(k) for k in not_found_keys)}"
        if not movies:
            return "ERROR: No valid movies found with the provided keys."

        try:
            playlist_future = asyncio.create_task(
                asyncio.to_thread(lambda: plex.createPlaylist(name, items=movies))
            )
            playlist = await asyncio.wait_for(playlist_future, timeout=15.0)
            logger.info("Playlist created successfully: %s", playlist.title)
            return f"Successfully created playlist '{name}' with {len(movies)} movie(s).\nPlaylist Key: {playlist.ratingKey}"
        except asyncio.TimeoutError:
            logger.warning(
                "Playlist creation is taking longer than expected for '%s'", name
            )
            return (
                "PENDING: Playlist creation is taking longer than expected. "
                "The operation might still complete in the background. "
                "Please check your Plex server to confirm."
            )
    except ValueError as e:
        logger.error("Invalid input format for movie keys: %s", e)
        return f"ERROR: Invalid input format. Please check movie keys are valid numbers. {str(e)}"
    except Exception as e:
        logger.exception("Error creating playlist")
        return f"ERROR: Failed to create playlist. {str(e)}"


@mcp.tool(
    name="delete_playlist",
    description="Delete a playlist from the Plex server by its key.",
    annotations=ToolAnnotations(
        title="Delete Playlist",
    ),
)
async def delete_playlist(
    playlist_key: Annotated[
        str,
        Field(
            description="The key of the playlist to delete.",
            examples=["12345", "67890"],
        ),
    ],
) -> str:
    """
    Delete a playlist from the Plex server.

    Parameters:
        playlist_key: The key of the playlist to delete.

    Returns:
        A success message if deletion is successful, or an error message.
    """
    try:
        plex = await get_plex_server()
    except Exception as e:
        return f"ERROR: Could not connect to Plex server. {str(e)}"

    try:
        key = int(playlist_key)
        all_playlists = await asyncio.to_thread(plex.playlists)
        playlist = next((p for p in all_playlists if p.ratingKey == key), None)  # type: ignore
        if not playlist:
            return f"No playlist found with key {playlist_key}."
        await asyncio.to_thread(playlist.delete)
        logger.info(
            "Playlist '%s' with key %s successfully deleted.",
            playlist.title,
            playlist_key,
        )
        return (
            f"Successfully deleted playlist '{playlist.title}' with key {playlist_key}."
        )
    except NotFound:
        return f"ERROR: Playlist with key {playlist_key} not found."
    except Exception as e:
        logger.exception("Failed to delete playlist with key '%s'", playlist_key)
        return f"ERROR: Failed to delete playlist. {str(e)}"


@mcp.tool(
    name="add_to_playlist",
    description="Add a movie to an existing playlist by their keys.",
    annotations=ToolAnnotations(
        title="Add Movie to Playlist",
    ),
)
async def add_to_playlist(
    playlist_key: Annotated[
        str,
        Field(
            description="The key of the playlist to add the movie to.",
            examples=["12345", "67890"],
        ),
    ],
    movie_key: Annotated[
        str,
        Field(
            description="The key of the movie to add to the playlist.",
            examples=["54321", "98765"],
        ),
    ],
) -> str:
    """
    Add a movie to an existing playlist.

    Parameters:
        playlist_key: The key of the playlist.
        movie_key: The key of the movie to add.

    Returns:
        A success message if the movie is added, or an error message.
    """
    try:
        plex = await get_plex_server()
    except Exception as e:
        return f"ERROR: Could not connect to Plex server. {str(e)}"

    try:
        p_key = int(playlist_key)
        m_key = int(movie_key)

        # Find the playlist
        all_playlists = await asyncio.to_thread(plex.playlists)
        playlist = next((p for p in all_playlists if p.ratingKey == p_key), None)  # type: ignore
        if not playlist:
            return f"No playlist found with key {playlist_key}."

        # Perform a global search for the movie
        movies = await asyncio.to_thread(
            lambda: plex.library.search(libtype="movie", ratingKey=m_key)
        )
        if not movies:
            return f"No movie found with key {movie_key}."

        movie = movies[
            0
        ]  # Since the search is scoped to the ratingKey, there should be at most one result

        # Add the movie to the playlist
        await asyncio.to_thread(lambda p=playlist, m=movie: p.addItems([m]))
        logger.info("Added movie '%s' to playlist '%s'", movie.title, playlist.title)  # type: ignore
        return f"Successfully added '{movie.title}' to playlist '{playlist.title}'."  # type: ignore
    except ValueError:
        return "ERROR: Invalid playlist or movie key. Please provide valid numbers."
    except Exception as e:
        logger.exception("Failed to add movie to playlist")
        return f"ERROR: Failed to add movie to playlist. {str(e)}"


@mcp.tool(
    name="recent_movies",
    description="Get recently added movies from the Plex library.",
    annotations=ToolAnnotations(
        title="Recent Movies",
    ),
)
async def recent_movies(
    count: Annotated[
        int,
        Field(
            description="The maximum number of recent movies to return.",
            examples=[5, 10, 20],
            ge=1,  # Ensure count is at least 1
            le=100,  # Limit to a maximum of 100 for performance reasons
            default=5,
        ),
    ] = 5,
) -> str:
    """
    Get recently added movies from the Plex library.

    Parameters:
        count: The maximum number of recent movies to return.

    Returns:
        A formatted string of recent movies or an error message.
    """
    if count <= 0:
        return "ERROR: Count must be a positive integer."

    try:
        plex = await get_plex_server()
    except Exception as e:
        return f"ERROR: Could not connect to Plex server. {str(e)}"

    try:
        # Perform a global search for recently added movies
        all_recent = await asyncio.to_thread(
            lambda: plex.library.search(libtype="movie", sort="addedAt:desc")
        )
        recent_movies_list = all_recent[:count]

        if not recent_movies_list:
            return "No recent movies found in your Plex library."

        formatted_movies = []
        for i, movie in enumerate(recent_movies_list, 1):
            formatted_movies.append(
                f"Recent Movie #{i}:\nKey: {movie.ratingKey}\nAdded: {movie.addedAt.strftime('%Y-%m-%d')}\n{format_movie(movie)}"  # type: ignore
            )
        return "\n---\n".join(formatted_movies)
    except Exception as e:
        logger.exception("Failed to fetch recent movies")
        return f"ERROR: Failed to fetch recent movies. {str(e)}"


@mcp.tool(
    name="get_movie_genres",
    description="Get genres for a specific movie by its key.",
    annotations=ToolAnnotations(
        title="Get Movie Genres",
    ),
)
async def get_movie_genres(
    movie_key: Annotated[
        str,
        Field(
            description="The key of the movie to retrieve genres for.",
            examples=["12345", "67890"],
        ),
    ],
) -> str:
    """
    Get genres for a specific movie.

    Parameters:
        movie_key: The key of the movie.

    Returns:
        A formatted string of movie genres or an error message.
    """
    try:
        plex = await get_plex_server()
    except Exception as e:
        return f"ERROR: Could not connect to Plex server. {str(e)}"

    try:
        key = int(movie_key)

        # Perform a global search for the movie
        all_movies = await asyncio.to_thread(
            lambda: plex.library.search(libtype="movie")
        )
        movie = next((m for m in all_movies if m.ratingKey == key), None)  # type: ignore
        if not movie:
            return f"No movie found with key {movie_key}."

        # Extract genres
        genres = (
            [genre.tag for genre in movie.genres] if hasattr(movie, "genres") else []
        )
        if not genres:
            return f"No genres found for movie '{movie.title}'."
        return f"Genres for '{movie.title}':\n{', '.join(genres)}"
    except ValueError:
        return f"ERROR: Invalid movie key '{movie_key}'. Please provide a valid number."
    except Exception as e:
        logger.exception("Failed to fetch genres for movie with key '%s'", movie_key)
        return f"ERROR: Failed to fetch movie genres. {str(e)}"


def parse_args():
    parser = argparse.ArgumentParser(description="Start the Plex FastMCP server.")

    parser.add_argument(
        "-u", "--plex-url",
        type=str,
        required=True,
        help="Base URL for the Plex server (e.g., http://localhost:32400)",
    )
    parser.add_argument(
        "-k", "--plex-token",
        type=str,
        required=True,
        help="Authentication token for accessing the Plex server",
    )
    parser.add_argument(
        "-t", "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport mechanism to use for FastMCP",
    )
    parser.add_argument(
        "-b", "--bind",
        type=str,
        default=None,
        help="IP address to bind the server to (required if transport is not stdio)",
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=None,
        help="Port number to bind the server to (required if transport is not stdio)",
    )
    parser.add_argument(
        "-m", "--mount",
        type=str,
        default=None,
        help="Path to mount the server on (only for streamable-http transport)",
    )
    parser.add_argument(
        "-l", "--log-level",
        type=str,
        default="info",
        help="Logging level for the server (e.g., debug, info, warning, error, critical)",
    )

    args = parser.parse_args()

    if args.transport != "stdio":
        if not args.bind:
            parser.error("--bind is required when transport is not 'stdio'")
        if not args.port:
            parser.error("--port is required when transport is not 'stdio'")

    return args

def run():
    args = parse_args()
    os.environ["PLEX_SERVER_URL"] = args.plex_url
    os.environ["PLEX_TOKEN"] = args.plex_token
    PlexClient(args.plex_url, args.plex_token)  # Initialize singleton
    asyncio.run(get_plex_server())
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "sse":
        mcp.run(
            transport="sse",
            host=args.bind,
            port=args.port,
            path=args.mount,
            log_level=args.log_level,
        )
    elif args.transport == "streamable-http":
        mcp.run(
            transport="http",
            host=args.bind,
            port=args.port,
            path=args.mount,
            log_level=args.log_level,
        )
    else:
        raise ValueError(f"Unsupported transport: {args.transport}")

# --- Main Execution ---
if __name__ == "__main__":
    run()
