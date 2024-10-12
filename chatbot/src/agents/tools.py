from datetime import datetime, timedelta
from os import getenv

from bs4 import BeautifulSoup

from googlemaps import Client

from langchain_community.tools import TavilySearchResults
from langchain_core.tools import BaseTool, tool

from utils.config import TIMEZONE


GOOGLE_MAPS_API_KEY = getenv("GOOGLE_MAPS_API_KEY", "")


def get_parsed_date(date: str, fmt: str = "%Y-%m-%d") -> datetime | str:
    if date in ("now", "today"):
        return datetime.now(TIMEZONE)
    else:
        try:
            return datetime.strptime(date, fmt)
        except Exception:
            return "invalid date format, please use format: YYYY-mm-dd"


@tool("compare-three-dates-tool")
def check_betweenness(
    first_date: str, second_date: str, third_date: str
) -> bool:
    """
    Check if first_date is between second_date and third_date
    """
    first, second, third = map(
        get_parsed_date, (first_date, second_date, third_date)
    )
    return second <= first <= third


@tool("compare-dates-tool")
def compare_dates(first_date: str, second_date: str) -> bool:
    """
    Check if first_date is before second_date.
    """
    first, second = map(get_parsed_date, (first_date, second_date))
    return first < second


@tool("datetime-tool")
def get_date(date: str) -> datetime:
    """
    Converts the given date string into a calendar date in YYYY-mm-dd format
    """
    return get_parsed_date(date)


@tool("weekday-name-tool")
def get_day_of_week(date: str) -> str:
    """
    Returns the day of week of a given date string
    """
    return get_parsed_date(date).strftime('%A')


@tool("day-difference-tool")
def get_delta_days_from_date(date: str, delta: int) -> str:
    """
    Returns the datetime that is delta days away from the given date
    """
    today = datetime.now(TIMEZONE)
    if date in ('today', 'now'):
        date1 = today
    elif date == 'yesterday':
        date1 = today - timedelta(days=1)
    elif date == 'tomorrow':
        date1 = today + timedelta(days=1)
    else:
        date1 = get_parsed_date(date)
    new_date = date1 + timedelta(days=delta)
    return new_date.strftime("%Y-%m-%d")


@tool("directions-tool")
def get_directions(
    start: str,
    end: str,
    transit_type: str = 'walking',
    start_time: datetime | None = None,
    waypoints: list | None = None,
) -> str:
    """
    Returns walking directions between a starting location and an \
    ending location.
    The starting position should be the business location which can \
    be found using the search-document tool.
    The ending location should come from the human input.
    """
    if start_time is None:
        start_time = datetime.now(TIMEZONE)
    gmap = Client(key=GOOGLE_MAPS_API_KEY)
    directions = gmap.directions(
                start,
                end,
                waypoints=waypoints,
                mode=transit_type,
                units="metric",
                optimize_waypoints=True,
                traffic_model="best_guess",
                departure_time=start_time,
    )
    direction_steps = ', '.join([
        step['html_instructions'] for step in directions[0]['legs'][0]['steps']
    ])
    soup = BeautifulSoup(direction_steps)
    return soup.text


tavily_tool: list[BaseTool] = [
    TavilySearchResults(
        handle_tool_error=True,
        max_results=5,
        include_answer=True,
        include_raw_content=True,
        include_images=True,
        search_depth="advanced",
        include_domains=[],
        exclude_domains=[],
    )
]


datetime_tools: list[BaseTool] = [
    compare_dates,
    get_date,
    get_day_of_week,
    check_betweenness,
    get_delta_days_from_date,
]


location_tools: list[BaseTool] = [
    get_directions,
]
