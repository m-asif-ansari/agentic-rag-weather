from dotenv import load_dotenv
from datetime import datetime
import requests
import os

load_dotenv()

OPEN_WEATHER_API = os.getenv("OPEN_WEATHER_API")


def fetch_weather_details(city: str) -> dict:
    """Fetch real-time weather data for a particular city from Open-Weather API

    Inputs:
        city (str) :City name for which weather data needs to be fetched

    Returns:
        Data (dict): Weather data for the City 
                    or Error code if there is any error in fetching   
    """

    try:
        params = {
            "q": city,
            "appid": OPEN_WEATHER_API,
            "units": "metric"
        }
        response = requests.get("http://api.openweathermap.org/data/2.5/weather", params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        return {
            "city": data["name"],
            "temperature": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "temp_min": data["main"]["temp_min"],
            "temp_max": data["main"]["temp_max"],
            "humidity": data["main"]["humidity"],
            "description": data["weather"][0]["description"],
            "wind_speed": data["wind"]["speed"],
            "timestamp": datetime.now().isoformat()
        }
        return data
    except requests.RequestException as e:
        return {"error": f"Weather API error: {str(e)}"}


if __name__ == "__main__":
    print("Testing OpenWeather API for Delhi City-")
    print()
    print(fetch_weather_details('delhi'))
    
    