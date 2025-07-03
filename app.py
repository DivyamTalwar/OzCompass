import streamlit as st
import pandas as pd
import requests
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import plotly.express as px

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

st.set_page_config(
    page_title="Airline Market Demand Analyzer",
    layout="wide",
)

load_dotenv() 
TEQUILA_API_KEY = os.getenv("TEQUILA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

TEQUILA_API_URL = "https://api.tequila.kiwi.com/v2/search"
AUSTRALIAN_CITIES = {
    "Sydney": "SYD",
    "Melbourne": "MEL",
    "Brisbane": "BNE",
    "Perth": "PER",
    "Adelaide": "ADL",
    "Canberra": "CBR",
    "Gold Coast": "OOL",
    "Cairns": "CNS",
}

@st.cache_data(ttl=3600)
def fetch_flight_data(origin_city_code):
    headers = {"apikey": TEQUILA_API_KEY}
    today = datetime.now()
    date_from = (today + timedelta(days=1)).strftime('%d/%m/%Y')
    date_to = (today + timedelta(days=180)).strftime('%d/%m/%Y')

    all_flights = []

    destinations = [code for code in AUSTRALIAN_CITIES.values() if code != origin_city_code]
    
    for dest_code in destinations:
        params = {
            "fly_from": origin_city_code,
            "fly_to": dest_code,
            "date_from": date_from,
            "date_to": date_to,
            "curr": "AUD",
            "limit": 50,
            "sort": "price",
            "one_for_city": 1,
        }
        try:
            response = requests.get(TEQUILA_API_URL, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            if 'data' in data and data['data']:
                for flight in data['data']:
                    all_flights.append({
                        "Origin": flight['cityFrom'],
                        "Origin Code": flight['flyFrom'],
                        "Destination": flight['cityTo'],
                        "Destination Code": flight['flyTo'],
                        "Price (AUD)": flight['price'],
                        "Airline": flight.get('airlines', ['N/A'])[0], # Airline code
                        "Departure Date": datetime.fromtimestamp(flight['dTime']).strftime('%Y-%m-%d'),
                    })
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching data for {origin_city_code} to {dest_code}: {e}")
            return None

    if not all_flights:
        return None

    return pd.DataFrame(all_flights)

def get_ai_insights(df):
    if df is None or df.empty:
        return "No data available to generate insights."

    if not OPENAI_API_KEY:
        return "OpenAI API key is not set. Cannot generate AI insights."

    data_summary = df.to_csv(index=False)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a senior market analyst for a chain of hostels in Australia. Your goal is to provide clear, actionable insights from airline booking data."),
        ("human", """
        Based on the following flight data summary, please provide a concise analysis of market demand trends.
        The data shows the cheapest available one-way flights found for various routes over the next 6 months.

        Focus on these key points:
        1.  **Most Affordable Routes:** Identify the top 3 routes with the lowest average prices.
        2.  **Highest Demand Routes (Proxy):** Identify routes that, despite having many flights searched, still maintain higher average prices, suggesting strong demand.
        3.  **Pricing Trends:** Briefly comment on the general price levels. Are there any standout expensive routes?
        4.  **Actionable Advice:** Provide a short, bullet-pointed recommendation for the hostel marketing team. For example, 'Consider running targeted ads for visitors from [City]' or 'Prepare for higher guest traffic from [City]'.

        Here is the data:
        ---
        {data_summary}
        ---
        Please present your analysis in a clean, easy-to-read format.
        """)
    ])

    try:
        chat_model = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo", temperature=0.7)
        output_parser = StrOutputParser()
        chain = prompt_template | chat_model | output_parser
        
        response = chain.invoke({"data_summary": data_summary})
        return response
    except Exception as e:
        return f"An error occurred while generating AI insights: {e}"


st.title("Airline Booking Market Demand Analyzer")
st.markdown("An internal tool for Australian hostels to analyze flight demand trends.")

st.sidebar.header("Filter Options")
selected_city_name = st.sidebar.selectbox(
    "Select Origin City:",
    options=list(AUSTRALIAN_CITIES.keys()),
)
selected_city_code = AUSTRALIAN_CITIES[selected_city_name]

if st.sidebar.button("Analyze Market Demand"):
    with st.spinner(f"🔍 Fetching and analyzing flight data from {selected_city_name}... This might take a moment."):
        
        flight_df = fetch_flight_data(selected_city_code)

        if flight_df is None or flight_df.empty:
            st.warning(f"Could not retrieve any flight data for routes from {selected_city_name}. This could be due to API limitations or no available flights. Please try again later.")
        else:
            st.header("AI-Powered Insights Summary")
            with st.spinner("Asking our AI analyst for insights..."):
                insights = get_ai_insights(flight_df)
                st.markdown(insights)

            st.header("📊 Visualized Data Insights")

            agg_df = flight_df.groupby('Destination').agg(
                Average_Price=('Price (AUD)', 'mean'),
                Flight_Count=('Price (AUD)', 'count')
            ).reset_index().sort_values('Average_Price')
            
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Average Price by Destination")
                fig_price = px.bar(
                    agg_df,
                    x='Destination',
                    y='Average_Price',
                    title=f"Average Flight Prices from {selected_city_name}",
                    labels={'Average_Price': 'Average Price (AUD)', 'Destination': 'Destination City'},
                    color='Average_Price',
                    color_continuous_scale=px.colors.sequential.Tealgrn,
                )
                fig_price.update_layout(xaxis_title=None, yaxis_title="Avg. Price (AUD)")
                st.plotly_chart(fig_price, use_container_width=True)

            with col2:
                st.subheader("Route Popularity")
                agg_df_sorted_by_count = agg_df.sort_values('Flight_Count', ascending=False)
                fig_count = px.bar(
                    agg_df_sorted_by_count,
                    x='Destination',
                    y='Flight_Count',
                    title=f"Flight Availability from {selected_city_name}",
                    labels={'Flight_Count': '# of Flights Found', 'Destination': 'Destination City'},
                    color='Flight_Count',
                    color_continuous_scale=px.colors.sequential.OrRd
                )
                fig_count.update_layout(xaxis_title=None, yaxis_title="# of Flights")
                st.plotly_chart(fig_count, use_container_width=True)

            st.header("📄 Raw Data Explorer")
            st.dataframe(flight_df)

else:
    st.info("Select an origin city from the sidebar and click 'Analyze Market Demand' to begin.")