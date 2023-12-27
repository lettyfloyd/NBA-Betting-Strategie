from langchain import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os, requests, datetime
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from lxml import html
import streamlit as st
load_dotenv()

llm = OpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.35,
    verbose=True,
    max_tokens=1500,
)

def get_opponent_team_defensive_stats(team: str) -> str:
    """
        Queries TeamRanking for the team and grabs the defensive stats table. Returns the table as a string bullet list with each item separated by newline.
    """
    # convert team to lower csae and spaces to dashes
    team = team.lower()
    team = team.replace(" ", "-")
    query = f"https://www.teamrankings.com/nba/team/{team}"
    page = requests.get(query)
    soup = BeautifulSoup(page.content, "html.parser")
    lxml_root = html.fromstring(str(soup))
    xpath_query = '//*[@id="html"]/body/div[3]/div[1]/div[3]/aside/table[3]'
    table_defensive_html = lxml_root.xpath(xpath_query)
    # save the html to a file
    with open("team_defensve_stats.html", "w") as f:
        f.write(html.tostring(table_defensive_html[0], pretty_print=True).decode())
    table_html_str = html.tostring(table_defensive_html[0], pretty_print=True).decode()
    # use llm to format the data
    table_html_str_updated = llm(f"format the following html table into bullet boints separated by newline.\nHTML Table: {table_html_str}")
    return table_html_str_updated
# get StatMuse table data for specific query
def search_statmuse(query: str, player_name=None, bid_valu=None, bid_type=None) -> str:
    """
       Goes to StatMuse and gets the table data for a specific query.
       Returns the table data as a HTML string.
    """
    URL = f'https://www.statmuse.com/nba/ask/{query}'
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)
    driver.get(URL)
    
    # Define an explicit wait with a timeout of 120 seconds
    wait = WebDriverWait(driver, 120)
    
    # Wait for the table element to be present in the DOM
    table = wait.until(EC.presence_of_element_located((By.TAG_NAME, 'table')))
    
    # Get the outer HTML of the table
    table_html = table.get_attribute('outerHTML')

    soup = BeautifulSoup(table_html, 'html.parser')

    header = soup.find('thead')
    column_names = []
    if header:
        column_names = [col.text.strip() for col in header.find_all('th')]

    rows = soup.find_all('tr')
    # Prepare the data for CSV
    data = [column_names]
    for row in rows:
        cols = row.find_all('td')
        # Extract the text content from each cell
        cols_text = [col.text.strip() for col in cols]
        data.append(cols_text)
    
    # Close the browser
    driver.quit()
    # return the data as a string
    return str(data)

def get_player_current_team(player_name: str) -> str:
    """
        Queries StatMuse for the player's current team and uses LLM to format it into being the name of the City and the team
        Returns the formatted string
    """
    query = f'Which team does {player_name} play on?'
    statmuse_url = f'https://www.statmuse.com/nba/ask/{query}'
    page = requests.get(statmuse_url)

    soup = BeautifulSoup(page.content, "html.parser")
    player_team_info = soup.find("h1", class_="nlg-answer").text
    # format the table data using LLM
    team_name = llm(f"Extract the NBA team name from the following string and return it in the following format: <Team Location Name> <Team Name>\nTeam Info: {player_team_info}")
    return team_name.replace("\n", "")

def get_player_opponent_team(team: str) -> str:
    """
        Queries StatMuse for the player's opponent team and uses LLM to format it into being the name of the City and the team
        Returns the formatted string
    """
    query = f'Which team does {team} play next?'
    statmuse_url = f'https://www.statmuse.com/nba/ask/{query}'
    page = requests.get(statmuse_url)
    soup = BeautifulSoup(page.content, "html.parser")
    team_info = soup.find("h1", class_="nlg-answer").text
    team_name = llm(f"Extract the other NBA team name out that's not {team} from the following string and return it in the following format: <Team Location Name> <Team Name>.\nTeam Info: {team_info}")
    return team_name.replace("\n", "")

def analyze_player(statmuse_query_1: str, statmuse_query_2: str, statmuse_query_3: str, player_name: str, bid_value: float, opponent_team: str) -> str:
    """
    Searches StatMuse with the given queries, processes the data with LLM,
    and runs the analysis function to determine whether the user should bid up or down
    for the player at the given bid value.
    """

    # =========================================================================================================================== #
    # =========================================================================================================================== #
    # =========================================================================================================================== #
    # =========================================================================================================================== #

    # Get StatMuse table data for specific queries
    table_html_1 = search_statmuse(statmuse_query_1)
    table_html_2 = get_opponent_team_defensive_stats(opponent_team)
    table_html_3 = search_statmuse(statmuse_query_3)

    date = datetime.datetime.now().strftime("%Y-%m-%d")

    # =========================================================================================================================== #
    # =========================================================================================================================== #
    # =========================================================================================================================== #
    # =========================================================================================================================== #

    # create prompt templates
    last_5_game_stats_template = PromptTemplate(
        template="below is a python list that is representing a csv table. csv table\ncsv table: {table_html}\nToday is {date}. Give me the previous 5 game stats before this date in bullet point format with each item separated by newline.",
        input_variables=["table_html", "date"],
    )

    opponent_team_defensive_stats_template  = PromptTemplate(
        template=(
            "Given the following defensive stats for the opposing team, analyze their defensive performance "
            "and determine whether it is likely that a {player_name} with a bid value of {bid_value} points will achieve "
            "that point total in the upcoming game against this team.\n\n"
            "Opposing Team's Defensive Stats:\n"
            "{opponent_team_defensive_stats}\n\n"
            "Player's Bid Value in Points: {bid_value}\n\n"
            "Analysis:\n"
        ),
        input_variables=[
            "opponent_team_defensive_stats",
            "player_name",
            "bid_value"
        ]
    )

    avg_last_5_game_stats_template = PromptTemplate(
        template="Extract out the point values from the last 5 game stats. return as a comma separated values in python with each point value being the value at the index. there should only be 5 values. the last 5 game stats is: \n{last_5_game_stats}\n",
        input_variables=["last_5_game_stats"],
    )

    # =========================================================================================================================== #
    # =========================================================================================================================== #
    # =========================================================================================================================== #
    # =========================================================================================================================== #

    # create LLMChains
    last_5_game_stats_chain = LLMChain(
        prompt=last_5_game_stats_template,
        llm=llm,
    )

    opponent_team_defensive_stats_chain = LLMChain(
        prompt=opponent_team_defensive_stats_template,
        llm=llm
    )

    avg_last_5_game_stats_chain = LLMChain(
        prompt=avg_last_5_game_stats_template,
        llm = llm
    )

    # =========================================================================================================================== #
    # =========================================================================================================================== #
    # =========================================================================================================================== #
    # =========================================================================================================================== #


    # Process table data with LLM
    last_5_game_stats_1 = last_5_game_stats_chain.predict(table_html=table_html_1, date=date)
    opponent_team_defensive_stats_analysis = opponent_team_defensive_stats_chain.predict(
        opponent_team_defensive_stats=table_html_2,
        player_name=player_name,
        bid_value=bid_value
    )
    last_5_game_stats_3 = last_5_game_stats_chain.predict(table_html=table_html_3, date=date)

    avg_last_5_game_stats_1 = avg_last_5_game_stats_chain.run(last_5_game_stats_1)
    avg_last_5_game_stats_3 = avg_last_5_game_stats_chain.run(last_5_game_stats_3)

    # Clean up extracted information
    avg_last_5_game_stats_1 = [float(x.strip()) for x in avg_last_5_game_stats_1.strip().split(",")]
    avg_last_5_game_stats_3 = [float(x.strip()) for x in avg_last_5_game_stats_3.strip().split(",")]

    avg_last_5_game_stats_value_1 = round(sum(avg_last_5_game_stats_1) / len(avg_last_5_game_stats_1), 2)
    avg_last_5_game_stats_value_3 = round(sum(avg_last_5_game_stats_3) / len(avg_last_5_game_stats_3), 2)

    # =========================================================================================================================== #
    # =========================================================================================================================== #
    # =========================================================================================================================== #
    # =========================================================================================================================== #
     
    analysis_template_1 = PromptTemplate(
        template=
            "You are a top of the line nba sports better. you've made millions from betting. Give your best insights on the following proposal:"
            "\n The average points of the last 5 game stats is: {avg_last_5_game_stats_value}"
            "\nStats table: \n{last_5_game_stats}"
            "\nWould u bid up or down for scoring on a bid value of {bid_value} points made for {player_name}? Give a good explanation mentioning avg, the spread of his points over the past 5 games and how they relate to your decision directly, and insight on field goal percentage showing nuance trends about that for your conclusion. Explain this in at least 150 words.",
        input_variables=[
            "avg_last_5_game_stats_value", "last_5_game_stats", "bid_value", "player_name"
        ],
    )
     
    analysis_template_3 = PromptTemplate(
        template=
            "You are a top of the line nba sports better. you've made millions from betting. Give your best insights on the following proposal:"
            "\nThe average points scored by {player_name} against {opponent_team} for their past 5 games is: {avg_last_5_game_stats_value}"
            "\nStats table: \n{last_5_game_stats}"
            "\nWould u bid up or down for scoring on a bid value of {bid_value} points made for {player_name}? Give a good explanation mentioning avg, the spread of his points over the past 5 games against the {opponent_team} and how they relate to your decision directly, and insight on field goal percentage showing nuance trends about that for your conclusion. Explain this in 150 - 200 words.",
        input_variables=[
            "avg_last_5_game_stats_value", "last_5_game_stats", "bid_value", "player_name", "opponent_team"
        ],
    )

    analysis_template_1_chain = LLMChain(
        llm=llm,
        prompt=analysis_template_1
    )

    analysis_template_3_chain = LLMChain(
        llm=llm,
        prompt=analysis_template_3
    )

    # =========================================================================================================================== #
    # =========================================================================================================================== #
    # =========================================================================================================================== #
    # =========================================================================================================================== #

    # Run analysis functions
    analysis_1 = analysis_template_1_chain.predict(
       avg_last_5_game_stats_value = avg_last_5_game_stats_value_1,
       last_5_game_stats = last_5_game_stats_1,
       bid_value= bid_value,
       player_name = player_name
    )

    analysis_3 = analysis_template_3_chain.predict(
        avg_last_5_game_stats_value = avg_last_5_game_stats_value_3,
        last_5_game_stats = last_5_game_stats_3,
        bid_value= bid_value,
        player_name = player_name,
        opponent_team = opponent_team
    )


    final_analysis_template = PromptTemplate(
        template=
            " You are a top of the line nba sports better. you've made millions from betting. Give your best insights on the following proposal for a player on their points. Should We Bid up or down?"
            "\nPlayer: {player_name}"
            "\nBid Value: {bid_value}"
            "\nPlayer Past 5 Game Analysis:"
            "\n{analysis_1}"
            "\nOpponent Team Defensive Analysis:"
            "\n{opponent_team_defensive_stats_analysis}"
            "\nPlayer Past 5 Game Analysis against {opponent_team}:"
            "\n{analysis_3}"
            "\n\nTake into account the player's past 5 game analysis, the opponent team's defensive analysis, and the player's past 5 game analysis against the opponent. Explain to me with the nuances of how all three of these are related to your decision. Make sure to weigh multiple odds and think of deep connections between these analysis to come up with an idea whether to bid up or down. write this in 300 - 400 words. The first line of the response should say either BID UP or BID  DOWN followed by a newline and then the explanation."
            ,
        input_variables=[
            "analysis_1", "opponent_team_defensive_stats_analysis", "analysis_3", "player_name", "bid_value", "opponent_team"
        ],
    )

    final_analysis_chain = LLMChain(
        llm=llm,
        prompt=final_analysis_template
    )

    final_analysis = final_analysis_chain.predict(
        analysis_1 = analysis_1,
        opponent_team_defensive_stats_analysis = opponent_team_defensive_stats_analysis,
        analysis_3 = analysis_3,
        player_name = player_name,
        bid_value = bid_value,
        opponent_team = opponent_team
    )

    return final_analysis

player = "steph curry"
bid_value = "30"
player_team = get_player_current_team(player_name=player)
print(f"Player team: {player_team}")
player_opponent_team = get_player_opponent_team(player_team)
print(f"Player opponent team: {player_opponent_team}")
analysis = analyze_player(f"{player} past 5 game stats", f"{player_opponent_team} past 5 game stats", f"{player} last 5 game stats against {player_opponent_team}", "steph curry", bid_value, player_opponent_team)
# opponent_defensive_stats = get_opponent_team_defensive_stats(player_opponent_team)
print(analysis)


# st.header(":blue[Bidding Bot] :sunglasses:")
# col1, col2 = st.columns(2)
# with col1:
#     player_name = st.text_input("You: ")
# with col2:
#     bid_value = st.number_input("Bid Value: ")
# # initialize memory
# if "memory" not in st.session_state:
#     st.session_state["memory"] = ""

# # streamlit button
# if st.button("Submit"):
#     pass