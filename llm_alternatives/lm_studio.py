# NOTE: Remember to start the LLM server before running this script.

import os
# pip install python-dotenv
from dotenv import load_dotenv
from crewai import Agent, Task, Process, Crew
from langchain.llms.openai import OpenAI
from langchain.agents import Tool
from langchain.agents import load_tools
from langchain_community.utilities import SerpAPIWrapper
from reader import SimpleReaderTool

load_dotenv()

llm = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

# To load Google SerpAPI (this api is for free: https://serpapi.com/)
# pip install google-search-results to use SerpAPI
api_serp = os.environ.get("SERPAPI_API_KEY")
os.environ["SERPAPI_API_KEY"] = api_serp

# Initialise SerpAPIWrapper
search = SerpAPIWrapper()

search_tool = Tool(
    name="Scrape google searches",
    func=search.run,
    description="useful for when you need to ask the agent to search the internet",
)

reader = SimpleReaderTool()

reader_tool = Tool(
    name="Read a webpage",
    func=reader.run,
    description="useful for when you need to ask the agent to read a webpage",
)

# Loading Human Tools
human_tools = load_tools(["human"])

"""
- define agents that are going to research latest AI tools and write a blog about it
- explorer will use access to internet to get all the latest news
- writer will write drafts
- critique will provide feedback and make sure that the blog text is engaging and easy to understand
"""
explorer = Agent(
    role="Senior Researcher",
    goal="Find and explore the most exciting projects and companies in the ai and machine learning space in 2024",
    backstory="""
        You are an expert strategist adept at identifying emerging trends and companies in the fields of AI, tech, and machine learning.
        You excel at discovering intriguing and exciting projects, using Google Search to find authoritative websites and browsing the pages for insights.
        You transform the insights into comprehensive reports, highlighting the most promising projects and companies within the AI/ML sphere.
        ONLY use data sourced from online scraping and web browsing.
        """,
    verbose=True,
    allow_delegation=False,
    tools=[search_tool, reader_tool],
    llm=llm

)

writer = Agent(
    role="Senior Technical Writer",
    goal="Write engaging and interesting blog post about latest AI projects using simple, layman vocabulary",
    backstory="""
        You are an expert writer with a specialization in technical innovation, particularly in the realms of AI and machine learning.
        Your writing style is engaging, interesting, yet simple and straightforward, enabling you to convey complex technical concepts to a general audience using layman terms.
        ONLY use data collected from the Senior Researcher in your writing.
        """,
    verbose=True,
    allow_delegation=True,
    llm=llm
)

critic = Agent(
    role="Expert Writing Critic",
    goal="Provide feedback and criticize blog post drafts. Make sure that the tone and writing style is compelling, simple and concise",
    backstory="""
        You are an expert in providing constructive feedback to technical writers.
        With a keen eye for detail, you can identify instances where blog text lacks concision, simplicity, or engagement.
        Your feedback is always helpful and aimed at enhancing the overall quality of the text.
        You ensure that the technical insights remain accessible by using layman terms as needed.
        """,
    verbose=True,
    allow_delegation=True,
    llm=llm
)

task_report = Task(
    description="""
        Create a comprehensive report summarizing the latest emerging projects in the AI sector using solely scraped data from the internet as your source.
        The final output must be a text-only analysis report, devoid of any code or non-textual content.
        Your report should include bullet points detailing no fewer than 5 and no more than 10 innovative new AI projects and tools.
        Be sure to mention the names of each tool or project in every bullet point. For each project or tool, provide three sentences of analysis that focus on a specific company, product, model, or finding derived from your internet research.
        """,
    agent=explorer,
)

task_blog = Task(
    description="""
        Write a blog article featuring a captivating yet succinct headline and comprised of at least 10 paragraphs, all text-only.
        The blog should encapsulate the essence of the report detailing the latest AI tools discovered by the Senior Researcher.
        Maintain a compelling style and tone that is both fun and technical while incorporating layman terms for wider accessibility.
        Highlight specific new, exciting projects, apps, and companies within the AI community.
        Begin each new paragraph on a fresh line rather than using "Paragraph [number]:" labels.
        Bold the names of projects and tools throughout the post.
        Always provide links to projects, tools, or research papers for added context.
        Use ONLY information sourced by the Senior Researcher.

        For your outputs use the following markdown format:
        ```
        ## [Title of post](link to project)
        - Intriguing facts about the project
        - Personal thoughts on how it connects to the overall theme of the newsletter
        ## [Title of second post](link to project)
        - Intriguing facts about the project
        - Personal thoughts on how it connects to the overall theme of the newsletter
        ```
        """,
    agent=writer,
)

task_critique = Task(
    description="""
        The Output MUST be in the following markdown format:
        ```
        ## [Title of post](link to project)
        - Interesting facts
        - Own thoughts on how it connects to the overall theme of the newsletter
        ## [Title of second post](link to project)
        - Interesting facts
        - Own thoughts on how it connects to the overall theme of the newsletter
        ```
        Make sure that it does and if it doesn't, rewrite it accordingly.
        """,
    agent=critic,
)

# instantiate crew of agents
crew = Crew(
    agents=[explorer, writer, critic],
    tasks=[task_report, task_blog, task_critique],
    verbose=2,
    process=Process.sequential,  # Sequential process will have tasks executed one after the other and the outcome of the previous one is passed as extra content into this next.
)

result = crew.kickoff()

print("######################")
print(result)
