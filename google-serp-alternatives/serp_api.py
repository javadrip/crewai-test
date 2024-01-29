import os
# pip install python-dotenv
from dotenv import load_dotenv
from crewai import Agent, Task, Process, Crew
# pip install -U langchain-google-genai
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.llms import Ollama
from langchain.agents import Tool
from langchain.agents import load_tools
from langchain_community.utilities import SerpAPIWrapper

load_dotenv()

# To Load Local models through Ollama
# mistral = Ollama(model="mistral")

# To Load GPT-4
# api = os.environ.get("OPENAI_API_KEY")

# To load gemini (this api is for free: https://makersuite.google.com/app/apikey)
api_gemini = os.environ.get("GEMINI-API-KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-pro", verbose=True, temperature=0.1, google_api_key=api_gemini
)

# To load Google SerpAPI (this api is for free: https://serpapi.com/)
# pip install google-search-results to use SerpAPI
api_serp = os.environ.get("SERPAPI_API_KEY")
os.environ["SERPAPI_API_KEY"] = api_serp

search = SerpAPIWrapper()

search_tool = Tool(
    name="Scrape google searches",
    func=search.run,
    description="useful for when you need to ask the agent to search the internet",
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
    backstory="""You are and Expert strategist that knows how to spot emerging trends and companies in AI, tech and machine learning.
    You're great at finding interesting, exciting projects on LocalLLama subreddit. You turned scraped data into detailed reports with names
    of most exciting projects an companies in the ai/ml world. ONLY use scraped data from the internet for the report.
    """,
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=llm # to load gemini

)

writer = Agent(
    role="Senior Technical Writer",
    goal="Write engaging and interesting blog post about latest AI projects using simple, layman vocabulary",
    backstory="""You are an Expert Writer on technical innovation, especially in the field of AI and machine learning. You know how to write in
    engaging, interesting but simple, straightforward and concise. You know how to present complicated technical terms to general audience in a
    fun way by using layman words.ONLY use scraped data from the internet for the blog.""",
    verbose=True,
    allow_delegation=True,
    llm=llm # to load gemini
)

critic = Agent(
    role="Expert Writing Critic",
    goal="Provide feedback and criticize blog post drafts. Make sure that the tone and writing style is compelling, simple and concise",
    backstory="""You are an Expert at providing feedback to the technical writers. You can tell when a blog text isn't concise,
    simple or engaging enough. You know how to provide helpful feedback that can improve any text. You know how to make sure that text
    stays technical and insightful by using layman terms.
    """,
    verbose=True,
    allow_delegation=True,
    llm=llm # to load gemini
)

task_report = Task(
    description="""Use and summarize scraped data from the internet to make a detailed report on the latest rising projects in AI. Use ONLY
    scraped data to generate the report. Your final answer MUST be a full analysis report, text only, ignore any code or anything that
    isn't text. The report has to have bullet points and with 5-10 exciting new AI projects and tools. Write names of every tool and project.
    Each bullet point MUST contain 3 sentences that refer to one specific ai company, product, model or anything you found on the internet.
    """,
    agent=explorer,
)

task_blog = Task(
    description="""Write a blog article with text only and with a short but impactful headline and at least 10 paragraphs. Blog should summarize
    the report on latest ai tools found on localLLama subreddit. Style and tone should be compelling and concise, fun, technical but also use
    layman words for the general public. Name specific new, exciting projects, apps and companies in AI world. Don't
    write "**Paragraph [number of the paragraph]:**", instead start the new paragraph in a new line. Write names of projects and tools in BOLD.
    ALWAYS include links to projects/tools/research papers. ONLY include information from LocalLLAma.
    For your Outputs use the following markdown format:
    ```
    ## [Title of post](link to project)
    - Interesting facts
    - Own thoughts on how it connects to the overall theme of the newsletter
    ## [Title of second post](link to project)
    - Interesting facts
    - Own thoughts on how it connects to the overall theme of the newsletter
    ```
    """,
    agent=writer,
)

task_critique = Task(
    description="""The Output MUST have the following markdown format:
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

# marketer = Agent(
#     role="Market Research Analyst",
#     goal="Find out how big is the demand for my products and suggest how to reach the widest possible customer base",
#     backstory="""You are an expert at understanding the market demand, target audience, and competition. This is crucial for
# 		validating whether an idea fulfills a market need and has the potential to attract a wide audience. You are good at coming up
# 		with ideas on how to appeal to widest possible audience.
# 		""",
#     verbose=True,  # enable more detailed or extensive output
#     allow_delegation=True,  # enable collaboration between agent
#     llm=llm # to load gemini
# )

# technologist = Agent(
#     role="Technology Expert",
#     goal="Make assessment on how technologically feasable the company is and what type of technologies the company needs to adopt in order to succeed",
#     backstory="""You are a visionary in the realm of technology, with a deep understanding of both current and emerging technological trends. Your
# 		expertise lies not just in knowing the technology but in foreseeing how it can be leveraged to solve real-world problems and drive business innovation.
# 		You have a knack for identifying which technological solutions best fit different business models and needs, ensuring that companies stay ahead of
# 		the curve. Your insights are crucial in aligning technology with business strategies, ensuring that the technological adoption not only enhances
# 		operational efficiency but also provides a competitive edge in the market.""",
#     verbose=True,  # enable more detailed or extensive output
#     allow_delegation=True,  # enable collaboration between agent
#     llm=llm # to load gemini
# )

# business_consultant = Agent(
#     role="Business Development Consultant",
#     goal="Evaluate and advise on the business model, scalability, and potential revenue streams to ensure long-term sustainability and profitability",
#     backstory="""You are a seasoned professional with expertise in shaping business strategies. Your insight is essential for turning innovative ideas
# 		into viable business models. You have a keen understanding of various industries and are adept at identifying and developing potential revenue streams.
# 		Your experience in scalability ensures that a business can grow without compromising its values or operational efficiency. Your advice is not just
# 		about immediate gains but about building a resilient and adaptable business that can thrive in a changing market.""",
#     verbose=True,  # enable more detailed or extensive output
#     allow_delegation=True,  # enable collaboration between agent
#     llm=llm # to load gemini
# )

# task1 = Task(
#     description="""Analyze what the market demand for plugs for holes in crocs (shoes) so that this iconic footware looks less like swiss cheese.
# 		Write a detailed report with description of what the ideal customer might look like, and how to reach the widest possible audience. The report has to
# 		be concise with at least 10 bullet points and it has to address the most important areas when it comes to marketing this type of business.
#     """,
#     agent=marketer,
# )

# task2 = Task(
#     description="""Analyze how to produce plugs for crocs (shoes) so that this iconic footware looks less like swiss cheese.. Write a detailed report
# 		with description of which technologies the business needs to use in order to make High Quality T shirts. The report has to be concise with
# 		at least 10  bullet points and it has to address the most important areas when it comes to manufacturing this type of business.
#     """,
#     agent=technologist,
# )

# task3 = Task(
#     description="""Analyze and summarize marketing and technological report and write a detailed business plan with
# 		description of how to make a sustainable and profitable "plugs for crocs (shoes) so that this iconic footware looks less like swiss cheese" business.
# 		The business plan has to be concise with
# 		at least 10  bullet points, 5 goals and it has to contain a time schedule for which goal should be achieved and when.
#     """,
#     agent=business_consultant,
# )

# crew = Crew(
#     agents=[marketer, technologist, business_consultant],
#     tasks=[task1, task2, task3],
#     verbose=2,
#     process=Process.sequential,  # Sequential process will have tasks executed one after the other and the outcome of the previous one is passed as extra content into this next.
# )

result = crew.kickoff()

print("######################")
print(result)
