

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import FirecrawlCrawlWebsiteTool
from crewai.tools import tool

import requests
from crewai_tools import SerperDevTool, FirecrawlSearchTool, ScrapeWebsiteTool, DallETool
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

# read the photo
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration



@tool("read the content of original design")
def read_image_content(image_path: str) -> str:
    """
    Reads the content of an image file and returns a description.
    """
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    raw_image = Image.open(requests.get(image_path, stream=True).raw).convert('RGB')
    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)

    return processor.decode(out[0], skip_special_tokens=True)


google_search = SerperDevTool()
fire_tool = FirecrawlCrawlWebsiteTool(url='firecrawl.dev')
img_tool = DallETool()

reasoner = llm = LLM(
    model="anthropic/claude-opus-4-20250514"
)

simple_scrape = ScrapeWebsiteTool(website_url='https://www.pantone.com/eu/fr/color-of-the-year/2025')




@CrewBase
class CrewFashionColor():
    """CrewFashionColor crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'], # type: ignore[index]
            verbose=True,
            tools=[
                google_search, fire_tool
            ],
            llm = 'gpt-4o'
        )

    @agent
    def design_insight_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['design_insight_specialist'], # type: ignore[index]
            verbose=True,
            tools = [google_search],
            llm = reasoner,
            alllow_delegation=True
        ) 
    
    @agent
    def designer(self) -> Agent:
        return Agent(
            config=self.agents_config['designer'], # type: ignore[index]
            verbose=True,
            llm = reasoner,
            tools = [img_tool]
        ) 
    
    @agent
    def design_explainer(self) -> Agent:
        return Agent(
            config=self.agents_config['design_explainer'], # type: ignore[index]
            verbose=True,
            llm = reasoner
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'], # type: ignore[index]
        )
    
    @task   
    def design_decision_making(self) -> Task:
        return Task(
            config=self.tasks_config['design_decision_making'], # type: ignore[index]
            context = [self.research_task()]
        )

    @task
    def design_illustration(self) -> Task:
        return Task(
            config=self.tasks_config['design_illustration'], # type: ignore[index]
            output_file='design_illustration.md',
            context = [self.design_decision_making()]
        )
    
    @task
    def design_explanation(self) -> Task:
        return Task(
            config=self.tasks_config['design_explanation'], # type: ignore[index]
            output_file='report.md',
            context = [self.research_task(), self.design_decision_making()]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the CrewFashionColor crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
