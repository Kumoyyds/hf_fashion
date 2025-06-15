

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import FirecrawlCrawlWebsiteTool, tool

from crewai_tools import SerperDevTool, FirecrawlSearchTool, ScrapeWebsiteTool
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

# read the photo
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

Image_path = input('enter the path of your original design:\n')

@tool("read the content of original design")
def read_image_content(image_path: str) -> str:
    """
    Reads the content of an image file and returns a description.
    """
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    raw_image = Image.open(image_path).convert('RGB')
    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)

    return processor.decode(out[0], skip_special_tokens=True)


google_search = SerperDevTool(n_results=10, country='fr')
fire_tool = FirecrawlCrawlWebsiteTool(url='firecrawl.dev')

reasoner = llm = LLM(
    model="anthropic/claude-opus-4-20250514"
)





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
                read_image_content,google_search, fire_tool
            ]
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'], # type: ignore[index]
            verbose=True
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
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'], # type: ignore[index]
            output_file='report.md'
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
