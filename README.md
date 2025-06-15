# CrewFashionColor Crew

## Installation

Ensure you have Python >=3.10 <3.13 installed on your system. This project uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling, offering a seamless setup and execution experience.

First, if you haven't already, install uv:

```bash
pip install uv
```

Next, navigate to your project directory and install the dependencies:

(Optional) Lock the dependencies and install them by using the CLI command:
```bash
crewai install
```
## Understanding  Crew

The crew_fashion_color Crew is composed of multiple AI agents, each with unique roles, goals, and tools. These agents collaborate on a series of tasks, defined in `config/tasks.yaml`, leveraging their collective skills to achieve complex objectives. The `config/agents.yaml` file outlines the capabilities and configurations of each agent in your crew.


## agent information:    
**researcher agent**: doing the market research to understand the trend of color which could be suitable to given product.   
**design_insight_specialist**: making the final decision on the colorway of the provided product.    
**designer**: generating the illustration of the new colorway design.  
**design_explainer**: providing the story and philosophy of the new design.  

##  how to run it. 
1. `crewai run`
2. Input your original design and your needs.   

waiting for a little while you will get:  
1. the new design image  
2. the explanation of the new design.


currently, we are still fixing the unstability of image generation.   
