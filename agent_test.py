from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from dotenv import load_dotenv
from autogen_agentchat.conditions import TextMentionTermination
# from autogen_agentchat.conditions import MaxMessageTermination, TextTerminationCondition
from autogen_agentchat.base import TaskResult
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# model_client = OpenAIChatCompletionClient(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

model_client = OpenAIChatCompletionClient(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "family": "unknown",
        "structured_output": True,
    },
)

# print("Setting up agents...")

async def my_agents(job_position = "AI Engineer"):
    interviewer = AssistantAgent(
        name="Interviewer",
        model_client=model_client,
        description=f"An AI agent that conducts interviews for the position of {job_position}.",
        system_message=f'''
        You are a professional interviewer for a {job_position} position.
        Ask one clear question at a time and wait for the candidate's response before asking the next question.
        Your job is to continue and ask questions, dont pay any attention to career coach response.
        Make sure to ask question based on Candidate's answer and your expertise in the field.
        Ask 3 questions in total covering technical skills and experience, problem-solving abilities, and cultural fit for the company.
        After asking 3 questions, say 'TERMINATE' at the end of the interview.
        Make question under 50 words.
        '''
    )
    
    
    candidate = UserProxyAgent(
        name="Candidate",
        description=f"An AI agent that acts as a candidate applying for the position of {job_position}.",
        input_func = input
    )
    
    evaluation = AssistantAgent(
        name="Evaluation",
        model_client=model_client,
        description=f"An AI agent that evaluates the candidate's performance in the interview for the {job_position} position.",
        system_message=f'''
        You are an evaluator for the {job_position} position.
        Assess the candidate's responses based on their technical skills, problem-solving abilities, and cultural fit.
        Provide a concise evaluation after the interview.
        Make in under 100 words.
        '''
    )

    terminate_condition = TextMentionTermination(text="TERMINATE")  
    team = RoundRobinGroupChat(
        participants=[interviewer, candidate, evaluation], 
        termination_condition=terminate_condition,
        max_turns=10
    )
    
    return team

async def run_interview(team):
    async for message in team.run_stream(task = 'Start the interview with the first question ?'):
        if isinstance(message, TaskResult):
            messsage = f'Interview completed with result: {message.stop_reason}'
            yield message
            
        else:
            messsage = f'{message.source}: {message.content}'
            yield messsage
            
async def main():
    job_position = "AI Engineer"
    team = await my_agents(job_position)
    
    async for message in run_interview(team):
        print('-' * 50)
        print(message)
        

    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())