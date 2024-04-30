# Summarizer_map.py
# Libraries
import os
from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.llm import LLMChain
from langfuse.callback import CallbackHandler
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Modules
from map_reduce_prompts_minimal import map_prompt_template, combine_prompt_template, map_prompt_template2, combine_prompt_template2, bullet_prompt, all_in_one_prompt
from tokenizer import encoding_gpt3, encoding_gpt4
from splitter import r_splitter

# Initialization variables set to None
llm = None
llm_final = None
llm_all_in_one = None
summary_chain1 = None
summary_chain2 = None
bullet_chain = None
all_in_one_chain = None
handler = None

llms = {}

def initialize_summarizer(llm_type="openai"):
    global llms, llm, llm_final, summary_chain1, summary_chain2, bullet_chain, handler, all_in_one_chain
    
    load_dotenv()
    
    PUBLIC_KEY = os.getenv('PUBLIC_KEY')
    SECRET_KEY = os.getenv('SECRET_KEY')
    
    handler = CallbackHandler(PUBLIC_KEY, SECRET_KEY)
    
    # get model name from environment variable:
    if llm_type == "openai":
        all_in_one_model_name = os.getenv('OPENAI_GPT_ALL_IN_ONE')
        final_model_name = os.getenv('OPENAI_GPT_FINAL')
        mapreduce_model_name = os.getenv('OPENAI_GPT_MAPREDUCE')
        llms = {
            "mapreduce": ChatOpenAI(temperature=0.7, model_name = mapreduce_model_name),
            "final": ChatOpenAI(temperature=0.7, model_name = final_model_name),
            "all_in_one": ChatOpenAI(temperature=0.7, model_name = all_in_one_model_name)
        }
    elif llm_type == "anthropic":
        all_in_one_model_name = os.getenv('ANTHROPIC_GPT_ALL_IN_ONE')
        final_model_name = os.getenv('ANTHROPIC_GPT_FINAL')
        mapreduce_model_name = os.getenv('ANTHROPIC_GPT_MAPREDUCE')
        llms = {
            "mapreduce": ChatAnthropic(temperature=0.7, model_name = mapreduce_model_name),
            "final": ChatAnthropic(temperature=0.7, model_name = final_model_name),
            "all_in_one": ChatAnthropic(temperature=0.7, model_name = all_in_one_model_name)
        }
    
    print("Initializing summarizer...")
    print("all in one prompt model: ", all_in_one_model_name)
    print("model for final prompt:", final_model_name)
    print("map-reduce model:", mapreduce_model_name)
    
    llm = llms["mapreduce"]
    llm_final = llms["final"]
    llm_all_in_one = llms["all_in_one"]

    summary_chain1 = load_summarize_chain (
        llm=llms["mapreduce"],
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=combine_prompt_template,
        verbose=False,
        token_max=4000
    )

    summary_chain2 = load_summarize_chain (
    llm=llm,
    chain_type='map_reduce',
    map_prompt=map_prompt_template2,
    combine_prompt=combine_prompt_template2,
    verbose=False,
    token_max=4000
    )
    bullet_chain = LLMChain(llm=llm_final, prompt=bullet_prompt, output_key="bullet-summary")
    all_in_one_chain = LLMChain(llm=llm_all_in_one, prompt=all_in_one_prompt, output_key="all-in-one-summary")

def generate_summary_map(docs, token_count_transcript, llm_type="openai"):
    global llm, llm_final, summary_chain1, summary_chain2, bullet_chain, handler, all_in_one_chain

    # if llm is None:
    initialize_summarizer(llm_type=llm_type)
    print("Summarizer initialized.")

    bullet_summary = None

    if token_count_transcript < 4000:
        print("Token count is less than 4000. Running single-prompt summary...")
        result = all_in_one_chain(docs)
        final_summary = result['all-in-one-summary']
    else:
        # Run first summary chain
        print("Running map reduce chain...")
        # first_summary = summary_chain1.run(docs, callbacks=[handler])
        first_summary = summary_chain1.run(docs)
        token_count_first_summary = len(encoding_gpt4.encode(first_summary))
        print(f"First summary chain complete. Used {token_count_first_summary} tokens.")
       
        # Check if second chain needs to be run
        if token_count_first_summary > 4000:
            print("Running the second summary chain due to token limit breach...")
            first_summary_docs = r_splitter.create_documents([first_summary])
            second_summary = summary_chain2.run(first_summary_docs)
            
            final_summary = second_summary
        else:
            final_summary = first_summary
        print("Generating final summary...")
        bullet_summary = bullet_chain.run(summary = final_summary,callbacks=[handler])
        print("Bullet summary generated.")

    return final_summary, bullet_summary