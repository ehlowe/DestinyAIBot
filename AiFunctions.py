# Imports
import openai
import tiktoken
import asyncio
import faiss
import numpy as np
import datetime
import math
import time
import json
import os
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from IPython.display import display, HTML

# Transcript Summarization Prompt
transcript_summarization_prompt="Your purpose is to take a transcript from a youtube streamer named Destiny and give a synopsis of the content and the sentiment/takes of the speaker. Include all of the topics even if they are covered briefly instead of just covering the main topic."


# Setup objects
enc=tiktoken.get_encoding("cl100k_base")



# # Basic Functions
# Get time of transcript
def get_time_at_length_transcript(nearest_times, length):
    i=0
    while nearest_times.get(str(int(length-i)),None)==None:
        i+=1
        if (length-i)<=0:
            return list(nearest_times.values())[0]

    return(nearest_times[str(int(length-i))])

# Calculate token costs
def get_cost(input_cost, output_cost, input_text, output_text):
    input_cost+=len(enc.encode(input_text))*(3/1000000.0)
    output_cost+=len(enc.encode(output_text))*(15/1000000.0)
    return(input_cost, output_cost)

# Put hyperlink to time in text
def convert_to_html(nearest_times, video_id, text, start_second, index):
    lines = text.split('\n')
    len_so_far=0
    html_lines = []
    for line in lines:
        line = line.strip()
        len_so_far+=len(line)
        if line:
            time_at_hyperlink=get_time_at_length_transcript(nearest_times, index*1000+len_so_far)-3
            if time_at_hyperlink<0:
                time_at_hyperlink=0
            line+=" |-------------| "+str(index*1000+len_so_far)
            hyperlinked_line = f'<a href="https://www.youtube.com/watch?v={video_id}#t={time_at_hyperlink}s"c">{line}</a>'
            html_lines.append(hyperlinked_line)

    html_text = '<br>'.join(html_lines)
    return html_text












# # Smart Functions
# Load or get transcript
def load_or_get_transcript_info(video_id):
    # Make folder if it doesn't exist
    if os.path.isdir("working_folder")==False:
        os.mkdir("working_folder")
    if os.path.isdir("working_folder/"+video_id)==False:
        os.mkdir("working_folder/"+video_id)

    if os.path.isfile("working_folder/"+video_id+"/transcript_info.json"):
        print("Loading transcript from file")
        with open("working_folder/"+video_id+"/transcript_info.json","r") as f:
            transcript_info=json.load(f)
        
        transcript=transcript_info["transcript"]
        nearest_times=transcript_info["nearest_times"]
    else:
        print("Getting transcript from YouTube")
        raw_transcript=YouTubeTranscriptApi.get_transcript(video_id)
        transcript=""
        nearest_times={}
        for m in raw_transcript:
            #print(m['text'], m["start"])
            transcript+=str(m['text'])+"\n"
            nearest_times[str(len(transcript))]=m["start"]
            # if len(transcript)>50000:
            #     break

        # save transcript
        with open("working_folder/"+video_id+"/transcript_info.json","w") as f:
            json.dump({"transcript":transcript,"nearest_times":nearest_times},f)
    
    return transcript, nearest_times






# Create or load vector db for transcript
async def load_or_get_vectordb_and_chunks(openai_client, transcript, nearest_times, video_id):
    # if files exist, load them
    if os.path.isfile("working_folder/"+video_id+"/vector_db.index"):
        print("Loading vector db from file")
        vector_db = faiss.read_index("working_folder/"+video_id+"/vector_db.index")
        print("Loading text chunks from file")
        with open("working_folder/"+video_id+"/text_chunks_dict.json","r") as f:
            text_chunks_dict=json.load(f)
    else:
        print("Generating text chunks and vector db")
        # make vector db
        async def make_vector_db_fast(openai_client, text_document):
            # Chunk text
            chunk_size=1000
            text_chunks=[text_document[i:i+chunk_size] for i in range(0, len(text_document), chunk_size)]
            print("Number of chunks: ",len(text_chunks))

            # Async function to fetch embeddings
            async def fetch_embeddings_async(text_chunks, model):
                model="text-embedding-3-large"
                async def fetch_embedding(chunk):
                    # Simulate an async call to the embeddings API
                    return await asyncio.to_thread(openai_client.embeddings.create, input=chunk, model=model)

                responses = await asyncio.gather(*(fetch_embedding(chunk) for chunk in text_chunks))
                embeddings = [response.data[0].embedding for response in responses]
                return np.array(embeddings)

            # Generate embeddings
            model="text-embedding-3-large"
            embeddings=await fetch_embeddings_async(text_chunks, model)
            print("Finished generating embeddings")

            # Make vector db
            vector_db=faiss.IndexFlatL2(embeddings.shape[1])
            vector_db.add(np.array(embeddings))

            # return text chunks and vector db
            return(text_chunks, embeddings, vector_db)
        
        # Make text chunks, embeddings, and vector db
        text_chunks, embeddings, vector_db= await make_vector_db_fast(openai_client, transcript)

        # Save json text chunks
        text_chunks_dict={}
        index=0
        for t_chunk in text_chunks:
            text_chunks_dict[str(index)]={"text":t_chunk, "start":get_time_at_length_transcript(nearest_times, transcript.find(t_chunk)), "end":get_time_at_length_transcript(nearest_times,  transcript.find(t_chunk)+len(t_chunk))}
            index+=1
        json.dump(text_chunks_dict,open("working_folder/"+video_id+"/text_chunks_dict.json","w"))

        # Save vector db
        faiss.write_index(vector_db, "working_folder/"+video_id+"/vector_db.index")
    
    return vector_db, text_chunks_dict

# vector search transcript
def search_vector_db(openai_client, vector_db, query, k):
    # Generate query embedding
    query_embedding = openai_client.embeddings.create(input=query,model="text-embedding-3-large").data[0].embedding
    query_embedding_np = np.array(query_embedding).astype('float32').reshape(1, -1)

    D, I = vector_db.search(query_embedding_np, k)
    return (D,I)




















# Make summarized segments of stream transcript
async def load_or_make_summarized_segments(client, transcript, nearest_times, video_id, increment_chars=10000, segments=1):
    output_cost=0
    input_cost=0
    if os.path.exists("working_folder/"+video_id+"/model_responses.json"):
        print("Loading model responses from file")
        model_responses=json.load(open("working_folder/"+video_id+"/model_responses.json","r"))
    else:
        global transcript_summarization_prompt

        # # Supporting Functions
        # Produce summaries for each transcript segment asynchroneously
        async def get_claude_responses(input_data):
            # Def synchronous api call
            def fetch_response(transcript,index):
                conv_messages=[{"role": "user", "content": "Transcript: "+transcript}]
                bot_response=""
                fails=0
                while True:
                    if fails>5:
                        print("Failed to get response for index: ",index)
                        return ["",index]

                    bot_response=""
                    print(str(index)+" ", end="")
                    try:
                        with client.messages.stream(
                                max_tokens=2024,
                                system=transcript_summarization_prompt,
                                messages=conv_messages,
                                model="claude-3-sonnet-20240229",
                            ) as stream:
                                for text in stream.text_stream:
                                    bot_response+=text
                        break
                    except:
                        fails+=1
                        print("Error:",str(index)+" ", end="")
                        time.sleep(10+(fails*2))
                        print("Retrying:",str(index)+" ", end="")

                return [bot_response,index]
            
            # Create thread to run api call
            async def thread_fetch(transcript,index):
                thread = await asyncio.to_thread(fetch_response, transcript,index)
                return(thread)

            # setup data to feed into model
            input_transcripts=[]
            input_indexes=[]
            for data_point in input_data:
                input_transcripts.append(data_point[0])
                input_indexes.append(data_point[1])

            # Gather all the responses
            responses = await asyncio.gather(*(thread_fetch(in_data[0],in_data[1]) for in_data in input_data))
            return(responses)

        # # Start of summary segmentation process
        # Setup Variables
        char_start_index=0
        model_responses=[]
        tasks=[]
        index=0

        # get a certain number of segments
        while (len(model_responses)<segments) and ((char_start_index+increment_chars)<=len(transcript)):
            input_transcript=transcript[char_start_index:char_start_index+increment_chars]
            conv_messages=[{"role": "user", "content": "Transcript: "+input_transcript}]
            bot_response=""

            # display start and endtime
            start_second_raw=get_time_at_length_transcript(nearest_times, char_start_index)
            hours = math.floor(start_second_raw / 3600)
            minutes = math.floor((start_second_raw % 3600) / 60)
            seconds = start_second_raw % 60

            # calculate end time
            end_second_raw=get_time_at_length_transcript(nearest_times, char_start_index+increment_chars)
            hours_end = math.floor(end_second_raw / 3600)
            minutes_end = math.floor((end_second_raw % 3600) / 60)
            seconds_end = end_second_raw % 60

            sf_str=f"Start time {int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}  End time {int(hours_end):02d}:{int(minutes_end):02d}:{seconds_end:06.3f}"
            
            model_responses.append({"bot": "","transcript": input_transcript,"time_string":sf_str,"char_start_finsih_indexes":[char_start_index,char_start_index+increment_chars], "index":index, "start_second":start_second_raw, "end_second":end_second_raw})

            index+=1
            char_start_index+=increment_chars-300



        # get approximate cost of run
        prev_cost=input_cost+output_cost
        temp_input_cost=0
        temp_output_cost=0
        for m in model_responses: 
            seg_costs = get_cost(input_cost, output_cost, m["transcript"],"a b c"*200)
            # unpack seg_costs tuple
            temp_input_cost+=seg_costs[0]
            temp_output_cost+=seg_costs[1]
            
        total_cost=temp_input_cost+temp_output_cost
        print("Approximate cost: ",total_cost-prev_cost, "  Number of segments: ",len(model_responses))

        # Get user decision to proceed
        proceed=input("Proceed with run? (y/n): ")
        if proceed.lower()!="y":
            print("Run cancelled")
            return(None,0,0)
        else:
            # Proceed with run
            bot_responses=await get_claude_responses([[m["transcript"],m["index"]] for m in model_responses])
            for i in range(len(bot_responses)):
                model_responses[i]["bot"]=bot_responses[i][0]
                input_cost, output_cost = get_cost(input_cost, output_cost, model_responses[i]["transcript"], model_responses[i]["bot"])
            total_cost=input_cost+output_cost
            print("Total Cost:", total_cost)

            # save model responses to json
            json.dump(model_responses,open("working_folder/"+video_id+"/model_responses.json","w"))
            print("Model responses saved to file")

    return model_responses, output_cost, input_cost















# Make meta summary from segment summaries
def load_or_make_meta_summary(anthropic_client, model_responses, video_id):
    if os.path.isfile("working_folder/"+video_id+"/meta_summary.txt"):
        print("Loading meta summary from file")
        with open("working_folder/"+video_id+"/meta_summary.txt","r") as f:
            bot_response=f.read()
    else:
        print("Generating meta summary")
        all_summaries=""
        for mr in model_responses:
            all_summaries+=mr["time_string"]+"\n"+mr["bot"]+"\n\n"

        # print expected cost and ask if user wants to proceed
        print("Expected cost: ",len(enc.encode(all_summaries))*(3/1000000.0))
        proceed=input("Proceed with run? (y/n): ")
        if proceed.lower()!="y":
            print("Run cancelled")
            return(None)
        else:
            meta_model_prompt="Your purpose is to take a conglomerate of summaries and compile it into one conglomerate which provides a comprehensive and effective way of knowing what things were talked about in the collection of summaries. The summaries are off of a youtube video transcript of a youtube streamer named Destiny. You should do two parts, main or big topics that were talked about as a main focus or for a long period and another section of smaller details or topic that were covered briefly. These should be two large sections, each may be 300-500 words. In total you need to write around 1000 words. Be sure to include a lot of detail and be comprehensive to get to that 1000 word mark."

            bias_injection="Some information about me the user, I like technology, specifically software but technology generally, I am interested in full democracy, I am probably a bit right leaning and am curious about critiques to conservative views, I am curious about science, and I enjoy humor. "

            if bias_injection!="":
                meta_model_prompt+=" If the user states information about them, cater the summary to their interests."

            # Meta summary
            bot_response=""
            with anthropic_client.messages.stream(
                    max_tokens=4096,
                    system=meta_model_prompt,
                    messages=[{"role":"user", "content": bias_injection+"Collection of summaries for the video/transcript: "+all_summaries}],
                    model="claude-3-sonnet-20240229",
                ) as stream:
                    for text in stream.text_stream:
                        bot_response+=text
                        print(text, end="", flush=True)
            # save bot response as text file
            with open("working_folder/"+video_id+"/meta_summary.txt","w") as f:
                f.write(bot_response)
            print("Meta summary saved to file")

    return bot_response


