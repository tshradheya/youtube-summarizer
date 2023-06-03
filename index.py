from youtube_transcript_api import YouTubeTranscriptApi
import openai
from langchain.llms import OpenAI
import os
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
import streamlit as st
from urllib.parse import urlparse
from urllib.parse import parse_qs
import requests

youtube_api_key = st.secrets['YOUTUBE_API_KEY']
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.title('Summarize YouTube Video')
youtubeUrl = st.text_input('Put YouTube URL here: ', 'https://www.youtube.com/watch?v=OiRLvUMUDGM&ab_channel=CNBC')
youtube_video_id = parse_qs(urlparse(youtubeUrl).query)['v'][0]
youtubeGetRequstUrl = 'https://www.googleapis.com/youtube/v3/videos?part=snippet&id=' + youtube_video_id + '&key=' + youtube_api_key
title = requests.get(youtubeGetRequstUrl).json()['items'][0]['snippet']['title']

st.text(title)

transcripts = YouTubeTranscriptApi.list_transcripts(youtube_video_id)
languages = [t.language_code for t in transcripts]

result = YouTubeTranscriptApi.get_transcript(youtube_video_id, languages)

text = 'Summarize the following caption of youtube video in sufficient detail with title: ' + title + ' and captions: '

# append 'text' of result variable into one string
for i in result:
    text += i['text'] + ' '


@st.cache_data
def getResult(text):
  llm = OpenAI(temperature=0)

  docs = [Document(page_content=text)]
  chain = load_summarize_chain(llm, chain_type='map_reduce')
  final_res = chain.run(docs)
  return final_res

ans = getResult(text)
st.success(ans)




# final_res = openai.Completion.create(
#   model="text-davinci-003",
#   prompt=embeddings,
#   max_tokens=4000,
#   temperature=0
# )

# print(final_res['choices'][0]['text'])
