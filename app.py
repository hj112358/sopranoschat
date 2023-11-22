# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 23:30:19 2023

@author: Howard
"""

# import requests
# from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import openai
import os
from langchain.chat_models import ChatOpenAI
# from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
# import numpy as np
# import faiss
import streamlit as st


# # # also extract episode title, so later they can be easily referenced
# # # instead of a big block of text?
# # # extract info from textbox at top of each episode page - title, episode and season number?
# # # extract: series name, episode name, episode and season # along with synopsis


# def get_episode_links():
#     # # # input the url of the "List of [show name] Episodes" wiki page, e.g.,
#     # # # https://en.wikipedia.org/wiki/List_of_The_Sopranos_episodes
#     # # # for later: check that URL is fully formed, it's a wikipedia link, etc.
#     url = input("Paste the complete URL of the 'List of [show name] Episodes' Wikipedia page here: ")
    
#     response = requests.get(url)
    
#     if response.status_code == 200:
#         episodeList_soup = BeautifulSoup(response.text, "html.parser")
#         # # # extract links to episode summaries
        
# # =============================================================================
# # get the series name right at the top when retrieving episode links.
# # when openAI is not overloaded anymore, go after this kind of element:       
# # <h1 id="firstHeading" class="firstHeading mw-first-heading">List of <i>The Sopranos</i> episodes</h1>
# # List of <i>{series}<i> episodes is what we need to find
# # =============================================================================
#         summary_tds = episodeList_soup.find_all("td", class_ = "summary")
#         episode_links = [td.a["href"] if td.a else None for td in summary_tds if td]
#         episode_links = [link for link in episode_links if link is not None]

#     else:
#         print(f"You done goofed. Provide a full and valid wikipedia link to a 'list of episodes' page, otherwise this won't work at all. Status: {response.status_code}")
    
#     return(episode_links)

# # episode_links = get_episode_links()

# def extract_episode_summaries(episode_links = get_episode_links()):     
#     # synopsis_textfile_path = "langchain/series-synopsis.txt"
#     synopsis_textfile_path = "series-synopsis.txt"
#     if os.path.exists(synopsis_textfile_path):
#         print("File already exists. Stopping function.")
#         return
    
#     # # # now extract text from each summary
#     for episode in episode_links:
#         try:
#             wiki_url_prefix = "https://en.wikipedia.org"
#             current_episode_link = f"{wiki_url_prefix}{episode}"
            
#             this_response = requests.get(current_episode_link)    
#             episode_soup = BeautifulSoup(this_response.text, "html.parser")
            
#             # synopsis_h2_span = episode_soup.find('span', {'class': 'mw-headline', 'id': 'Synopsis'})
#             synopsis_h2_span = episode_soup.find('span', {'class': 'mw-headline'}, id=lambda x: x and x.lower() in {'plot', 'synopsis'})
#             synopsis_h2 = synopsis_h2_span.find_parent('h2')
            
#             synopsis_paragraphs = []
#             current_element = synopsis_h2.find_next_sibling()
#             while current_element and current_element.name != 'h2':
#                 if current_element.name == 'p':
#                     synopsis_paragraphs.append(current_element.get_text(strip = True, separator = " "))
#                 current_element = current_element.find_next_sibling()
                
#             synopsis_text = ' '.join(synopsis_paragraphs).replace(" 's", "'s")
            
#             # synopsis_textfile_path = "langchain/series-synopsis.txt"
#             synopsis_textfile_path = "series-synopsis.txt"

            
#             with open(synopsis_textfile_path, 'a', encoding='utf-8') as synopsis_textfile:
#                 synopsis_textfile.write(synopsis_text + "\n")
#         except Exception as e:
#             print(f"Error processing episode. {e}")
            
# extract_episode_summaries()

# synopsis_textfile_path = "langchain/series-synopsis.txt"
synopsis_textfile_path = "series-synopsis.txt"


# # # now to try embedding

# loader = TextLoader(synopsis_textfile_path)  
# text = loader.load()
with open(synopsis_textfile_path, encoding='utf-8') as f:
    text = f.read()

# # # will this text split work better than a /n which was inadvertently splitting at least one synopsis into two..?
docs = text.split("\n")

documents = []
for i, doc_text in enumerate(docs):
    doc = Document(page_content=doc_text, metadata={"id": i})
    documents.append(doc)

embedding_method = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embedding_method)
retriever = db.as_retriever()

        


# 5. Build an app with streamlit
def main():
    st.set_page_config(
        page_title="Sopranos Chatbot", page_icon=":bird:")

    st.header("Sopranos Chatbot :bird:")
    message = st.text_area("Ask a question...")

    if message:
        st.write("Generating message...")

        prompt = ChatPromptTemplate.from_template("""
                                                  Use your knowledge of the Sopranos: {context}
                                                  to answer the following question: {question}
                                                  """)
        model = ChatOpenAI()
        chain = (
            {"context": retriever, "question": RunnablePassthrough()} | prompt | model | StrOutputParser()
            )
        
        result = chain.invoke(message)
        
        st.info(result)


if __name__ == '__main__':
    main()

