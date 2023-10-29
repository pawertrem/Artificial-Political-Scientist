# Импорт библиотек﻿

import pandas as pd
import openai
import numpy as np
import tiktoken
from typing import List
from typing import Dict
from typing import Tuple
import time
import re
from sentence_transformers import SentenceTransformer
import telebot

# Сохранение API-ключей для Telegram-бота и OpenAI

bot = telebot.TeleBot("...")
openai.api_key = "..."

# Импорт данных 

df = pd.read_csv(r'C:\Users\User\Desktop\APS\df', encoding = 'utf-8')
df = df.set_index(["Name", "id"])
df = df[(df['Text']!='') & (df['Text'].isna()!=True)]

enc = tiktoken.encoding_for_model("text-davinci-003")
df['tokens'] = df['Text'].apply(lambda x: len(enc.encode(x)))

context_embeddings = pd.read_csv(r'C:\Users\User\Desktop\APS\embeddings.csv')
context_embeddings = context_embeddings.to_dict('list')

for i in context_embeddings.keys():
    context_embeddings[i] = np.array(context_embeddings[i])

# Функции для создания эмбеддингов, подсчета семантического сходства и ранжирования релевантных фрагментов текста

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2') #(пример лидерборда: https://github.com/avidale/encodechka)

def get_embedding(text: str) -> List[float]:
    result = model.encode(text)
    return result

def compute_doc_embeddings(df: pd.DataFrame) -> Dict[Tuple[str, str], List[float]]:
    a = {}
    for idx, r in df.iterrows():
        b = {idx: get_embedding(r[0].replace("\n", " "))}
        a.update(b)
        time.sleep(1)
    return a

def vector_similarity(x: List[float], y: List[float]) -> float:
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts: Dict[Tuple[str, str], np.array]) -> List[Tuple[float, Tuple[str, str]]]:
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities

#Функция для создания подходящего промпта

MAX_SECTION_LEN = 2000
SEPARATOR = "\n* "

separator_len = len(enc.encode(SEPARATOR))

def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
        
        chosen_sections_len += document_section.tokens.values[0] + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.Text.values[0].replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
            
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))
    
    header = """Ответь на вопрос как можно правдивее, используя предоставленный контекст.\n\nКонтекст:\n"""
    
    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"


COMPLETIONS_API_PARAMS = {
    "temperature": 0.0,
    "max_tokens": 2000,
    "model": "text-davinci-003",
    "stop":['<<END>>']
}

#Функция для формулировани ответа

def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: Dict[Tuple[str, str], np.array],
    show_prompt: bool = False
) -> str:
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )
    
    if show_prompt:
        print(prompt)

    if len(enc.encode(prompt))>2097:
        prompt = enc.decode(enc.encode(prompt)[:2097])

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )
   
    return response["choices"][0]["text"].strip(" \n")

#Запуск Telegram-бота

@bot.message_handler(func = lambda _: True)
def handle_message(message):
    response = answer_query_with_context(message.text, df, context_embeddings)
    bot.send_message(chat_id = message.from_user.id, text = response)

bot.polling()
