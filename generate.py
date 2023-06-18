import openai
import re
import tiktoken
import pandas as pd
import os
from openai.embeddings_utils import get_embedding
import sys
from dotenv import load_dotenv


load_dotenv()

# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"
openai.api_type = "azure"
openai.api_key = os.environ.get("gpt_token")
openai.api_base = os.environ.get("gpt_endpoint")
openai.api_version = "2023-03-15-preview"

# Place input csv file
df = pd.read_csv(os.path.join(os.getcwd(), sys.argv[1]), encoding='utf8')

df_answer = df[['idx', 'content']]

# https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html
# #evaluation-order-matters
pd.options.mode.chained_assignment = None


# normalize input
# s is input text
def normalize_text(s, sep_token=" \n "):
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r". ,", "", s)
    # remove all instances of multiple spaces
    s = s.replace("..", ".")
    s = s.replace(". .", ".")
    s = s.replace("\n", "")
    s = s.strip()

    return s


df_answer['content'] = df_answer["content"].apply(lambda x: normalize_text(x))
tokenizer = tiktoken.get_encoding("cl100k_base")
df_answer['n_tokens'] = df_answer["content"].apply(lambda x: len(tokenizer.encode(x)))
df_answer = df_answer[df_answer.n_tokens < 8192]

# engine should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model
df_answer['ada_v2'] = df_answer["content"].apply(lambda x: get_embedding(x, engine=EMBEDDING_MODEL))

pd.DataFrame(df_answer).to_csv(os.environ.get("file_name"), index=False)
