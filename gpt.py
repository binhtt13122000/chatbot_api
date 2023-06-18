import openai
import re
import os
import pandas as pd
from openai.cli import display
from openai.embeddings_utils import get_embedding, cosine_similarity
import tiktoken
from translator import detect_language, translate
from dotenv import load_dotenv
from ast import literal_eval

load_dotenv()

# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"
openai.api_type = "azure"
openai.api_key = os.environ.get("gpt_token")
openai.api_base = os.environ.get("gpt_endpoint")
openai.api_version = "2023-03-15-preview"

# Place input csv file
df = pd.read_csv(os.path.join(os.getcwd(), os.environ.get("file_name")), encoding='utf8')

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


df_answer = pd.read_csv(os.path.join(os.getcwd(), os.environ.get("file_name")), encoding='utf8')
df_answer["ada_v2"] = df_answer["ada_v2"].apply(lambda x: literal_eval(x))


# search through the reviews for a specific product
def search_docs(data_frame, user_query, top_n=3, to_print=False):
    embedding = get_embedding(
        user_query,
        engine=EMBEDDING_MODEL
        # engine should be set to the deployment name you chose when you deployed the text-embedding-ada-002
        # (Version2) model
    )

    data_frame["similarities"] = data_frame.ada_v2.apply(lambda x: cosine_similarity(x, embedding))

    res = (
        data_frame.sort_values("similarities", ascending=False)
        .head(top_n)
    )
    if to_print:
        display(res)
    return res


def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def query_message(
        query: str,
        data_frame: pd.DataFrame,
        model: str,
        token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    res = search_docs(data_frame, query, top_n=10)
    results = res['content']
    introduction = 'Đây là chatbot để tư vấn về các thủ tục hành chính'
    question = f"\n\nQuestion: {query}"
    message = introduction
    # print(results)
    for string in results:
        next_article = f'\n\nThe law data:\n"""\n{string}\n"""'
        if (
                num_tokens(message + next_article + question, model=model)
                > token_budget
        ):
            break
        else:
            message += next_article
    return message + question


def ask(
        query: str,
        data_frame: pd.DataFrame = df_answer,
        model: str = GPT_MODEL,
        token_budget: int = 4096 - 500,
        print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, data_frame, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": 'Bạn trả lời các câu hỏi liên quan tư vấn về các thủ tục hành chính dựa trên '
                                      'tập dữ liệu của tôi, nếu câu hỏi không có trong tập dữ liệu, chatbot sẽ tìm '
                                      'kiếm trên internet thông tin về luật pháp Việt Nam và cung cấp các thông tin '
                                      'liên quan. Nếu người dùng đặt câu hỏi không liên quan đến luật pháp, '
                                      'chatbot sẽ trả lời: "Tôi không thể trả lời câu hỏi của bạn, vì nó không liên '
                                      'quan đến cơ sở dữ liệu.'},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        deployment_id='gpt-35-turbo',
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response["choices"][0]["message"]["content"]
    # Detect the language
    language = detect_language(response_message)
    if language != 'vi':
        return translate(response_message, language, 'vi')
    else:
        return response_message
