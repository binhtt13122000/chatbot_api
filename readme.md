## Chatbot with GPT

###### 1, Install Python 3.8

###### 2, Setup Virtual Environment

> pip install virtualenv

###### 3, Get GPT Token

> - Login to Azure Portal -> Azure Open AI -> Create Project -> Get Token
> - Create token for translator
> - Create file .env from .env.example
> - Set gpt_token and translate_key like sample file

###### 4, Get Dependencies

> pip3 install -r requirements.txt

###### 5, Run app

> flask run

###### 6, Test API

> curl --location 'http://localhost:5000/api/v1/chat-bot' \
--form 'question="Thời hạn của giấy chứng nhận kiến thức về an toàn thực phẩm?"'