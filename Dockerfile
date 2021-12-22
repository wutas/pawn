FROM python:3.7
WORKDIR /code

#COPY . /code

COPY main.py /code/main.py
COPY requirements.txt /code/requirements.txt
COPY  images/wait4.gif /code/wait4.gif
COPY models/* /code/

RUN pip install -r requirements.txt

CMD ["streamlit", "run", "main.py"]
