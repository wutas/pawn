FROM python:3
WORKDIR /code

COPY main.py /code
COPY requirements.txt /code
COPY  images/wait4.gif /code
COPY models/* /code/

RUN pip install -r requirements.txt

#CMD ["streamlit", "run", "main.py"]