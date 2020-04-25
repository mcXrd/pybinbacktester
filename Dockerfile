FROM continuumio/miniconda3:latest

WORKDIR /app
COPY requirements.txt /app/requirements.txt
COPY dj/ /app/dj/

RUN conda create --name py37 python=3.7 psycopg2 twisted regex

RUN echo "source activate py37" > ~/.bashrc
ENV PATH /opt/conda/envs/py37/bin:$PATH

RUN pip install -r requirements.txt

CMD ["python"]
