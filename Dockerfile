FROM continuumio/miniconda3:latest

WORKDIR /app
COPY requirements.txt /app/requirements.txt
COPY dj/ /app/dj/

RUN conda create --name py36 python=3.6 psycopg2 twisted regex

RUN echo "source activate py36" > ~/.bashrc
ENV PATH /opt/conda/envs/py36/bin:$PATH

RUN pip install -r requirements.txt

CMD ["python"]