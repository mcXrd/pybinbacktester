# pybinbacktester
Solution for trading crypto on binance

# installation guide
1) in dj/dj/ create secrets.py file with 
BINANCE_API_KEY="yourapikey"
BINANCE_SECRET_KEY="yoursecretkey"
2) in dj/ create empty static_storage folder - this folder serves as
place for storing intermediate market data hdfs

3) before installing reqs - install
    - conda install -c anaconda psycopg2
    - conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
