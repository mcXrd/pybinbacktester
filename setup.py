from setuptools import setup

install_requires = ['django', 'python-binance', 'psycopg2', 'tables', 'pandas', 'jsonschema', 'requests', 'mock',
                    'responses']

setup(name='pybinbacktester', version='0.0.1', packages=[''], url='', license='BSD 3-Clause', author='mcXrd',
      author_email='', description='', install_requires=install_requires)
