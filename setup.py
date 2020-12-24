from setuptools import setup

setup(name='intermediate-gradients',
      version='0.1.0',
      description='',
      author='Kieran Heese',
      author_email='kh8fb@virginia.edu',
      url='https://github.com/kh8fb/intermediate-gradients',
      packages=['intermediate_gradients'],
      install_requires=[
          "captum==0.2.0",
          "certifi==2020.4.5.2",
          "chardet==3.0.4",
          "click==7.1.2",
          "cycler==0.10.0",
          "filelock==3.0.12",
          "future==0.18.2",
          "idna==2.9",
          "joblib==0.15.1",
          "kiwisolver==1.2.0",
          "matplotlib==3.2.2",
          "numpy==1.19.0",
          "packaging==20.4",
          "pyparsing==2.4.7",
          "python-dateutil==2.8.1",
          "regex==2020.6.8",
          "requests==2.24.0",
          "sacremoses==0.0.43",
          "sentencepiece==0.1.91",
          "six==1.15.0",
          "tokenizers==0.7.0",
          "torch==1.5.1",
          "tqdm==4.46.1",
          "transformers==2.11.0",
          "urllib3==1.25.9",
      ]
     )
