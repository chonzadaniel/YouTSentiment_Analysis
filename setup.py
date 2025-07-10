from setuptools import find_packages, setup

setup(
    name='YouTSentimentAnalysis',  # project name
    version='0.1.0',
    author='Emmanuel Daniel Chonza',
    author_email='chonzadaniel@gmail.com',
    description='A YouTube Sentiment Analysis project with NLP preprocessing and ML pipelines',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'spacy',
        'symspellpy',
        'beautifulsoup4',
        'pyyaml',
        'certifi',
        'nltk',
        'matplotlib',  # Optional if you use plots
        'seaborn'      # Optional for visualization
    ],
    classifiers=[
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='<=3.12'
)
