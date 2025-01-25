from setuptools import setup, find_packages

setup(
    name='historical_text_classifier',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
        'requests',
        'seaborn',
        'matplotlib',
        'chardet'
    ],
    package_data={
        'historical_text_classifier': ['data/dataset.json']
    }
)