"""
Setup script for Email Classifier package.
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read the contents of requirements file
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name="email-classifier",
    version="1.0.0",
    author="Email Classifier Team",
    author_email="team@example.com",
    description="Multi-label email classification with chained and hierarchical strategies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/email-classifier",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-asyncio>=0.15.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "isort>=5.9.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "email-classifier=src.core.pipeline:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    keywords="email classification, machine learning, nlp, multi-label, chained, hierarchical",
    project_urls={
        "Bug Reports": "https://github.com/your-username/email-classifier/issues",
        "Source": "https://github.com/your-username/email-classifier",
        "Documentation": "https://email-classifier.readthedocs.io/",
    },
)
