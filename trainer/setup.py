from setuptools import find_packages, setup

setup(
    name="clv-trainer",
    version="0.1.0",
    description="XGBoost customer lifetime value regression for Vertex AI custom training",
    packages=find_packages(),
    install_requires=["google-cloud-bigquery", "db-dtypes", "pyarrow", "pandas", "cloudml-hypertune"],
)
