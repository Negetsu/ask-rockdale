import pandas as pd
from langchain_community.document_loaders.dataframe import DataFrameLoader
excel_file = "data/Ordinances.xlsx"
ordinances = pd.read_excel(excel_file)
ordinances = ordinances.dropna(subset=['Unnamed: 4'])
ordinances['Unnamed: 4'] = ordinances['Unnamed: 4'].astype(str)
ordifix = pd.DataFrame(ordinances)
ordiloader = DataFrameLoader(ordifix,page_content_column='Unnamed: 4')
ordidocs = ordiloader.load()
print("right before")
print(ordidocs)