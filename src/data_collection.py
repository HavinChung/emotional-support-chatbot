from datasets import load_dataset
import os

esconv = load_dataset("thu-coai/esconv")
empathetic = load_dataset("bdotloh/empathetic-dialogues-contexts")


esconv['train'].to_parquet("data/raw/esconv/train.parquet")
esconv['validation'].to_parquet("data/raw/esconv/val.parquet")
esconv['test'].to_parquet("data/raw/esconv/test.parquet")

empathetic['train'].to_parquet("data/raw/empathetic/train.parquet")
empathetic['validation'].to_parquet("data/raw/empathetic/val.parquet")
empathetic['test'].to_parquet("data/raw/empathetic/test.parquet")
