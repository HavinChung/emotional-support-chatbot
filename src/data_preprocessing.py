import pandas as pd
import json
import os

esconv_train = pd.read_parquet("data/raw/esconv/train.parquet")
esconv_val = pd.read_parquet("data/raw/esconv/val.parquet")
esconv_test = pd.read_parquet("data/raw/esconv/test.parquet")

emp_train = pd.read_parquet("data/raw/empathetic/train.parquet")
emp_val = pd.read_parquet("data/raw/empathetic/val.parquet")
emp_test = pd.read_parquet("data/raw/empathetic/test.parquet")

def parse_esconv(df):
    records = []
    for conv_id, row in enumerate(df['text']):
        data = json.loads(row)
        emotion = data.get('emotion_type', '')
        problem = data.get('problem_type', '')
        situation = data.get('situation', '')
        initial_intensity = data['survey_score']['seeker'].get('initial_emotion_intensity', None)
        final_intensity = data['survey_score']['seeker'].get('final_emotion_intensity', None)
        for turn_idx, turn in enumerate(data['dialog']):
            records.append({
                'conv_id': conv_id,
                'turn_idx': turn_idx,
                'speaker': turn['speaker'],
                'text': turn['text'],
                'strategy': turn.get('strategy', None),
                'emotion_type': emotion,
                'problem_type': problem,
                'situation': situation,
                'initial_intensity': initial_intensity,
                'final_intensity': final_intensity
            })
    return pd.DataFrame(records)

def clean_empathetic(df):
    return df.drop(columns=['Unnamed: 0'])

esconv_train_parsed = parse_esconv(esconv_train)
esconv_val_parsed = parse_esconv(esconv_val)
esconv_test_parsed = parse_esconv(esconv_test)

emp_train_clean = clean_empathetic(emp_train)
emp_val_clean = clean_empathetic(emp_val)
emp_test_clean = clean_empathetic(emp_test)

os.makedirs("data/processed/esconv", exist_ok=True)
os.makedirs("data/processed/empathetic", exist_ok=True)

esconv_train_parsed.to_parquet("data/processed/esconv/train.parquet", index=False)
esconv_val_parsed.to_parquet("data/processed/esconv/val.parquet", index=False)
esconv_test_parsed.to_parquet("data/processed/esconv/test.parquet", index=False)

emp_train_clean.to_parquet("data/processed/empathetic/train.parquet", index=False)
emp_val_clean.to_parquet("data/processed/empathetic/val.parquet", index=False)
emp_test_clean.to_parquet("data/processed/empathetic/test.parquet", index=False)