import pandas as pd
import json
import regex as re

relation_id_mapping = {
    'birth date': 'P1',
    'Social Security Number': 'P2',
    'phone number': 'P3',
    'email address': 'P4',
    'home address': 'P5'
}

def process_record(record):

    pattern = r"^What is the\s+(birth date|home address)\s+of\s+(.*)\?$"
    replacement = r"What is \2's \1?"
    record['prompt'] = re.sub(pattern, replacement, record['prompt'].strip())
  
    subject_match = re.search(r"What is (.*?)'s (birth date|Social Security Number|phone number|email address|home address)", record.get('prompt', ''))
    if subject_match:
        subject = subject_match.group(1)
    else:
        subject = f"Person_{record.get('known_id')}"

    attribute = record.get('prompt', '').lower()
    if 'birth' in attribute:
        record['subject'] = subject
        record['template'] = "What is {}'s birth date?"
        record['relation_id'] = relation_id_mapping['birth date']
    elif 'social security' in attribute:
        record['subject'] = subject
        record['template'] = "What is {}'s Social Security Number?"
        record['relation_id'] = relation_id_mapping['Social Security Number']
    elif 'phone' in attribute:
        record['subject'] = subject
        record['template'] = "What is {}'s phone number?"
        record['relation_id'] = relation_id_mapping['phone number']
    elif 'email' in attribute:
        record['subject'] = subject
        record['template'] = "What is {}'s email address?"
        record['relation_id'] = relation_id_mapping['email address']
    elif 'address' in attribute:
        record['subject'] = subject
        record['template'] = "What is {}'s home address?"
        record['relation_id'] = relation_id_mapping['home address']
    return record



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='semeval25-unlearning-data/data', help='data directory')

    args = parser.parse_args()

    forget_val = pd.read_parquet(f"{args.data_dir}/forget_validation-00000-of-00001.parquet")
    retain_val = pd.read_parquet(f"{args.data_dir}/retain_validation-00000-of-00001.parquet")

    sample1 = forget_val[(forget_val['task_type']=='QA') & (forget_val['task']=='Task2')]
    sample2 = retain_val[(retain_val['task_type']=='QA') & (retain_val['task']=='Task2')]
    sample = pd.concat((sample1, sample2), axis=0, ignore_index=True)

    sample.loc[sample['input'].str.strip().str[-1] == '?', 'task_type']='QA'

    sample["known_id"] = sample.index.to_series()
    sample['subject'] = ''
    sample['template'] = ''
    sample['relation_id'] = ''

    sample["prompt"] = sample['input']
    sample["prediction"] = sample['output']
    sample['attribute'] = sample['output']
    sample.drop(['id', 'input', 'output', 'task', 'split', 'task_type'], axis=1)

    updated_data = [process_record(x) for x in sample.to_dict(orient='records')]

    with open('data/known_QA_T2.json', 'w') as f:
        f.write(json.dumps(updated_data))