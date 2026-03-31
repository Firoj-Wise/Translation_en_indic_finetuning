import os
import glob
import pandas as pd

def build_combined_csv(input_dir: str, output_csv: str):
    data = []
    
    # Domains: Agriculture, Banking, Health, Micro_Insurance
    # Languages: english, nepali, maithili
    
    domains = set([f.split('_english.txt')[0] for f in os.listdir(input_dir) if '_english.txt' in f])
    
    for domain in domains:
        en_path = os.path.join(input_dir, f"{domain}_english.txt")
        npi_path = os.path.join(input_dir, f"{domain}_nepali.txt")
        mai_path = os.path.join(input_dir, f"{domain}_maithili.txt")
        
        with open(en_path, 'r', encoding='utf-8') as fen, \
             open(npi_path, 'r', encoding='utf-8') as fnpi, \
             open(mai_path, 'r', encoding='utf-8') as fmai:
            
            en_lines = fen.read().strip().split('\n')
            npi_lines = fnpi.read().strip().split('\n')
            mai_lines = fmai.read().strip().split('\n')
            
            min_len = min(len(en_lines), len(npi_lines), len(mai_lines))
            
            for i in range(min_len):
                en_text = en_lines[i].strip()
                npi_text = npi_lines[i].strip()
                mai_text = mai_lines[i].strip()
                
                if not en_text or not npi_text or not mai_text:
                    continue
                    
                # Pair 1: eng -> npi
                data.append({'domain': domain, 'src': en_text, 'tgt': npi_text, 'src_lang': 'eng_Latn', 'tgt_lang': 'npi_Deva'})
                # Pair 2: npi -> eng
                data.append({'domain': domain, 'src': npi_text, 'tgt': en_text, 'src_lang': 'npi_Deva', 'tgt_lang': 'eng_Latn'})
                # Pair 3: eng -> mai
                data.append({'domain': domain, 'src': en_text, 'tgt': mai_text, 'src_lang': 'eng_Latn', 'tgt_lang': 'mai_Deva'})
                # Pair 4: mai -> eng
                data.append({'domain': domain, 'src': mai_text, 'tgt': en_text, 'src_lang': 'mai_Deva', 'tgt_lang': 'eng_Latn'})

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Created {output_csv} with {len(df)} sentence pairs across 4 directions.")

if __name__ == "__main__":
    input_dir = "finetuning_data"
    output_dir = "full_dataset.csv"
    build_combined_csv(input_dir, output_dir)
