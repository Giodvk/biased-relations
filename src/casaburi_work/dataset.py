import pandas as pd
import random
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split

def build_surgical_dataset(seed=42):
    print("1. Download dei Dati Generici/Geografici...")
    # Usiamo un dataset leggero con frasi neutrali su paesi e capitali.
    geo_data = load_dataset("xiaozeroone/Country-city-animals", 'Corpus_narrative', split="train", )
    
    # Questo dataset ha varie colonne, estraiamo le frasi narrative.
    # Se la colonna 'text' non esiste, potresti dover adattare il nome (es. 'sentence')
    geo_sentences = []
    # Molti dataset simili hanno chiavi come 'Corpus_narrative' o semplicemente generano stringhe
    # Qui simuliamo l'estrazione:
    for row in geo_data:
        # Adatta questa chiave in base a come HuggingFace restituisce il dict
        if 'text' in row:
             geo_sentences.append(row['text'])
        elif 'sentence' in row:
             geo_sentences.append(row['sentence'])
             
    # Fallback se il dataset è strutturato diversamente (molto comune su HF)
    if len(geo_sentences) == 0:
         print("   [!] Fallback su testo generico wikitext per sicurezza.")
         wiki_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
         geo_sentences = [t for t in wiki_data['text'] if len(t.strip()) > 30][:2000]

    print(f"   -> Ottenute {len(geo_sentences)} frasi neutrali.")

    print("\n2. Download di CrowS-Pairs (Le Trappole Anti-Bias)...")
    # Scarica il dataset CrowS-Pairs
    crows_dataset = load_dataset('csv', data_files="biased-relations-main\\src\\casaburi_work\\crows_pairs_anonymized.csv")

    train_crows, test_crows = train_test_split(crows_dataset['train'], test_size=0.2, random_state=seed)
    
    anti_stereotypes = []
    for row in test_crows:
        # A noi interessa "curare" il modello, quindi usiamo sent_less (l'anti-stereotipo)
        anti_stereotypes.append({
            "text": row["sent_less"],
            "domain": row["bias_type"] # Es: 'race-color', 'gender'. FONDAMENTALE per le maschere!
        })
    print(f"   -> Estratte {len(anti_stereotypes)} trappole controfattuali.")

    print("\n3. Mescolamento (Shuffling)...")
    # Creiamo una lista unica di dizionari.
    # Alle frasi neutre diamo il dominio "neutral" (così sappiamo di NON applicare la maschera L2)
    final_list = [{"text": s, "domain": "neutral"} for s in geo_sentences] + anti_stereotypes
    
    # Il seed garantisce che se rilanci lo script, l'ordine casuale sarà identico
    random.seed(seed)
    random.shuffle(final_list)

    # 4. Creazione dell'oggetto Dataset per PyTorch/HuggingFace
    hf_dataset = Dataset.from_pandas(pd.DataFrame(final_list))
    
    print(f"\nOperazione Completata! Frasi Totali: {len(hf_dataset)}")
    
    return hf_dataset

if __name__ == "__main__":
    # Avvia la creazione
    my_dataset = build_surgical_dataset()
    
    print("\nAnteprima delle prime 5 frasi del dataset miscelato:")
    for i in range(5):
        print(f"[{my_dataset[i]['domain'].upper()}] {my_dataset[i]['text'][:80]}...")
        
    # Salva il dataset processato sul tuo computer, così non devi riscaricarlo ogni volta!
    my_dataset.save_to_disk("./surgical_training_dataset")
    print("\nDataset salvato in ./surgical_training_dataset. Pronto per il DataLoader!")