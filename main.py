import torch
from transformers import AutoModel, AutoTokenizer
from model import Multi
import pandas as pd
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn


def create_dataset(sentence,cats,senti,tokenizer,max_length=128):

    """
    Tokenize and encode sentences into a TensorDataset for multi-task learning.

    Args:
        sentences (List[str]): List of input sentences.
        categories (List[str]): Corresponding category labels for each sentence.
        sentiments (List[str]): Corresponding sentiment labels ("Positive", "Neutral", "Negative").
        tokenizer: Pretrained HuggingFace tokenizer.
        max_length (int, optional): Max token length for padding/truncation. Defaults to 128.

    Returns:
        TensorDataset: Contains input_ids, attention_mask, category_ids, sentiment_ids.
    """
      
    unique_cats = sorted(set(cats))
    print(unique_cats)
    category_mapping = {cat: i for i, cat in enumerate(unique_cats)}
    sentiment_mapping = {"Positive": 0, "Neutral": 1, "Negative": 2}
    encodings = tokenizer(list(sentence),truncation=True,padding="max_length",max_length=max_length,return_tensors="pt")
    category_ids = torch.tensor([category_mapping[cat] for cat in cats], dtype=torch.long)
    sentiment_ids = torch.tensor([sentiment_mapping[s] for s in senti], dtype=torch.long)
    return TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"],
        category_ids,
        sentiment_ids
    )    

def sentence_embeddings(sentences, tokenizer, model, device, max_length=128):

    #Task 1 

    """
    Print out the CLS embedding for the first two sentences.

    Args:
        sentences (List[str]): Sentences to embed.
        tokenizer: Pretrained HuggingFace tokenizer.
        model: HuggingFace transformer model.
        device: torch.device ("cpu" or "cuda").
        max_length (int, optional): Max token length. Defaults to 128.
    """

    model.eval()
    sentences = sentences[:2]
    with torch.no_grad():
        for sentence in sentences:
            inputs = tokenizer(sentence,return_tensors="pt",padding="max_length",truncation=True,max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state  
            cls_embedding = last_hidden[:, 0, :]    
            print(f"Sentence: {sentence}")
            print("Embedding:", cls_embedding.squeeze().cpu().numpy())

def train_loop(classification_tag, sentiment_tag, model, train_data, device, val_dataset = None):

    #Task 4

    """
    Fine-tune a transformer for joint category classification and sentiment analysis.

    Args:
        classification_tags (List[str]): Original category labels.
        sentiment_tags (List[str]): Original sentiment labels.
        transformer_model (nn.Module): Pretrained transformer to use as encoder.
        train_dataset (TensorDataset): Training data from create_dataset().
        device (torch.device): Device for training.
    """

    num_classes = len(set(classification_tag))
    num_sentiments = len(set(sentiment_tag))
    multitask_model = Multi(model, classes=num_classes, sentiments=num_sentiments).to(device)
    batch_size = 16
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = None
    if val_dataset is not None and len(val_dataset )> 0:
        val_loader = DataLoader(val_dataset, batch_size= 16)

    optimizer = torch.optim.AdamW(multitask_model.parameters(), lr=2e-5)
    criterion_cls  = nn.CrossEntropyLoss()
    criterion_sent = nn.CrossEntropyLoss()
    num_epochs = 3
    #Forward pass
    for epoch in range(1, num_epochs + 1):
        multitask_model.train()
        running_loss = 0.0
        total_cls_correct  = 0
        total_senti_correct= 0
        total_samples      = 0
        for batch in train_loader:
            input_ids, attention_mask, labels_cls, labels_sent = batch
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels_cls     = labels_cls.to(device)
            labels_sent    = labels_sent.to(device)

            optimizer.zero_grad()
            logits_cls, logits_sent = multitask_model(input_ids, attention_mask)

            loss_cls  = criterion_cls(logits_cls,  labels_cls)
            loss_sent = criterion_sent(logits_sent, labels_sent)
            loss = loss_cls + loss_sent

            loss.backward()
            optimizer.step()
            total_samples += input_ids.size(0)

            running_loss += loss.item()
            preds_cls   = torch.argmax(logits_cls, dim=1)
            preds_senti = torch.argmax(logits_sent, dim=1)
            total_cls_correct   += (preds_cls   == labels_cls).sum().item()
            total_senti_correct += (preds_senti == labels_sent).sum().item()

        # For handiling hypothetical data
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_cls_corr = 0
            val_senti_corr = 0
            val_samples = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids, attention_mask, cls_labels, senti_labels = batch
                    input_ids      = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    cls_labels     = cls_labels.to(device)
                    senti_labels   = senti_labels.to(device)

                    cls_logits, senti_logits = model(input_ids, attention_mask)
                    loss = criterion_cls(cls_logits, cls_labels) + criterion_sent(senti_logits, senti_labels)
                    val_loss += loss.item() * input_ids.size(0)
                    val_samples += input_ids.size(0)

                    preds_cls   = torch.argmax(cls_logits, dim=1)
                    preds_senti = torch.argmax(senti_logits, dim=1)
                    val_cls_corr   += (preds_cls   == cls_labels).sum().item()
                    val_senti_corr += (preds_senti == senti_labels).sum().item()

            avg_val_loss = val_loss / val_samples
            val_cls_acc  = val_cls_corr   / val_samples
            val_senti_acc= val_senti_corr / val_samples
            print(f" Val  loss: {avg_val_loss:.4f} | "f"Cls acc: {val_cls_acc:.4f} | Senti acc: {val_senti_acc:.4f}")

        #Metrics
        cls_acc  = total_cls_correct   / total_samples
        senti_acc= total_senti_correct / total_samples
        print(f"Epoch {epoch}/{num_epochs} — avg loss: {running_loss:.4f}")
        print(f"Cls acc: {cls_acc:.4f} | Senti acc: {senti_acc:.4f}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    checkpoint = "distilbert-base-uncased"
    tokenizer  = AutoTokenizer.from_pretrained(checkpoint)
    model      = AutoModel.from_pretrained(checkpoint).to(device)
    #Synthetic data generate from Chat-GPT to demostrate the functioning of training_loop
    data = [
    {"sentence": "The latest smartphone’s advanced AI features significantly improved battery life and overall performance during my daily usage.", "tag": "Technology", "sentiment": "Positive"},
    {"sentence": "Despite the hype around the new VR headset, I found its motion tracking to be unreliable in most applications.", "tag": "Technology", "sentiment": "Negative"},
    {"sentence": "The device supports both Bluetooth and Wi-Fi connectivity, making it compatible with most modern peripherals and networks.", "tag": "Technology", "sentiment": "Neutral"},
    {"sentence": "The software update introduced several critical security patches but unfortunately slowed down system boot times noticeably.", "tag": "Technology", "sentiment": "Neutral"},
    {"sentence": "I upgraded my laptop’s RAM to 32 GB, and now resource-intensive applications run without any lag or stuttering.", "tag": "Technology", "sentiment": "Positive"},
    {"sentence": "Our team’s defensive strategy held the opponents scoreless for three quarters, leading to an unexpected victory tonight.", "tag": "Sports", "sentiment": "Positive"},
    {"sentence": "The match was delayed due to heavy rain, causing both players to lose momentum and struggle on the slippery court.", "tag": "Sports", "sentiment": "Negative"},
    {"sentence": "The athlete’s rigorous training regimen includes daily sprints, weightlifting sessions, and precise nutritional planning for peak performance.", "tag": "Sports", "sentiment": "Neutral"},
    {"sentence": "Ticket prices for the championship game soared this year, making it difficult for many fans to afford seats near the field.", "tag": "Sports", "sentiment": "Negative"},
    {"sentence": "The coach’s halftime adjustments clearly motivated the players, resulting in a dramatic comeback during the final minutes of play.", "tag": "Sports", "sentiment": "Positive"},
    {"sentence": "The proposed legislation will undergo several committee reviews before any final votes are scheduled next month in Congress.", "tag": "Politics", "sentiment": "Neutral"},
    {"sentence": "Critics argue that the new tax reforms disproportionately benefit large corporations at the expense of middle-class families nationwide.", "tag": "Politics", "sentiment": "Negative"},
    {"sentence": "Voter turnout in the recent elections exceeded expectations, signaling renewed civic engagement across diverse demographic groups.", "tag": "Politics", "sentiment": "Positive"},
    {"sentence": "The senator’s speech addressed climate change concerns but lacked specific policy proposals to enforce meaningful environmental regulations.", "tag": "Politics", "sentiment": "Negative"},
    {"sentence": "Debates between the candidates remained civil but failed to cover essential topics such as healthcare affordability and economic inequality.", "tag": "Politics", "sentiment": "Negative"},
    {"sentence": "Regular exercise combined with a balanced diet has been proven to reduce the risk of chronic diseases like diabetes and heart conditions.", "tag": "Health", "sentiment": "Positive"},
    {"sentence": "Despite undergoing rigorous treatment, the patient experienced several unexpected side effects that required immediate medical attention.", "tag": "Health", "sentiment": "Negative"},
    {"sentence": "The hospital’s new electronic records system improved data accessibility but encountered occasional downtime during peak hours.", "tag": "Health", "sentiment": "Neutral"},
    {"sentence": "Researchers published promising early results from the clinical trial but cautioned that larger studies are necessary for conclusive evidence.", "tag": "Health", "sentiment": "Neutral"},
    {"sentence": "The public health campaign successfully raised awareness about vaccination benefits and increased immunization rates in underserved communities.", "tag": "Health", "sentiment": "Positive"},
    {"sentence": "The stock market rally has delivered impressive returns this quarter, boosting investor confidence in the technology sector’s growth potential.", "tag": "Finance", "sentiment": "Positive"},
    {"sentence": "Persistent inflation concerns and rising interest rates have caused bond yields to fluctuate unpredictably in recent weeks.", "tag": "Finance", "sentiment": "Neutral"},
    {"sentence": "Several investors lost significant capital after the cryptocurrency exchange suffered a major security breach last month.", "tag": "Finance", "sentiment": "Negative"},
    {"sentence": "The company’s quarterly earnings report revealed stagnant revenue growth despite aggressive cost-cutting measures implemented earlier this year.", "tag": "Finance", "sentiment": "Negative"},
    {"sentence": "Diversifying investments across stocks, bonds, and real estate remains a prudent strategy for long-term financial stability.", "tag": "Finance", "sentiment": "Positive"},
]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    df = pd.DataFrame(data)
    sentences = df['sentence'].tolist()
    classification_tag = df['tag'].tolist()
    sentiment_tag = df['sentiment'].tolist()
    sentence_embeddings(sentences,tokenizer,model,device) #Task 1
    train_data = create_dataset(sentences, classification_tag, sentiment_tag, tokenizer)
    train_loop(classification_tag, sentiment_tag, model, train_data, device) #Task 4

if __name__ == "__main__":
    main()