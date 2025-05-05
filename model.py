import torch.nn as nn

#Task 2
#Task A - sentence classification
#Task B - sentiment analysis
#Added 2 extra layers at the end of Distil-BERT model to handle the tasks
class Multi(nn.Module):

    """
    A multi‐task head on top of a transformer encoder for joint classification
    and sentiment analysis.

    Args:
        trans (nn.Module): Pretrained transformer model (e.g., BERT, DistilBERT).
        classes (int): Number of target classes for the classification task.
        sentiments (int): Number of target classes for the sentiment task.
    """

    def __init__(self, transformer, classes=5, sentiments=3):
        super(Multi, self).__init__()
        self.transformer = transformer
        self.classification_head = nn.Linear(self.transformer.config.hidden_size, classes)
        self.sentiment_head      = nn.Linear(self.transformer.config.hidden_size, sentiments)

    def forward(self, inp_ids, mask):

        """
        A multi‐task head on top of a transformer encoder for joint classification
        and sentiment analysis.

        Args:
            trans (nn.Module): Pretrained transformer model (e.g., BERT, DistilBERT).
            classes (int): Number of target classes for the classification task.
            sentiments (int): Number of target classes for the sentiment task.
        """
        transformer_output = self.transformer(input_ids=inp_ids, attention_mask=mask)
        cls_embedding = transformer_output.last_hidden_state[:, 0, :] # Extract the CLS token embedding
        classification_output = self.classification_head(cls_embedding)
        sentiment_output      = self.sentiment_head(cls_embedding)
        return classification_output, sentiment_output
