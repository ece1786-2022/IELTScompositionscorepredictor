
import csv
import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from transformers import AutoModelForSequenceClassification, GPT2Tokenizer, get_scheduler
from torch.optim import AdamW
from tqdm.auto import tqdm
import evaluate
import matplotlib.pyplot as plt

class GPT2Dataset(Dataset):

  def __init__(self, topics, contents, labels, tokenizer, max_length=512):

    self.tokenizer = tokenizer
    self.input_ids = []
    self.attn_masks = []
    self.labels = []

    for i in range(len(topics)):
      input = topics[i].strip() + " " + contents[i].strip()
      encodings_dict = self.tokenizer(input, truncation=True, max_length=max_length, padding="max_length")
      self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
      if labels[i] != "<4":
        self.labels.append(classes[str(float(labels[i]))])
      else:
        self.labels.append(classes[labels[i]])
      self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    return self.input_ids[idx], self.labels[idx], self.attn_masks[idx]


topics = []
contents = []
labels = []

with open('/content/drive/MyDrive/ECE1786/project/IEL.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile)
    for i, row in enumerate(spamreader):
        if i != 0:
          topics.append(row[0])
          contents.append(row[1])
          labels.append(row[2])

classes = {
    '<4': 0,
    '4.0': 1,
    '4.5': 2,
    '5.0': 3,
    '5.5': 4,
    '6.0': 5,
    '6.5': 6,
    '7.0': 7,
    '7.5': 8,
    '8.0': 9,
    '8.5': 10,
    '9.0': 11
}

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")

tokenizer.pad_token = tokenizer.eos_token

batch_size = 2
dataset = GPT2Dataset(topics, contents, labels, tokenizer)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size


torch.manual_seed(0)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(
            train_dataset,
            sampler = RandomSampler(train_dataset),
            batch_size = batch_size
        )

validation_dataloader = DataLoader(
            val_dataset,
            sampler = SequentialSampler(val_dataset),
            batch_size = batch_size
        )
## make sure the order of data is the same every time
print(torch.sum(train_dataset[0][0]))

torch.manual_seed(0)
model = AutoModelForSequenceClassification.from_pretrained("gpt2-medium", num_labels=12)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-5)

num_epochs = 10
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

torch.manual_seed(0)

progress_bar = tqdm(range(num_training_steps))

model.config.pad_token_id = tokenizer.pad_token_id

train_accs = []
val_accs = []
for epoch in range(num_epochs):
    model.train()
    metric = evaluate.load("accuracy")
    for input, targets, attn_masks in train_dataloader:
        x = {
            "input_ids": input,
            "labels": targets,
            "attention_mask": attn_masks
        }

        batch = {k: v.to(device) for k, v in x.items()}
        outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    train_acc = metric.compute()['accuracy']
    train_accs.append(train_acc)

    print("Epoch: {}, Train acc: {}".format(epoch + 1, train_acc))
    torch.save(model.state_dict(), "./gpt2_{}.pt".format(epoch))

val_accs = []
for i in range(10):
  model.load_state_dict(torch.load("/content/drive/MyDrive/ECE1786/project/gpt2_{}.pt".format(i)))

  model.eval()
  model.config.pad_token_id = tokenizer.pad_token_id

  metric = evaluate.load("accuracy")
  for input, targets, attn_masks in validation_dataloader:
      x = {
          "input_ids": input,
          "labels": targets,
          "attention_mask": attn_masks
      }
      batch = {k: v.to(device) for k, v in x.items()}
      outputs = model(**batch)

      logits = outputs.logits
      predictions = torch.argmax(logits, dim=-1)
      metric.add_batch(predictions=predictions, references=batch["labels"])
  val_acc = metric.compute()['accuracy']
  val_accs.append(val_acc)

plt.plot(range(1, 11), val_accs, label='Validation Accuracy')
plt.plot(range(1, 11), train_accs, label='Training Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()