#------------Import Data---------------
csv_path_cnt= ('/Connectome.csv')
cntset = pd.read_csv(csv_path_cnt)
csv_path_ann= ('/DoOR.csv')
X_ann = pd.read_csv(csv_path_ann)

ff=cntset[cntset["directionality"]=='feedforward']
otp=ff[(ff["pre_class"]=='ORN') & (ff["post_class"]=='ALPN')]
ptk=ff[(ff['pre_class']=='ALPN') & (ff['post_class']=='KC')]

#-----------ORN to ALPN mask: 'mask_otp'-----------------
otp_unique = otp[['pre_root_id','post_root_id']]
mask_otp_norm = pd.crosstab(
    index=otp_unique['pre_root_id'],    # Row: ORN ID
    columns=otp_unique['post_root_id']  # Column: ALPN ID
    , values=otp['syn_count']           # syn count 
    , aggfunc='sum'
    ).fillna(0)                         #NaN = 0 

mask_otp = torch.from_numpy(mask_otp_norm.T.values).float().to(device)  # (615, 2278)

#------------ORN to ALPN mask: 'mask_otp'-----------------
alpn_ids=mask_otp_norm.columns

ptk_unique=ptk[['pre_root_id', 'post_root_id']]
mask_ptk_norm = (
    pd.crosstab(
        index=ptk_unique['pre_root_id'],    # Row: ALPN ID 
        columns=ptk_unique['post_root_id'], # Column: KC ID
        values=ptk['syn_count'],
        aggfunc='sum'
    )
    .reindex(index=alpn_ids,   # Make ALPN ID 615
             fill_value=0)     
    .fillna(0)                 # NaN = 0
)
mask_ptk = torch.from_numpy(mask_ptk_norm.T.values).float().to(device)  # (4907,334)

#-----------------------MaskedLinear()----------------------------
class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, weight):
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.bias   = nn.Parameter(torch.zeros(out_features))

        self.register_buffer('mask', torch.sign(weight).float())    # +1/-1/0 mask

    def forward(self, x):
        w = torch.abs(self.weight) * self.mask
        # Use functional linear transformation
        return torch.nn.functional.linear(x, w, self.bias)

#---------------------Odor Classifier()---------------------------
class Odor_classifier(nn.Module):
  def __init__(self,input_dim,output_dim, mask1, mask2):
    super().__init__()
    self.net=nn.Sequential(
        MaskedLinear(input_dim,615, mask1),
        nn.ReLU(),
        MaskedLinear(615,4907, mask2),
        nn.ReLU(),
        nn.Linear(4907,output_dim)
    )
  def forward(self, x):
    return self.net(x)

model = Odor_classifier(input_dim=2278, output_dim=250, mask1=mask_otp, mask2=mask_ptk).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

#-----------------------Train & Test----------------------
cls_labels = []
for i in range(250):
  cls_labels.append(f'Odor_({i+1})')

le=LabelEncoder()
y_int=le.fit_transform(cls_labels)
y = y_int
y_out=torch.tensor(y,dtype=torch.long)

x_train=[]
y_train=[]
x_test=[]
y_test=[]

idx=torch.randint(0,250,(1000,))

for n in idx:
  noise=torch.normal(mean=0,std=0.2,size=(2278,))

  xi=X_ann.iloc[n.item()] # Convert tensor to scalar and use .iloc for integer-based indexing
  xn  = torch.zeros(2278, dtype=torch.float32)
  xn=torch.tensor(xi.values, dtype=torch.float32)+noise # Convert pandas Series to tensor
  x_train.append(xn)
  y_train.append(y_out[n]) # Append the individual element y_out[n]

idx=torch.randint(0,250,(200,))

for k in idx:
  noise=torch.normal(mean=0,std=0.2,size=(2278,))

  xi=X_ann.iloc[k.item()] # Convert tensor to scalar and use .iloc for integer-based indexing
  xn  = torch.zeros(2278, dtype=torch.float32)
  xn=torch.tensor(xi.values, dtype=torch.float32)+noise # Convert pandas Series to tensor
  x_test.append(xn)
  y_test.append(y_out[k]) # Append the individual element y_out[k]

x_train = torch.stack(x_train).to(dtype=torch.float32)
y_train = torch.stack(y_train).to(dtype=torch.long)
x_test = torch.stack(x_test).to(dtype=torch.float32)
y_test = torch.stack(y_test).to(dtype=torch.long)

train_ds = TensorDataset(x_train, y_train)
test_ds  = TensorDataset(x_test,  y_test)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

#---------------Train loop----------------

num_epochs = 100
train_losses, train_accs = [], []

for epoch in trange(num_epochs, desc="Epochs"):
    model.train()
    tr_loss = tr_acc = 0.0
    y_true = []
    y_pred = []
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        tr_loss += loss.item() * xb.size(0)
        tr_acc  += (logits.argmax(dim=1) == yb).float().sum().item()

        # for confusion matrix
        y_true.append(yb.cpu())
        y_pred.append(logits.argmax(dim=1).cpu())

    tr_loss /= len(train_loader.dataset)
    tr_acc  /= len(train_loader.dataset)
    train_losses.append(tr_loss)
    train_accs.append(tr_acc)

    print(f"Epoch {epoch+1:02d}/{num_epochs}  "
          f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.4%}  ")


#------------Test Accuracy---------
model.eval()
te_correct = 0
total_samples = 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        te_correct += (logits.argmax(dim=1) == yb).float().sum().item()
        total_samples += xb.size(0)
test_acc = te_correct / total_samples
print(f"Test set accuracy: {test_acc:.4%}")


#---------Train result plot---------
fig, ax1 = plt.subplots(figsize=(8, 4))
# Loss: 왼쪽 y축
ax1.plot(train_losses, color='tab:blue', label='Train Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
# Accuracy: 오른쪽 y축
ax2 = ax1.twinx()
ax2.plot(train_accs, color='tab:orange', label='Train Acc')
ax2.set_ylabel('Accuracy', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')

plt.title('Train Loss & Accuracy')
plt.show()


# ---------Confusion_matrix---------
y_true_all = torch.cat(y_true).numpy()
y_pred_all = torch.cat(y_pred).numpy()

cm = confusion_matrix(y_true_all, y_pred_all)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix")
plt.show()
