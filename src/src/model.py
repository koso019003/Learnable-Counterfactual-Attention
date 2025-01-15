import torch
import lightning
import torchmetrics
import torch.nn as nn
import torch.nn.functional as torch_f
from transformers import AutoModel


class TimeSeqMoSEMERTModel(lightning.LightningModule):
    def __init__(
        self,
        mert,
        num_class=10,
        learning_rate=0.0001,
        config=None,
    ):
        super(TimeSeqMoSEMERTModel, self).__init__()
        time_length = 749  # 749 for 10s, 2298 for all wav
        hidden_dim = config["representation_dim"]
        
        self.learning_rate = learning_rate
        self.use_cf = config["mose_cf"]
        self.loss_weights = config["loss_weights"]
        self.two_steps_training_stage = 0

        # Model architecture
        self.feature_extract_layer = config["mert_feature_layer"]
        if config["mert_feature_layer"] == "all":
            self.feature_extract_layer = list(range(13))
        
        self.mert = AutoModel.from_pretrained(mert, trust_remote_code=True)
        for param in self.mert.parameters():
            param.requires_grad = False

        feature_num = len(self.feature_extract_layer)
        if feature_num > 1:
            self.aggregator = nn.Conv2d(in_channels=feature_num, out_channels=1, kernel_size=1)
        else:
            self.aggregator = Passing()
        
        ## Attention layer
        self.time_att = AvgPoolAtt1D(time_length)
        self.z_dropout = nn.Dropout(config["main_dropout"])
        if self.use_cf:
            self.time_att_cf = AvgPoolAtt1D(time_length, config["random_att"])
            self.z_cf_dropout = nn.Dropout(config["cf_dropout"])
        
        self.hidden_layer = nn.Linear(768, hidden_dim)
        
        ## Ouput layer
        self.ouput_layer = nn.Linear(hidden_dim, num_class)

        # Metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_class)
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_class)
        
        self.phase = "train"
        self.save_hyperparameters()

    def set_forward(self, phase):
        assert phase in ('att', 'rep', 'test', 'predict', 'train', 'val')
        self.phase = phase

    def forward(self, batch):
        self.phase = 'test'
        return self._forward(batch)

    def _forward(self, batch):
        x, y = batch
        batch_size = y.shape[0]
        # x (batch_size, time_seq)

        with torch.no_grad():
            outputs = self.mert(**x, output_hidden_states=True)
        # outputs [13 layer, batch_size, time_seq, 768 feature_dim]

        mert_feature = torch.stack([outputs.hidden_states[i] for i in self.feature_extract_layer], dim=1)
        # mert_feature (batch_size, # layer, time_seq, 768 feature_dim)
        
        mert_feature = self.aggregator(mert_feature).squeeze()
        # mert_feature (batch_size, time_seq, 768 feature_dim)
        
        att = self.time_att(mert_feature)
        # att (batch_size, 1, time_seq)
                
        z = torch.matmul(att, mert_feature).squeeze()
        # z (batch_size, 768 feature_dim)
        
        z = self.z_dropout(z)
        
        x_rep = self.hidden_layer(z)
        # (batch, hidden_dim)

        y_logit = self.ouput_layer(x_rep)  # (batch, num_class)
        y_pred = torch_f.softmax(y_logit, dim=1)  # (batch, num_class)

        y_cf_loss = 0
        y_effect_loss = 0
        y_cf_entropy = 0
        attention_loss = 0
        att_cf, x_cf_rep, y_cf = None, None, None
        
        if self.use_cf:
            att_cf = self.time_att_cf(mert_feature)

            z_cf = torch.matmul(att_cf, mert_feature).squeeze()
            
            z_cf = self.z_cf_dropout(z_cf)
            
            x_cf_rep = self.hidden_layer(z_cf)  

            y_cf_logit = self.ouput_layer(x_cf_rep)  # (batch, num_class)
            y_cf = torch_f.softmax(y_cf_logit, dim=1)  # (batch, num_class)
            y_effect = y_pred - y_cf

            y_cf_loss += torch_f.cross_entropy(y_cf_logit, y)
            y_cf_entropy += -torch_f.cross_entropy(y_cf_logit, y_cf)
            y_effect_loss += torch_f.cross_entropy(y_effect, y)
            attention_loss += -torch_f.l1_loss(att, att_cf)

        if self.phase == "att":
            return att, att_cf
        elif self.phase == "rep":
            return x_rep, x_cf_rep
        elif self.phase in ("predict", "test"):
            return y_pred, y_cf

        y_loss = torch_f.cross_entropy(y_logit, y)
        y_entropy = -torch_f.cross_entropy(y_logit, y_pred)

        loss = 0
        loss += self.loss_weights["y_pred"] * y_loss
        
        loss += self.loss_weights["y_entropy"] * y_entropy * self.two_steps_training_stage
        loss += self.loss_weights["y_effect"] * y_effect_loss * self.two_steps_training_stage
        loss += self.loss_weights["y_cf"] * y_cf_loss * self.two_steps_training_stage
        loss += self.loss_weights["y_cf_entropy"] * y_cf_entropy * self.two_steps_training_stage
        loss += self.loss_weights["att_loss"] * attention_loss * self.two_steps_training_stage

        if self.phase == "train" or self.phase == "val":
            if self.phase == "train":
                self.log(
                    f"{self.phase}_step_loss",
                    loss,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                )
            self.log(
                f"{self.phase}_loss", 
                loss,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
            if self.use_cf:
                self.log_dict(
                    {
                        f"{self.phase}_y_loss": y_loss,
                        f"{self.phase}_y_entropy": y_entropy,
                        f"{self.phase}_y_effect_loss": y_effect_loss,
                        f"{self.phase}_y_cf_loss": y_cf_loss,
                        f"{self.phase}_y_cf_entropy": y_cf_entropy,
                        f"{self.phase}_att_loss": attention_loss,
                    },
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch_size,
                )

        return loss, y_pred

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        _, y = batch

        self.set_forward("train")
        loss, y_pred = self._forward(batch)

        self.train_acc.update(y_pred, y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        _, y = batch

        self.set_forward("val")
        _, y_pred = self._forward(batch)

        self.valid_acc.update(y_pred, y)
        self.log("val_acc", self.valid_acc, on_step=False, on_epoch=True, prog_bar=True)


    def on_validation_epoch_end(self) -> None:
        print()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class AvgPoolAtt1D(nn.Module):
    def __init__(self, num_feature, random_att=False):
        super(AvgPoolAtt1D, self).__init__()
        self.random_att = random_att
        self.num_feature = num_feature
        
        if not random_att:
            # tensor_input (batch_size, num_feature, d)
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                # tensor (batch_size, num_feature, 1)
                nn.Unflatten(dim=1, unflattened_size=(1, num_feature)),
                # tensor (batch_size, 1, num_feature, 1)
                nn.Conv2d(1, 16, kernel_size=1, stride=1),
                # tensor (batch_size, 16, num_feature, 1)
                nn.Tanh(),
                nn.Conv2d(16, 1, kernel_size=1, stride=1),
                # tensor (batch_size, 1, num_feature, 1)
                nn.Flatten(start_dim=2, end_dim=-1),
                nn.Softmax(dim=-1)
                # tensor (batch_size, 1, num_feature)
            )

    def forward(self, feature):
        # feature (batch_size, num_feature, 768 feature_dim)
        
        if not self.random_att:
            att = self.attention(feature)
            # att (batch_size, 1, num_feature)
        else:
            device = feature.get_device() if feature.get_device() >= 0 else None
            att = torch.rand(feature.shape[0], 1, feature.shape[1], device=device) * 2

        return  att


class Passing(nn.Module):
    def forward(self, x):
        return x
