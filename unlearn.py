import inspect
from typing import Optional
from torch.utils.data import Dataset, RandomSampler
from transformers import Trainer, DataCollatorWithPadding
import torch.nn.functional as F
import torch
from tqdm import trange
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForCausalLM
import os
import shutil
import json

os.environ["TOKENIZERS_PARALLELISM"] = 'true'

print(
    f"Available GPUs: {torch.cuda.device_count()}\nAvailable CPUs: {os.cpu_count()}")


class DataGenerator(Dataset):
    def __init__(self, forget_set: pd.DataFrame, retain_set: pd.DataFrame, tokenizer):
        super(DataGenerator, self).__init__()
        negative_data_dict = forget_set.sort_values(
            'task', ignore_index=True).to_dict(orient='records')
        positive_data_dict = retain_set.sort_values(
            'task', ignore_index=True).to_dict(orient='records')
        self.input_ids = []
        self.loss_mask = []
        self.factor = []
        self.length = []

        for i in trange(len(negative_data_dict), desc="Forget Set"):
            inp_ids, loss_mask = tokenizer.apply_chat_template(
                [{"role": "user", "content": negative_data_dict[i]['input']}],
                tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors='pt').values()
            loss_mask[:] = 0
            output_ids, op_att_mask = tokenizer(
                negative_data_dict[i]['output'],
                return_attention_mask=True, return_tensors='pt').values()
            inp_ids = torch.cat((inp_ids, output_ids), axis=-1)
            loss_mask = torch.cat((loss_mask, op_att_mask), axis=-1)

            self.input_ids.append(inp_ids.tolist()[0])
            self.loss_mask.append(loss_mask.tolist()[0])
            self.factor.append(-1)
            self.length.append(inp_ids.shape[1])

        for i in trange(len(positive_data_dict), desc="Retain Set"):
            inp_ids, loss_mask = tokenizer.apply_chat_template(
                [{"role": "user", "content": positive_data_dict[i]['input']}],
                tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors='pt').values()
            loss_mask[:] = 0
            output_ids, op_att_mask = tokenizer(
                positive_data_dict[i]['output'],
                return_attention_mask=True, return_tensors='pt').values()
            inp_ids = torch.cat((inp_ids, output_ids), axis=-1)
            loss_mask = torch.cat((loss_mask, op_att_mask), axis=-1)
            self.input_ids.append(inp_ids.tolist()[0])
            self.loss_mask.append(loss_mask.tolist()[0])
            self.factor.append(1)
            self.length.append(inp_ids.shape[1])

    def __getitem__(self, i):
        return dict(
            input_ids=self.input_ids[i],
            loss_mask=self.loss_mask[i],
            factor=self.factor[i],
            length=self.length[i]
        )

    def __len__(self):
        return len(self.input_ids)


class CustomDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        features = self.tokenizer.pad(features, padding=True)

        features['attention_mask'] = torch.tensor(features['attention_mask'])
        features['input_ids'] = torch.tensor(features['input_ids'])
        features['length'] = torch.tensor(features['length'])
        max_len = int(features['length'].max())
        for i in range(len(features['loss_mask'])):
            curr_seq_len = len(features['loss_mask'][i])
            features['loss_mask'][i] = F.pad(torch.tensor(
                features['loss_mask'][i]), (max_len-curr_seq_len, 0), value=0).unsqueeze(0)
        features['loss_mask'] = torch.concat(features['loss_mask'], 0)
        features['factor'] = torch.tensor(features['factor'])

        return features


class AscentPlusDescentTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.average_positive_loss = 0
        self.n_positive_examples = 0
        self.previous_positive_avg_loss = 0

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys = None,
    ):
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()

        return (loss, None, None)


    def compute_loss(self, model, inputs, return_outputs=False):
        try:
            if "factor" not in inputs.keys():
                return super().compute_loss(model, inputs, return_outputs)
            factors = inputs.pop("factor")
            if len(inputs) == 0:
                print("small inputs")
                print(inputs)
            negative_inputs = {key: val[factors == -1]
                               for key, val in inputs.items()}
            positive_inputs = {key: val[factors != -1]
                               for key, val in inputs.items()}
            if len(negative_inputs["input_ids"]) != 0:
                outputs = model(
                    negative_inputs['input_ids'], attention_mask=negative_inputs['attention_mask'])

                neg_labels = negative_inputs['input_ids']
                # only calculate loss for output
                neg_labels[negative_inputs['loss_mask'] == 0] = -100
                neg_shift_logits = outputs['logits'][..., :-1, :].contiguous()
                neg_shift_labels = neg_labels[..., 1:].contiguous()
                neg_shift_logits = neg_shift_logits.view(-1,
                                                 outputs['logits'].shape[-1])
                neg_shift_labels = neg_shift_labels.view(-1)

                neg_shift_labels = neg_shift_labels.to(neg_shift_logits.device)
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(neg_shift_logits, neg_shift_labels)

                negative_loss = loss * -1
                printable_neg_loss = negative_loss.detach().cpu().item()
            else:
                negative_loss = 0
                printable_neg_loss = 0
            change_in_loss = 0
            if len(positive_inputs["input_ids"]) != 0:
                outputs = model(
                    positive_inputs['input_ids'], attention_mask=positive_inputs['attention_mask'])

                labels = positive_inputs['input_ids']
                # only calculate loss for output
                labels[positive_inputs['loss_mask'] == 0] = -100
                shift_logits = outputs['logits'][..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_logits = shift_logits.view(-1,
                                                 outputs['logits'].shape[-1])
                shift_labels = shift_labels.view(-1)

                shift_labels = shift_labels.to(shift_logits.device)
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits, shift_labels)

                curr_epoch = int(self.state.epoch)
                multiplier = 1.2
                if curr_epoch == 0:
                    self.average_positive_loss += loss.detach().cpu().item()
                    self.n_positive_examples += len(labels)
                    self.previous_positive_avg_loss = self.average_positive_loss/self.n_positive_examples
                elif loss.detach().cpu().item() > self.previous_positive_avg_loss:
                    change_in_loss = loss.detach().cpu().item()-self.previous_positive_avg_loss
                    multiplier = min(
                        max(round((0.3*(6**change_in_loss)+0.8), 1), 1.2), 2.3)

                positive_loss = loss*multiplier

                printable_pos_loss = positive_loss.detach().cpu().item()
                actual_loss = printable_neg_loss+(printable_pos_loss/multiplier)

            else:
                positive_loss, printable_pos_loss, multiplier = 0, 0, 0
                actual_loss = printable_neg_loss
            loss = negative_loss + positive_loss

            self.log({'negative_loss': printable_neg_loss, 'positive_loss': printable_pos_loss, 'positive_loss_multiplier': multiplier,
                     'loss': loss.detach().cpu().item(), 'change_in_loss': change_in_loss, 'actual_loss': actual_loss})

            return (loss, outputs) if return_outputs else loss

        except BaseException as e:
            raise (e)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        return RandomSampler(self.train_dataset)

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(
                set(["label", "label_ids"] + self.label_names)
            )
            self._signature_columns.append("factor")


def unlearn(model_dir, output_dir, forget_dir, retain_dir, validation_dir=None, 
            trainable_layers = [1,2,3,4,5,6], kind='mlp', del_checkpoints=False,
            epochs=8, data_ratio=1.0):
    df1 = pd.read_json(f"{forget_dir}/forget.jsonl", lines=True)
    df2 = pd.read_json(f"{retain_dir}/retain.jsonl", lines=True)
    df_train = pd.concat((df1, df2), axis="rows", ignore_index=True)
    df_train = df_train[['input', 'output', 'split', 'task']]
    start_len = df_train.shape[0]
    df_train = df_train.sample(frac=data_ratio)
    print(f"Training length reduced to: {df_train.shape[0]} from {start_len}")

    if not bool(validation_dir):
        df_test = df_train.groupby('task').sample(500)
    else:
        df1 = pd.read_json(f"{validation_dir}/forget.jsonl", lines=True)
        df2 = pd.read_json(f"{validation_dir}/retain.jsonl", lines=True)
        valid = pd.concat((df1, df2), axis="rows", ignore_index=True)
        valid = valid[['input', 'output', 'split', 'task']]
        df_train = pd.concat((valid, df_train), axis=0, ignore_index=True)
        df_test = valid

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, padding_side='left', padding=True)
    data_collator = CustomDataCollator(tokenizer=tokenizer)

    train_dataset = DataGenerator(
        df_train[df_train['split'] == 'forget'], df_train[df_train['split'] == 'retain'], tokenizer)
    test_dataset = DataGenerator(
        df_test[df_test['split'] == 'forget'], df_test[df_test['split'] == 'retain'], tokenizer)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,device_map='auto', use_cache=False, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    if isinstance(trainable_layers, str):
        assert trainable_layers=='all'
        trainable_layers = [i for i in range(0,33)]

    for k, v in model.named_parameters():
        try:
            if int(k.split('.')[2]) in trainable_layers:
                if k.split('.')[3] == kind or kind == 'all':
                    print(f'Training Layer {k}')
                    continue
        except ValueError:
            v.requires_grad_(False)
        except IndexError:
            v.requires_grad_(False)
        v.requires_grad_(False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=12,
        gradient_accumulation_steps=10,
        learning_rate=1.5e-5,
        num_train_epochs=epochs,
        logging_dir="log_dir/",
        logging_strategy='epoch',
        logging_first_step=True,
        report_to=None,
        save_strategy='steps',
        save_total_limit=1,
        resume_from_checkpoint=True,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        eval_strategy='steps',
        prediction_loss_only=True,
        eval_steps=4,
        save_steps=4)

    unlearner = AscentPlusDescentTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
    )
    print("Starting training")
    unlearner.train()
    if del_checkpoints:
        checkpoints = os.listdir(args.output_dir)
        for checkpoint in checkpoints:
            shutil.rmtree(os.path.join(args.output_dir, checkpoint))
        
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    f = open(os.path.join(args.output_dir, "checkpoint.txt"), 'w')
    f.write(f"'Best checkpoint': {unlearner.state.best_model_checkpoint},'Best metric': {unlearner.state.best_metric}")
    f.close()

    return

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, help='pretrained model directory')
    parser.add_argument('--output_dir', type=str, help='Directory to save model into')
    parser.add_argument('--forget_dir', type=str, help='directory containing forget dataset')
    parser.add_argument('--retain_dir', type=str, help='directory containing retain dataset')
    parser.add_argument('--validation_dir', type=str, default=None, help='directory containing validation dataset')
    parser.add_argument('--trainable_layers', type=str, default="[0,1,2,3,4,5]", help='Layers to train')
    parser.add_argument('--train_kind', type=str, default='mlp', help='one of self_attn, mlp, all')
    parser.add_argument('--epochs', type=int, default=8, help='number of epochs')
    parser.add_argument('--data_ratio', type=float, default=1.0, help='ratio of data to use for training')

    args = parser.parse_args()
    if args.trainable_layers != 'all':
        args.trainable_layers = json.loads(args.trainable_layers)

    unlearn(args.model_dir, args.output_dir, args.forget_dir, args.retain_dir, 
            args.validation_dir, args.trainable_layers, args.train_kind, del_checkpoints=True,
            epochs=args.epochs, data_ratio=args.data_ratio)