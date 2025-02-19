import argparse
import glob
import logging
import os
import random
import numpy as np
import torch
import time
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,  TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup, AdamW,
                          RobertaConfig,
                          RobertaTokenizer, T5Config, T5EncoderModel, RobertaModel)

from model import CodeT5ForSequenceClassification, CodeBERTForSequenceClassification
from utils import (compute_metrics, convert_examples_to_features, output_modes, processors, output_weights,
                   delete_code_pattern)

logger = logging.getLogger(__name__)
MODEL_CLASSES = {'roberta': (RobertaConfig, CodeBERTForSequenceClassification, RobertaTokenizer),
                 'codet5': (T5Config, CodeT5ForSequenceClassification, RobertaTokenizer)}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, train_dataset, model, tokenizer, optimizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, t_total)
    # 初始化计时器
    total_training_start = time.time()  # 总训练时间开始计时

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))

    # Train!
    logger.info("***** Running training with early stopping *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    print(torch.cuda.is_available())
    print(args.device)

    global_step = args.start_step
    tr_loss, logging_loss = 0.0, 0.0
    best_f1 = 0.0  # 使用 F1 作为评估指标
    no_improve_epochs = 0  # 记录没有提升的 epoch 次数
    early_stopping_patience = args.early_stopping_patience  # 从命令行参数中获取耐心值
    model.zero_grad()
    train_iterator = trange(args.start_epoch, int(args.num_train_epochs), desc="Epoch",
                            disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproducibility
    epoch_times = []  # 保存每个 epoch 的耗时
    model.train()

    for epoch_idx, _ in enumerate(train_iterator):
        epoch_start_time = time.time()  # 每个 epoch 开始计时
        tr_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                      'labels': batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                try:
                    from apex import amp
                except ImportError:
                    raise ImportError(
                        "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer, checkpoint=str(global_step))
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                            logger.info('loss %s', str(tr_loss - logging_loss))
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss
                if args.max_steps > 0 and global_step > args.max_steps:
                    # epoch_iterator.close()
                    break

        # Evaluate after each epoch
        if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
            results = evaluate(args, model, tokenizer, checkpoint=str(epoch_idx))
            current_f1 = results['f1']  # 获取当前 F1 分数

            # 早停逻辑
            if current_f1 > best_f1:
                best_f1 = current_f1
                no_improve_epochs = 0  # 重置耐心值
                # 保存最佳模型
                output_dir = os.path.join(args.output_dir, 'checkpoint-best')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save, output_dir + '/pytorch_model.bin')
                tokenizer.save_pretrained(output_dir)

                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", output_dir)

                logger.info(f"Epoch {epoch_idx}: New best F1 score {current_f1}, saving model.")
            else:
                no_improve_epochs += 1
                logger.info(f"Epoch {epoch_idx}: F1 score {current_f1}, no improvement for {no_improve_epochs} epoch(s).")

                if no_improve_epochs >= early_stopping_patience:
                    logger.info(f"Early stopping triggered. Best F1: {best_f1}")
                    train_iterator.close()
                    break
        epoch_time = time.time() - epoch_start_time  # 当前 epoch 结束计时
        epoch_times.append(epoch_time)
        logger.info(f"Epoch {epoch_idx} finished in {epoch_time:.2f} seconds.")

    total_training_time = time.time() - total_training_start  # 总训练时间结束计时
    logger.info(f"Total training time: {total_training_time:.2f} seconds.")

    # 保存时间记录到文件
    time_log_file = os.path.join(args.output_dir, "training_time_log.txt")
    with open(time_log_file, "w") as f:
        f.write(f"Total training time: {total_training_time:.2f} seconds\n")
        for idx, epoch_time in enumerate(epoch_times):
            f.write(f"Epoch {idx + 1} time: {epoch_time:.2f} seconds\n")

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def evaluate(args, model, tokenizer, checkpoint=None, prefix="", mode='dev'):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        if (mode == 'dev'):
            eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, ttype='dev')
        elif (mode == 'test'):
            eval_dataset, instances = load_and_cache_examples(args, eval_task, tokenizer, ttype='test')

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                          # XLM don't use segment_ids
                          'labels': batch[3],
                          'output_attentions': True}
                outputs = model(**inputs)
                # print(inputs.keys())  # 输出所有键
                # print(outputs.keys())  # 打印输出中的键

                if args.output_attention:
                    attentions = outputs.attentions
                    # print(attentions)
                    tokens = []
                    for token in inputs['input_ids']:
                        # print(token)
                        tokens.append(tokenizer.convert_ids_to_tokens(token))
                    # print(tokens)
                    output_weights(attentions, tokens, args.output_dir, output_words=False, outputFileIndex=0)


                tmp_eval_loss, logits = outputs[:2]

                if args.model_layer > 12:
                    logger.warning("layer number exceed...")
                    logger.warning("layer number should under 12")
                elif args.model_layer < 12:
                    logits = model.classifier(outputs[2][1:][args.model_layer])

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:

                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
        # eval_accuracy = accuracy(preds,out_label_ids)
        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds_label = np.argmax(preds, axis=1)
        result = compute_metrics(eval_task, preds_label, out_label_ids)
        results.update(result)
        if (mode == 'dev'):
            output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
            with open(output_eval_file, "a+") as writer:
                logger.info("***** Eval results {} *****".format(prefix))
                writer.write('evaluate %s\n' % checkpoint)
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
        elif (mode == 'test'):
            output_test_file = args.test_result_dir
            output_dir = os.path.dirname(output_test_file)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(output_test_file, "w") as writer:
                logger.info("***** Output test results *****")
                all_logits = preds.tolist()
                for i, logit in tqdm(enumerate(all_logits), desc='Testing'):
                    instance_rep = '<CODESPLIT>'.join(
                        [item.encode('ascii', 'ignore').decode('ascii') for item in instances[i]])
                    writer.write(instance_rep + '<CODESPLIT>' + '<CODESPLIT>'.join([str(l) for l in logit]) + '\n')
                for key in sorted(result.keys()):
                    print("%s = %s" % (key, str(result[key])))

    return results


def load_and_cache_examples(args, task, tokenizer, ttype='train'):
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    if ttype == 'train':
        file_name = args.train_file.split('.')[0]
    elif ttype == 'dev':
        file_name = args.dev_file.split('.')[0]
    elif ttype == 'test':
        file_name = args.test_file.split('.')[0]
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}_{}'.format(
        file_name,
        str(args.model_type),
        str(args.max_seq_length),
        str(args.prune_strategy),
        str(task)))
    # print(cached_features_file)
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        if ttype == 'test':
            examples, instances = processor.get_test_examples(args.data_dir, args.test_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if ttype == 'train':
            examples = processor.get_train_examples(args.data_dir, args.train_file)
        elif ttype == 'dev':
            examples = processor.get_dev_examples(args.data_dir, args.dev_file)
        elif ttype == 'test':
            examples, instances = processor.get_test_examples(args.data_dir, args.test_file)
        #
        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
                                                cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                sep_token=tokenizer.sep_token,
                                                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 1,
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                prune_strategy=args.prune_strategy,
                                                lang=args.lang)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    if (ttype == 'test'):
        return dataset, instances
    else:
        return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--task_name", default='vulnerability', type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default="./model/output1", type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default=None, type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to run predict on the test set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=20.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument("--train_file", default="train_processed.txt", type=str,
                        help="train file")
    parser.add_argument("--dev_file", default="val_processed.txt", type=str,
                        help="dev file")
    parser.add_argument("--test_file", default="test_processed.txt", type=str,
                        help="test file")
    parser.add_argument("--pred_model_dir", default=None, type=str,
                        help='model for prediction')
    parser.add_argument("--test_result_dir", default='test_results.tsv', type=str,
                        help='path to store test result')
    parser.add_argument("--model_layer", default=12, type=int, help='model layer number')
    parser.add_argument("--output_attention", action='store_true', help='whether output transformer attention')
    parser.add_argument("--lang", default='cpp', type=str, help='target program language')
    parser.add_argument("--prune_strategy", default="None", type=str, help="prune strategy")
    parser.add_argument("--early_stopping_patience", type=int, default=2,
                        help="Number of epochs with no improvement in F1 score before early stopping is triggered.")

    args = parser.parse_args()
    logger.info(args)

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.no_cuda:  # 如果禁用 GPU，强制使用 CPU
        device = torch.device("cpu")
        args.n_gpu = 0
    else:  # 否则，检查 GPU 是否可用
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = 1 if torch.cuda.is_available() else 0  # 如果有 GPU 可用，设置为 1，否则为 0

    args.device = device

    # 打印设备信息，供调试用
    print(f"Using device: {args.device}")
    if args.n_gpu > 0:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    logger.info("-"*20)
    logger.info(args.output_attention)


    # # Set seed
    # set_seed(args)

    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
    #                                       #num_labels=num_labels,
    #                                       #finetuning_task=args.task_name,
    #                                       ignore_mismatched_sizes=True)

#    config.max_position_embeddings = 128
#    config.vocab_size = 5026
#    logger.info(config)


    #testing
#    from transformers import RobertaModel
#    import sys
#    model = RobertaModel.from_pretrained('microsoft/codebert-base', config=config, ignore_mismatched_sizes=True)
#    torch.save(model.state_dict(), './m/test_2')
#    logger.info("exited")
#    sys.exit()

    if args.tokenizer_name:
        tokenizer_name = args.tokenizer_name
    elif args.model_name_or_path:
        tokenizer_name = 'roberta-base'
    logger.info("tokenizer_name is %s", tokenizer_name)
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name, do_lower_case=args.do_lower_case)
    if args.model_type == 'codet5':
        logger.info('codet5')
        encoder = T5EncoderModel.from_pretrained('./codet5')
        model = CodeT5ForSequenceClassification(encoder)
    else:
        logger.info('codebert')
        encoder = RobertaModel.from_pretrained('./codebert')
        model = CodeBERTForSequenceClassification(encoder)
    # 手动加载检查点权重
    checkpoint = os.path.join(args.output_dir, 'pytorch_model.bin')
    if os.path.exists(checkpoint):
        logger.info(f"Loading complete model object from {checkpoint}")
        if args.model_name_or_path == './codet5':
            model = torch.load(checkpoint, map_location=args.device)  # 直接加载完整模型对象
        elif args.model_name_or_path == './codebert':
            model = torch.load(checkpoint, map_location=args.device)  # 直接加载完整模型对象
        else:
            model = torch.load(args.model_name_or_path)
    else:
        logger.error(f"Checkpoint not found at {checkpoint}")



    # Make sure only the first process in distributed training will download model & vocab
    if args.local_rank == 0:
        torch.distributed.barrier()

    # Distributed and parallel training
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, ttype='train')
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, optimizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        # if args.model_type == 'codet5':
        #     torch.save(model_to_save, args.output_dir + '/pytorch_model.bin')
        # else:
        #     model_to_save.save_pretrained(args.output_dir)
        torch.save(model_to_save, args.output_dir + '/pytorch_model.bin')
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        if args.model_type == 'codet5':
            # 加载 T5 编码器
            encoder = T5EncoderModel.from_pretrained('./codet5')
            # 初始化自定义分类模型
            model = CodeT5ForSequenceClassification(encoder)
        elif args.model_type == 'roberta':
            # 加载 CodeBERT编码器
            encoder = RobertaModel.from_pretrained('./codebert')
            # 初始化自定义分类模型
            model = CodeBERTForSequenceClassification(encoder)
        else:
            # 对其他 Hugging Face 模型，继续使用 from_pretrained
            model = model_class.from_pretrained(checkpoint)
        # 将模型加载到设备
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            # print(checkpoint)
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            if args.model_type == 'codet5':
                # 加载 T5 编码器
                encoder = T5EncoderModel.from_pretrained('./codet5')
                # 初始化自定义分类模型
                model = CodeT5ForSequenceClassification(encoder)
            elif args.model_type == 'roberta':
                # 加载 CodeBERT编码器
                encoder = RobertaModel.from_pretrained('./codebert')
                # 初始化自定义分类模型
                model = CodeBERTForSequenceClassification(encoder)
            else:
                # 对其他 Hugging Face 模型，继续使用 from_pretrained
                model = model_class.from_pretrained(checkpoint)
            # 将模型加载到设备
            model.to(args.device)
            result = evaluate(args, model, tokenizer, checkpoint=checkpoint, prefix=global_step)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    if args.do_predict:
        logger.info("-"*20 + "starting predict" + "-"*20)
        logger.info(args.pred_model_dir)
        if args.pred_model_dir != None:
            if args.model_type == 'codet5':
                encoder = T5EncoderModel.from_pretrained('./codet5')
                model = CodeT5ForSequenceClassification(encoder)
                model = torch.load(args.pred_model_dir)
            elif args.model_type == 'roberta':
                encoder = RobertaModel.from_pretrained('./codebert')
                model = CodeBERTForSequenceClassification(encoder)
                model = torch.load(args.pred_model_dir)
            else:
                model = model_class.from_pretrained(args.pred_model_dir, output_attentions=True)
        model.to(args.device)
        evaluate(args, model, tokenizer, checkpoint=None, prefix='', mode='test')
    return results

if __name__ == "__main__":
    main()

# # 配置日志
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# if __name__ == "__main__":
#     # Args 类配置
#     class Args:
#         data_dir = "../dataset/processed_data"  # 替换为你的数据路径
#         model_type = "codet5"  # 使用 CodeBERT 模型
#         model_name_or_path = "./codet5"
#         task_name = "vulnerability"  # 任务名称
#         max_seq_length = 256
#         prune_strategy = "slim"
#         lang = "cpp"
#         local_rank = -1
#         train_file = "train_processed.txt"
#         dev_file = "val_processed.txt"
#         test_file = "test_processed.txt"
#
#     args = Args()
#     # 初始化 tokenizer
#     tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    # # 加载训练数据
    # logger.info("Loading training dataset...")
    # try:
    #     train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, ttype='train')
    #     logger.info(f"Training dataset size: {len(train_dataset)}")
    #     # logger.info(f"First training example: {train_dataset[0] if len(train_dataset) > 0 else 'None'}")
    # except Exception as e:
    #     logger.error(f"Failed to load training dataset: {e}")
    # processor = processors[args.task_name]()
    # simplified_details={}
    # examples = processor.get_train_examples(args.data_dir, args.train_file)
    # for (ex_index, example) in enumerate(examples):
    #     print(f"before slim: {example.text_a}")
    #     example.text_a = delete_code_pattern(example.text_a, args.prune_strategy, 'cpp')
    #     simplified_details[ex_index] = {'label':example.label, 'text_a':example.text_a}
    #     print(f"after slim: {example.text_a}")
    #
    # utils.save_result_to_file('train',args.prune_strategy, simplified_details)
    # 加载验证数据
    # logger.info("Loading validation dataset...")
    # try:
    #     dev_dataset = load_and_cache_examples(args, args.task_name, tokenizer, ttype='dev')
    #     logger.info(f"Validation dataset size: {len(dev_dataset)}")
    #     logger.info(f"First validation example: {dev_dataset[0] if len(dev_dataset) > 0 else 'None'}")
    # except Exception as e:
    #     logger.error(f"Failed to load validation dataset: {e}")

    # processor = processors[args.task_name]()
    # simplified_details={}
    # examples = processor.get_dev_examples(args.data_dir, args.train_file)
    # for (ex_index, example) in enumerate(examples):
    #     print(f"before slim: {example.text_a}")
    #     example.text_a = delete_code_pattern(example.text_a, args.prune_strategy, 'cpp')
    #     simplified_details[ex_index] = {'label':example.label, 'text_a':example.text_a}
    #     print(f"after slim: {example.text_a}")
    #
    # utils.save_result_to_file('dev',args.prune_strategy, simplified_details)
    # 加载测试数据
    # logger.info("Loading testing dataset...")
    # try:
    #     test_dataset, instances = load_and_cache_examples(args, args.task_name, tokenizer, ttype='test')
        # logger.info(f"Testing dataset size: {len(test_dataset)}")
        # logger.info(f"First testing example: {test_dataset[0] if len(test_dataset) > 0 else 'None'}")
        # logger.info(f"Number of test instances: {len(instances)}")
    # except Exception as e:
    #     logger.error(f"Failed to load testing dataset: {e}")

    # processor = processors[args.task_name]()
    # examples, instances = processor.get_test_examples(args.data_dir, args.test_file)
    # simplified_details = {}
    # for (ex_index, example) in enumerate(examples):
    #     print(f"before slim: {example.text_a}")
    #     example.text_a = delete_code_pattern(example.text_a, args.prune_strategy, 'cpp')
    #     simplified_details[ex_index] = {'label': example.label, 'text_a': example.text_a}
    #     print(f"after slim: {example.text_a}")
    #
    # utils.save_result_to_file('test', args.prune_strategy, simplified_details)
