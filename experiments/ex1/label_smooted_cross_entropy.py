# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from rouge_score import rouge_scorer
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.data import iterators, encoders
from fairseq.models.roberta import RobertaModel
from torchviz import make_dot
import random

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion("label_smoothed_cross_entropy")
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.task = task
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--report-accuracy', action='store_true',
                            help='report accuracy metric')
        parser.add_argument('--ignore-prefix-size', default=0, type=int,
                            help='Ignore first N tokens')
        # fmt: on

    def reward(self, tokens, target):
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        tokens_lst = tokens.tolist()
        tgts_lst = target.tolist()
        r = torch.empty(tokens.size(0), device='cuda', requires_grad=False)
        for i in range(tokens.size(0)):
            token_str = ' '.join(map(str, tokens_lst[i]))
            tgt_str = ' '.join(map(str, tgts_lst[i]))
            score = scorer.score(token_str, tgt_str)
            r[i] = torch.tensor(score['rougeL'].fmeasure)
        return r
    
    def sts(self, roberta, gen, target):
        v2 = torch.tensor([2], dtype=int, device='cuda').repeat(gen.size(0),1)
        combined = torch.cat((gen, v2, target), dim=1)
        score = roberta.predict('sentence_classification_head', combined[:, 0:512], return_logits=True)
        return score

    def top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        """
        # assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            # indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            indices_to_remove = torch.zeros_like(logits, dtype=torch.uint8).scatter_(
                                                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove )
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
            cumulative_probs = torch.cumsum(f.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # indices_to_remove = sorted_indices[sorted_indices_to_remove]
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove.type(torch.bool))
            logits[indices_to_remove] = filter_value
        return logits

    def random_choice_prob_index(self, a, axis=1):
        r = np.expand_dims(np.random.rand(a.shape[1-axis]), axis=axis)
        return (a.cumsum(axis=axis) > r).argmax(axis=axis)

    def log1pexp(self, x):
        # more stable version of log(1 + exp(x))
        return torch.where(x < 50, torch.log1p(torch.exp(x)), x)

    def forward(self, model, sample, roberta, num, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        with torch.autograd.set_detect_anomaly(True):
            # Initialize
            lb_decay_step=1000
            lb_decay_rate=1
            sample_size = (
                sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
            )
            src_tokens = sample['net_input']['src_tokens']
            prev_output_tokens = sample['net_input']['prev_output_tokens']
            bsz, length = prev_output_tokens.size()
            (out_vae, extra) = model(**sample['net_input'])
            (mean_enc, lvar_enc, mean_dec, lvar_dec) = extra['kl']
            
            # VAE
            # KL-divergence
            kld = -0.5 * (1 + lvar_dec - lvar_enc
                            - torch.div(torch.pow(mean_dec - mean_enc, 2), torch.exp(lvar_enc))
                            - torch.div(torch.exp(lvar_dec), torch.exp(lvar_enc))).sum(dim=1)

            # Annealing schedule for KL
            num = model.count
            step_2 = int(num) % int(lb_decay_step/2)
            b_num = int(num) / int(lb_decay_step/2)
            b_num = b_num % 2
            lb = torch.tensor(min( (step_2+b_num*lb_decay_step/2)*2.0/lb_decay_step, 1 )).cuda().detach()

            # Loss
            loss_ce, _ = self.compute_loss(model, (out_vae, extra), sample, reduce=reduce)
            kld_2 = sample_size * kld.mean()
            loss = loss_ce + lb * kld_2

            loss_gen = torch.tensor(0., dtype=torch.float32, device='cuda')

        logging_output = {
            "loss": loss_ce.detach().data,
            "nll_loss": kld_2.detach().data,
            "critic_loss": loss_gen.detach().data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output, 0

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        critic_loss_sum = sum(log.get("critic_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )
        metrics.log_scalar(
            "critic_loss", critic_loss_sum / sample_size / math.log(2), sample_size, round=3
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
