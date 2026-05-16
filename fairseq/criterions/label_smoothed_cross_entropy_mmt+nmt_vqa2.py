# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
import numpy as np


@dataclass
class LabelSmoothedCrossEntropyMMTandNMT_VQA2Config(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    weight1: float = field(
        default=0.1,
        metadata={"help": "vqa loss weight"}
    )
    weight2: float = field(
        default=0.1,
        metadata={"help": "consistency loss weight"}
    )
    strategy: str = field(
        default=" ",
        metadata={"help": "weight2 strategy"}
    )
    current_epoch: int = field(
        default=0,
    )


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
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

def compute_ot_loss(x, y):
    y = y.view(y.size(-1), -1, y.size(0))
    x = x.view(x.size(-1), -1, x.size(0))
    y = F.normalize(y, p=2, dim=0, eps=1e-5)
    x = F.normalize(x, p=2, dim=0, eps=1e-5)
    y = y.transpose(0, 1)
    x = x.transpose(0, 1)
    C1 = cost(x, y)
    weight1 = torch.linalg.norm(x, dim=-1) / torch.linalg.norm(x, dim=-1).sum(dim=-1, keepdim=True)
    res1 = (C1.min(dim=-1)[0] * weight1.detach().clone()).sum()
    C2 = cost(x, y)
    weight2 = torch.linalg.norm(y, dim=-1) / torch.linalg.norm(y, dim=-1).sum(dim=-1,keepdim=True)
    res2 = (C2.min(dim=-1)[0] * weight2.detach().clone()).sum()
    loss = 0.5 * (res1 + res2)
    return loss

def kl_div_loss(lprobs_p, lprobs_q, target, ignore_index=None, reduce=True):
    '''
    Kullback–Leibler divergence between probability distributions
    '''
    loss = F.kl_div(lprobs_p, lprobs_q.exp(), reduction='none')
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        loss.masked_fill_(pad_mask, 0.0)
    else:
        loss = loss.squeeze(-1)
    if reduce:
        loss = loss.sum()
    return loss

@register_criterion(
    "label_smoothed_cross_entropy_mmt_nmt_vqa2", dataclass=LabelSmoothedCrossEntropyMMTandNMT_VQA2Config
)
class LabelSmoothedCrossEntropyMMTandNMT_VQA2Criterion(FairseqCriterion):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            weight1,
            weight2,
            strategy,
            ignore_prefix_size=0,
            report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.weight1 = weight1
        self.weight2 = weight2
        self.strategy = strategy
        # print(self.sentence_avg, self.eps, self.weight)

    def vqa_get_src_lprobs_and_target(self, src_vqa_x, sample):
        src_lprobs = utils.log_softmax(src_vqa_x, dim=-1, onnx_trace=False)
        src_target = sample["src_ans"]
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            src_lprobs = src_lprobs[:, self.ignore_prefix_size:, :].contiguous()
            src_target = src_target[:, self.ignore_prefix_size:].contiguous()
        return src_lprobs.view(-1, src_lprobs.size(-1)), src_target.view(-1)

    def vqa_get_tgt_lprobs_and_target(self, tgt_vqa_x, sample):
        tgt_lprobs = utils.log_softmax(tgt_vqa_x, dim=-1, onnx_trace=False)
        tgt_target = sample["tgt_ans"]
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            tgt_lprobs = tgt_lprobs[:, self.ignore_prefix_size:, :].contiguous()
            tgt_target = tgt_target[:, self.ignore_prefix_size:].contiguous()
        return tgt_lprobs.view(-1, tgt_lprobs.size(-1)), tgt_target.view(-1)

    def compute_vqa_loss(self, src_vqa_x, tgt_vqa_x, sample, reduce):
        src_lprobs, src_target = self.vqa_get_src_lprobs_and_target(src_vqa_x, sample)
        tgt_lprobs, tgt_target = self.vqa_get_tgt_lprobs_and_target(tgt_vqa_x, sample)
        src_loss, src_nll_loss = label_smoothed_nll_loss(
            src_lprobs,
            src_target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        tgt_loss, tgt_nll_loss = label_smoothed_nll_loss(
            tgt_lprobs,
            tgt_target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        loss = src_loss + tgt_loss
        nll_loss = src_nll_loss + tgt_nll_loss

        # print("src_vqa_x size: ",src_vqa_x.size())

        max_length = max(src_vqa_x.size(1), tgt_vqa_x.size(1))  # 获取最长的序列长度
        src_vqa_x = F.pad(src_vqa_x, (0, 0, 0, max_length - src_vqa_x.size(1)))  # 填充到最大长度
        tgt_vqa_x = F.pad(tgt_vqa_x, (0, 0, 0, max_length - tgt_vqa_x.size(1)))

        max_length_t = max(src_target.size(0), tgt_target.size(0))  # 获取最长的序列长度
        src_target = F.pad(src_target, (0,max_length_t - src_target.size(0)))  # 填充到最大长度
        tgt_target = F.pad(tgt_target, (0,max_length_t - tgt_target.size(0)))

#1
        # cons_loss = torch.abs(src_loss - tgt_loss)
#2
        # cons_loss = torch.abs(src_vqa_x - tgt_vqa_x)
        # cons_loss = cons_loss.sum()
#3
        # cos_sim = F.cosine_similarity(src_vqa_x, tgt_vqa_x, dim=-1)  # 在特征维度 (9715) 上计算
        # cons_loss = 1 - cos_sim.mean()  # 最小化 1 - 余弦相似度，确保两者尽量接近
#4
        src_target = src_target.float()
        tgt_target = tgt_target.float()
        src_vqa_x = src_vqa_x.float()
        tgt_vqa_x = tgt_vqa_x.float()

        # Step 1: 计算标签之间的距离
        # 可以使用欧氏距离或余弦相似度
        # target_distance = torch.norm(src_target - tgt_target,p=2)
        target_cos_sim = F.cosine_similarity(src_target, tgt_target, dim=-1)
        target_distance = 1 - target_cos_sim  # 转换为余弦距离

        # Step 2: 计算预测之间的距离
        # pred_distance = torch.norm(src_vqa_x - tgt_vqa_x,p=2)
        pred_cos_sim = F.cosine_similarity(src_vqa_x, tgt_vqa_x, dim=-1)
        pred_distance = 1 - pred_cos_sim  # 转换为余弦距离
        if torch.isnan(pred_distance).any():
            print("NaN detected in pred_distance")
            pred_distance = torch.nan_to_num(pred_distance, nan=0.0)

        # Step 3: 使得预测距离尽可能接近标签距离
        cons_loss = torch.abs(pred_distance - target_distance).sum()
        return loss, nll_loss, cons_loss

    def update_lambda(self, current_epoch):
        if self.strategy == 'linear': #1
            # 线性变化
            lambda_consistency = self.weight2 * ( (current_epoch+30) / 100) # +30:40.22; +100:
        elif self.strategy == 'sigmoid': #2
            # 指数增长
            lambda_consistency = 1 / (1 + np.exp(-0.1 * (current_epoch - 30))) #50: 41.07
        elif self.strategy == 'step': #3
            # 分段变化
            if current_epoch < 21:
                lambda_consistency = 0
            # elif current_epoch >=100:
            #     lambda_consistency = 1
            else:
                lambda_consistency = (current_epoch - 20) / (100 - 20) * self.weight2  #20: 41.16
                # lambda_consistency = self.weight2 #20:40.76
        elif self.strategy == 'step2': #3-2
            if current_epoch < 10: #20: 40.37
                lambda_consistency = 0
            else:
                lambda_consistency = self.weight2
        elif self.strategy == 'combine':
            if current_epoch < 20:
                return 0
            elif current_epoch <= 50:
                return (current_epoch / 50) * self.weight2
            else:
                return self.weight2 * (1 - 0.99 ** (current_epoch - 50))
        else:
            # 默认不变
            lambda_consistency = self.weight2

        return lambda_consistency

    def forward(self, model, sample, current_epoch, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # print("train?",model.training)
        # print("net_input :",sample["net_input"])
        encoder_output, net_output = model(**sample["net_input"])  #net_output
        # print("net_output :", net_output)
        mt_loss, nll_loss, mmt_loss, nmt_loss, kl_loss = self.compute_loss(model, net_output, sample, reduce=reduce)   #mt_loss=(mmt_loss+nmt_loss)/2 + kl_weight·kl_mt_loss
        if self.training and sample["net_input"]["src_query"] is not None and sample["net_input"]["tgt_query"] is not None:
            vqa_loss, vqa_nll_loss, cons_loss = self.compute_vqa_loss(net_output['mmt'][1]["src_vqa_x"], net_output['mmt'][1]["tgt_vqa_x"], sample, reduce=reduce)
        else:
            vqa_loss, vqa_nll_loss, cons_loss = 0, 0, 0

        # nll_loss = vqa_nll_loss * self.weight + nll_loss

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        # loss = vqa_loss * self.weight1 + cons_loss * self.weight2 + mmt_loss
        # 更新lambda_consistency
        if self.training:
            lambda_consistency = self.update_lambda(current_epoch)
        else :
            lambda_consistency = 0
        # print("----------------------------------lambda_consistency = ",lambda_consistency)

        # loss = mt_loss + vqa_loss * self.weight1 + cons_loss * lambda_consistency
        if self.training:
            # loss = mt_loss + vqa_loss * self.weight1 + cons_loss * lambda_consistency
            loss = mt_loss + vqa_loss * self.weight1 + cons_loss * lambda_consistency
        else:
            loss = mmt_loss

        # loss = mmt_loss + self.weight2 * vqa_loss *  cons_loss
        if self.training and sample["net_input"]["src_query"] is not None and sample["net_input"]["tgt_query"] is not None:
            logging_output = {
                "loss": loss.data,
                "mt_loss": mt_loss.data,
                "mmt_loss": mmt_loss.data,
                "nmt_loss": nmt_loss.data,
                "kl_mt_loss": kl_loss.data,
                "vqa_loss": vqa_loss.data,
                "cons_loss": cons_loss.data,
                "nll_loss": nll_loss.data,
                "lambda_consistency": lambda_consistency,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }
        else:
            logging_output = {
                "loss": loss.data,
                "mt_loss": mt_loss.data,
                "mmt_loss": mmt_loss.data,
                "nmt_loss": nmt_loss.data,
                "kl_mt_loss": kl_loss.data,
                "nll_loss": nll_loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

        # return loss, sample_size, logging_output

        return loss, mmt_loss, nmt_loss, vqa_loss, cons_loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
            target = target[:, self.ignore_prefix_size:].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        loss_dict = {}
        mmt = 'mmt' in net_output and net_output['mmt'] is not None
        nmt = 'nmt' in net_output and net_output['nmt'] is not None
        assert mmt and nmt, "You need to specify both mmt and nmt"
        if mmt:
            lprobs_mmt, target = self.get_lprobs_and_target(model, net_output['mmt'], sample)
            loss_dict['mmt'], loss_dict['mmt_nll'] = label_smoothed_nll_loss(
                lprobs_mmt,
                target,
                self.eps,
                ignore_index=self.padding_idx,
                reduce=reduce,
            )
            mmt_loss = loss_dict['mmt']
            nll_loss = loss_dict['mmt_nll']
        if nmt:
            lprobs_nmt, target = self.get_lprobs_and_target(model, net_output['nmt'], sample)
            loss_dict['nmt'], loss_dict['nmt_nll'] = label_smoothed_nll_loss(
                lprobs_nmt,
                target,
                self.eps,
                ignore_index=self.padding_idx,
                reduce=reduce,
            )
            nmt_loss = loss_dict['nmt']
            loss = (mmt_loss + nmt_loss) / 2 if mmt_loss is not None else nmt_loss
            nll_loss = (nll_loss + loss_dict['nmt_nll']) / 2 if nll_loss is not None else loss_dict['nmt_nll']

        self.kl_weight = 0.5

        if mmt and nmt:
            if self.kl_weight:
                kl_mt_loss = kl_div_loss(lprobs_mmt, lprobs_nmt, target)
                loss_dict['kl_mt'] = kl_mt_loss
                loss += self.kl_weight * kl_mt_loss

        return loss, nll_loss, mmt_loss, nmt_loss, kl_mt_loss  # loss=(mmt_loss+nmt_loss)/2 + kl_weight·kl_mt_loss

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
        mt_loss_sum = sum(log.get("mt_loss", 0) for log in logging_outputs)
        mmt_loss_sum = sum(log.get('mmt_loss', 0) for log in logging_outputs)
        nmt_loss_sum = sum(log.get("nmt_loss", 0) for log in logging_outputs)
        kl_loss_sum = sum(log.get("kl_loss", 0) for log in logging_outputs)
        vqa_loss_sum = sum(log.get('vqa_loss', 0) for log in logging_outputs)
        cons_loss_sum = sum(log.get('cons_loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        lambda_consistency_value = sum(log.get("lambda_consistency", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "mt_loss", mt_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "mmt_loss", mmt_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nmt_loss", nmt_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "kl_loss", kl_loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "vqa_loss", vqa_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "cons_loss", cons_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "lambda_consistency",lambda_consistency_value, ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
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