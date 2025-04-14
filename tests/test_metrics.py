# -*- coding: utf-8 -*-
import evaluate
from tabulate import tabulate

cer_metric = evaluate.load("src/metrics/cer")
wer_metric = evaluate.load("src/metrics/wer")

zh_references = ["我喜欢学习人工智能", "今天天气真好", "我们一起去吃火锅吧"]
zh_predictions = ["我喜爱学习人工智能", "今天天气真好", "我们去吃火锅吧"]

en_references = ["I love studying artificial intelligence", "The weather is great today", "Let's go eat hot pot together"]
en_predictions = ["I like studying artificial intelligence", "The weather is great today", "Let's go eat hotpot together"]


def evaluate_samples(predictions, references, lang):
    sample_results = []

    for pred, ref in zip(predictions, references):
        cer = cer_metric.compute(predictions=[pred], references=[ref])
        wer = wer_metric.compute(predictions=[pred], references=[ref])
        sample_results.append({"Language": lang, "Reference": ref, "Prediction": pred, "CER": round(cer, 4), "WER": round(wer, 4)})

    batch_cer = cer_metric.compute(predictions=predictions, references=references)
    batch_wer = wer_metric.compute(predictions=predictions, references=references)

    return sample_results, batch_cer, batch_wer


zh_sample_results, zh_cer_batch, zh_wer_batch = evaluate_samples(zh_predictions, zh_references, "ZH")
en_sample_results, en_cer_batch, en_wer_batch = evaluate_samples(en_predictions, en_references, "EN")
all_sample_results = zh_sample_results + en_sample_results

print("Sample-level CER & WER:")
print(tabulate(all_sample_results, headers="keys", tablefmt="grid"))

print("\nBatch-level Average:")
batch_results = [["ZH", round(zh_cer_batch, 4), round(zh_wer_batch, 4)], ["EN", round(en_cer_batch, 4), round(en_wer_batch, 4)]]
print(tabulate(batch_results, headers=["Language", "Avg CER", "Avg WER"], tablefmt="grid"))
